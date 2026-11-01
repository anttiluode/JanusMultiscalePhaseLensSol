"""Frozen-world held-out test. No target is used to change dictionary weights."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from phase_world import (ComplexDirectionalFrame,PhaseVocabulary,generate_scene,
                         extract_patches,patches_to_coeff,mse,soft_threshold_coeff)

SIZE=64
TRAIN_SEEDS=range(1000,1032)
VALID_SEEDS=range(3000,3008)
TEST_SEEDS=range(9000,9024)
ALIEN_SEEDS=range(19000,19012)
NOISE_SCALES=(.12,.22)


def frames(seeds,domain='familiar'):
    return [generate_scene(int(s),SIZE,domain) for s in seeds]


def noisy_image(clean,sigma,seed):
    rng=np.random.default_rng(seed)
    return clean+rng.normal(0,sigma,clean.shape)


def wiener_prior(train):
    ft=np.fft.fft2(np.asarray(train)-np.asarray(train).mean(axis=(1,2),keepdims=True),axes=(-2,-1))
    return np.mean(np.abs(ft)**2,axis=0)


def wiener(noisy,signal_power,sigma):
    noise_power=noisy.size*sigma**2
    weights=signal_power/(signal_power+noise_power)
    return np.fft.ifft2(np.fft.fft2(noisy-noisy.mean())*weights).real+noisy.mean()


def infer_world(frame,vocab,obs,blend=0,phase=True,scrambled=False):
    coeff=frame.analyze(obs)
    parts=extract_patches(coeff)
    fit, ids, gains=vocab.infer(parts,phase=phase,scrambled=scrambled)
    recon=frame.synthesize(patches_to_coeff(fit,frame))
    # blend parameter chosen only on validation, not per test image
    return (1-blend)*recon + blend*obs, {
        'atom_ids': ids,
        'fraction_used':len(np.unique(ids))/vocab.n_atoms,
        'mean_phase':float(np.angle(gains).mean()),
        'observation_residual':mse(recon,obs),
    }


def train_world(frame,train,n_atoms=24,seed=22):
    patches=[extract_patches(frame.analyze(im)) for im in train]
    x=np.concatenate(patches)
    rng=np.random.default_rng(seed)
    ids=rng.choice(len(x),size=min(len(x),1000),replace=False)
    return PhaseVocabulary(n_atoms=n_atoms,seed=seed,iterations=8).fit(x[ids])


def tune(frame,vocab,validation,prior,sigma=.17):
    candidates={'gaussian':[.45,.7,1.0,1.3],
                'shrink':[.006,.015,.03,.06,.1],
                'world_blend':[0,.15,.30,.50,.75,1.]}
    scores={k:[] for k in candidates}
    for key,vals in candidates.items():
        for val in vals:
            losses=[]
            for i,clean in enumerate(validation):
                noisy=noisy_image(clean,sigma,70000+i)
                if key=='gaussian':pred=gaussian_filter(noisy,val)
                elif key=='shrink':pred=soft_threshold_coeff(frame.analyze(noisy),val,frame)
                else:pred,_=infer_world(frame,vocab,noisy,blend=val)
                losses.append(mse(clean,pred))
            scores[key].append(round(float(np.mean(losses)),9))
    best={k:float(candidates[k][int(np.argmin(scores[k]))]) for k in candidates}
    return best, {'candidates':candidates,'validation_mse':scores,'selection_noise':sigma}


def evaluate(frame,vocab,train,params,domain='familiar',seeds=TEST_SEEDS,noise_scales=NOISE_SCALES):
    out=[]
    prior=wiener_prior(train)
    for sigma in noise_scales:
        for index,seed in enumerate(seeds):
            clean=generate_scene(int(seed),SIZE,domain)
            obs=noisy_image(clean,sigma,500000+int(seed)+int(sigma*1000))
            preds={
                'raw':obs,
                'gaussian':gaussian_filter(obs,params['gaussian']),
                'wiener_known_noise':wiener(obs,prior,sigma),
                'complex_shrink':soft_threshold_coeff(frame.analyze(obs),params['shrink'],frame),
            }
            # Same frozen dictionary and blending for all three fitting arms.
            for key,enabled,scrambled in [('vocabulary_phase',True,False),('vocabulary_no_phase',False,False),('vocabulary_scrambled',True,True)]:
                preds[key],_=infer_world(frame,vocab,obs,blend=params['world_blend'],phase=enabled,scrambled=scrambled)
            for key,pred in preds.items():
                out.append({'domain':domain,'seed':int(seed),'noise_sigma':sigma,
                            'method':key,'mse':mse(clean,pred),
                            'mae':float(np.mean(np.abs(clean-pred)))})
    return out


def aggregates(rows):
    summaries=[]
    for domain in sorted(set(r['domain'] for r in rows)):
        for sigma in sorted(set(r['noise_sigma'] for r in rows)):
            for method in sorted(set(r['method'] for r in rows)):
                subs=[r['mse'] for r in rows if r['domain']==domain and r['noise_sigma']==sigma and r['method']==method]
                summaries.append({'domain':domain,'noise_sigma':sigma,'method':method,
                                  'mean_mse':float(np.mean(subs)),
                                  'sd_mse':float(np.std(subs,ddof=1)),'n':len(subs)})
    return summaries


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/main.json')
    p.add_argument('--atoms',type=int,default=24)
    args=p.parse_args()
    started=time.perf_counter()
    np.random.seed(1)
    frame=ComplexDirectionalFrame(SIZE,scales=4,orientations=4)
    training=frames(TRAIN_SEEDS)
    valid=frames(VALID_SEEDS)
    vocab=train_world(frame,training,args.atoms)
    params,tuning=tune(frame,vocab,valid,wiener_prior(training))
    rows=(evaluate(frame,vocab,training,params,'familiar',TEST_SEEDS)
          +evaluate(frame,vocab,training,params,'unrelated',ALIEN_SEEDS))
    result={
        'description':'Frozen complex directional frame and complex prototype fitting. One phase+amplitude gain per overlapping patch.',
        'training_seeds':list(TRAIN_SEEDS),'validation_seeds':list(VALID_SEEDS),
        'test_seeds':list(TEST_SEEDS),'unrelated_seeds':list(ALIEN_SEEDS),
        'n_atoms':args.atoms,'n_channels':frame.channels,'patch_size':8,'stride':4,
        'frame_partition_error':frame.frame_error(),
        'validation_selected':params,'validation':tuning,
        'results':aggregates(rows),'time_seconds':time.perf_counter()-started,
        'limitations':['synthetic grayscale images','known additive Gaussian noise sigma for Wiener baseline',
                       'frame masks are frequency-directional and exactly reconstructing but not a standard steerable-pyramid implementation',
                       'prototype choice is independently fit per overlapping patch; no temporal dynamics or learned branch growth',
                       'blend is selected on held-out validation at sigma .17 and shared by all methods'],
    }
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    with path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'validation_selected':params,'test_results':result['results'],'seconds':result['time_seconds']},indent=2))

if __name__=='__main__':main()
