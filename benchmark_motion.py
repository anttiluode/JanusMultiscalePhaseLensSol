"""Experiment 002: phase geometry under held-out subpixel translations.

No trainable parameters, no clean images, ground-truth motion, or class labels
are used for inference. The synthetic generator supplies truth ONLY for scoring.
The original learned vocabulary remains untouched; this isolates the proposed
geometric phase rule before trying to mix it with learned atoms.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from phase_world import ComplexDirectionalFrame,generate_scene,mse
from phase_motion import (phase_motion,classical_phase_correlation,
                          pixel_translation_fit,translate_fft,phase_shift_only)


def evaluate(seed=71,n_familiar=18,n_unrelated=10,scales=(0.06,0.12,0.22)):
    frame=ComplexDirectionalFrame(64,4,4)
    records=[];vis=[]
    rng=np.random.default_rng(seed)
    for domain,count in [('familiar',n_familiar),('unrelated',n_unrelated)]:
        for k in range(count):
            scene_seed=int(rng.integers(200000,900000))
            clean=generate_scene(scene_seed,64,domain=domain)
            d=rng.uniform(-.98,.98,size=2)
            target=translate_fft(clean,*d)
            for sig in scales:
                nrng=np.random.default_rng(scene_seed+int(sig*1e5))
                prev=clean+nrng.normal(0,sig,clean.shape)
                current=target+nrng.normal(0,sig,target.shape)
                ests={
                    'frame_phase_channels':phase_motion(frame,prev,current),
                    'scrambled_phase_geometry':phase_motion(frame,prev,current,scramble_geometry=True),
                    'classical_phase_correlation':classical_phase_correlation(prev,current),
                    'pixel_registration':pixel_translation_fit(prev,current),
                    'oracle_translation':d,
                    'no_alignment':np.zeros(2),
                }
                baselines={'raw':current,'gaussian':gaussian_filter(current,.7)}
                for name,pred in baselines.items():
                    records.append(dict(domain=domain,seed=scene_seed,sigma=sig,
                                        method=name,dx_error='',dy_error='',motion_error='',
                                        mse=mse(target,pred)))
                for name,estimate in ests.items():
                    aligned=translate_fft(prev,*estimate)
                    pred=.5*(aligned+current)
                    records.append(dict(domain=domain,seed=scene_seed,sigma=sig,
                                        method=name,dx_error=float(estimate[0]-d[0]),
                                        dy_error=float(estimate[1]-d[1]),
                                        motion_error=float(np.linalg.norm(estimate-d)),
                                        mse=mse(target,pred)))
                # Evaluate whether *phase rotation alone* is an actual image motion.
                phase_rotated=phase_shift_only(frame,clean,*d)
                records.append(dict(domain=domain,seed=scene_seed,sigma=sig,
                                    method='phase_rotation_only_clean',
                                    dx_error='',dy_error='',motion_error='',
                                    mse=mse(target,phase_rotated)))
                if len(vis)<2 and k==0 and sig==.12:
                    vis.append((domain,clean,target,current,
                                .5*(translate_fft(prev,*ests['frame_phase_channels'])+current),
                                .5*(translate_fft(prev,*ests['classical_phase_correlation'])+current),
                                .5*(translate_fft(prev,*d)+current),phase_rotated,d))
    return records,vis


def summarize(rows):
    result=[]
    for domain in sorted(set(r['domain'] for r in rows)):
        for sigma in sorted(set(r['sigma'] for r in rows)):
            for method in sorted(set(r['method'] for r in rows)):
                sub=[r for r in rows if r['domain']==domain and r['sigma']==sigma and r['method']==method]
                e=np.array([r['mse'] for r in sub])
                motion=[r['motion_error'] for r in sub if r['motion_error']!='']
                result.append(dict(domain=domain,sigma=sigma,method=method,n=len(sub),
                                   mean_mse=float(e.mean()),sd_mse=float(e.std(ddof=1)),
                                   mean_motion_error=float(np.mean(motion)) if motion else None,
                                   sd_motion_error=float(np.std(motion,ddof=1)) if motion else None))
    return result


def plot_examples(vis,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(vis),6,figsize=(15,5.8),squeeze=False)
    for row,(domain,clean,target,current,p,classic,oracle,phase_rotated,d) in enumerate(vis):
        for ax,label,image in zip(axs[row],['Input scene','True shifted','Noisy observation',
                                         'Phase-channel estimate','Classical phase corr.','Oracle motion'],
                                  [clean,target,current,p,classic,oracle]):
            ax.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(label,fontsize=9);ax.axis('off')
        axs[row,0].set_ylabel(f'{domain}\nshift ({d[0]:.2f},{d[1]:.2f})',fontsize=10)
    fig.tight_layout();fig.savefig(out,dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/motion_main.json')
    p.add_argument('--familiar',type=int,default=18);p.add_argument('--unrelated',type=int,default=10)
    p.add_argument('--seed',type=int,default=71)
    args=p.parse_args()
    start=time.perf_counter()
    rows,vis=evaluate(args.seed,args.familiar,args.unrelated)
    results=dict(title='Experiment 002: geometric phase and subpixel motion',
                 seed=args.seed,n_familiar=args.familiar,n_unrelated=args.unrelated,
                 noise_scales=[.06,.12,.22],size=64,frame_channels=17,
                 known_max_abs_shift=.98,method_search_range=1.5,
                 result=summarize(rows),seconds=time.perf_counter()-start,
                 warnings=['Motion range is known to all registration algorithms.',
                           'No learning: this test isolates the geometric motion rule.',
                           'Scene shift is globally circular and generated by the Fourier shift theorem.',
                           'Synthetic grayscale still scenes, without occlusion or lighting change.',
                           'Oracle translation sees true shift as a ceiling only.',
                           'Classical phase correlation and pixel registration share both observations.',
                           'Gaussian smoothing .7 is an untuned reference, not a best possible single-image denoiser.'])
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2))
    with path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot_examples(vis,path.parent/'motion_examples.png')
    print(json.dumps(dict(results=results['result'],seconds=results['seconds']),indent=2))


if __name__=='__main__':main()
