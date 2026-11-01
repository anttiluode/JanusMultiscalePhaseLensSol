"""Compact, falsifiable phase-based visual component fitting experiment.

The transform is an *exactly reconstructing redundant complex directional frame*;
it is not a published steerable pyramid implementation or a neural model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ComplexDirectionalFrame:
    size: int = 64
    scales: int = 4
    orientations: int = 6

    def __post_init__(self):
        n = self.size
        kx = np.fft.fftfreq(n)[None, :]
        ky = np.fft.fftfreq(n)[:, None]
        r = np.sqrt(kx*kx + ky*ky)
        ang = np.arctan2(ky, kx)
        # The half-octave-shaped radial responses cover the Fourier plane.
        # The broad lowpass guarantees no holes near DC.  All filters are
        # pointwise normalized so sum_j |H_j(k)|^2 = 1 to machine precision.
        raw = [np.exp(-.5*(r/.045)**4)]
        for s in range(self.scales):
            center = .055 * 2**s
            radial = np.exp(-.5*(np.log2(np.maximum(r, 1e-10)/center)/.62)**2)
            for o in range(self.orientations):
                a = 2*np.pi*o/self.orientations
                delta = np.arctan2(np.sin(ang-a),np.cos(ang-a))
                # Directed filters, not even-symmetric: analysis is complex.
                orient = np.exp(-.5*(delta/.72)**2)
                raw.append(radial * orient)
        filters = np.stack(raw)
        norm = np.sqrt(np.sum(filters**2,axis=0))
        self.filters = filters / np.maximum(norm, 1e-15)
        self.channels = self.filters.shape[0]

    def analyze(self, image):
        assert image.shape == (self.size,self.size)
        return np.fft.ifft2(np.fft.fft2(image)[None,:,:]*self.filters,axes=(-2,-1))

    def synthesize(self, coeff):
        assert coeff.shape == (self.channels,self.size,self.size)
        return np.fft.ifft2(np.sum(np.fft.fft2(coeff,axes=(-2,-1))*self.filters,axis=0)).real

    def frame_error(self):
        return float(np.max(np.abs(np.sum(self.filters**2,axis=0)-1)))


def generate_scene(seed, size=64, domain='familiar'):
    """Independent deterministic scenes: held-out seeds, not re-used target frames."""
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[0:size,0:size].astype(float)
    xx=(xx+0.5)/size;yy=(yy+0.5)/size
    image=np.full((size,size),rng.uniform(.15,.5))
    if domain == 'familiar':
        image += rng.uniform(-.16,.16)*(xx-.5) + rng.uniform(-.16,.16)*(yy-.5)
        for _ in range(rng.integers(3,8)):
            kind=rng.integers(3);cx,cy=rng.uniform(.1,.9,2);amp=rng.uniform(-.5,.5)
            if kind == 0:
                rr=rng.uniform(.035,.24)
                m=((xx-cx)**2+(yy-cy)**2) < rr**2
            elif kind == 1:
                w,h=rng.uniform(.06,.30,2)
                m=(np.abs(xx-cx)<w)&(np.abs(yy-cy)<h)
            else:
                a=rng.uniform(0,np.pi);width=rng.uniform(.01,.08)
                m=np.abs((xx-cx)*np.cos(a)+(yy-cy)*np.sin(a))<width
            image += amp*m.astype(float)
        for _ in range(2):
            cx,cy=rng.uniform(.1,.9,2);std=rng.uniform(.04,.20)
            image += rng.uniform(-.25,.25)*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*std**2))
    elif domain == 'unrelated':
        # A diagnostic out-of-distribution regime, distinct from training.
        a=rng.uniform(0,np.pi);freq=rng.uniform(13,32)
        phase=xx*np.cos(a)+yy*np.sin(a)
        image += .23*np.sign(np.sin(2*np.pi*freq*phase+.2*np.sin(15*yy)))
        image += .20*np.sign(np.sin(2*np.pi*rng.uniform(8,19)*xx)*np.sin(2*np.pi*rng.uniform(8,19)*yy))
    else:
        raise ValueError(domain)
    return np.clip(image,0,1)


def extraction_grid(size=64,patch=8,stride=4):
    return [(y,x) for y in range(0,size-patch+1,stride) for x in range(0,size-patch+1,stride)]


def extract_patches(coeff, patch=8, stride=4):
    c,n,_=coeff.shape
    loc=extraction_grid(n,patch,stride)
    return np.stack([coeff[:,y:y+patch,x:x+patch].reshape(-1) for y,x in loc])


def patches_to_coeff(patches,frame,patch=8,stride=4):
    n=frame.size;c=frame.channels
    out=np.zeros((c,n,n),np.complex128)
    denom=np.zeros((n,n))
    # Positive overlap-add weights (no corner holes).
    w=np.outer(np.hanning(patch+2)[1:-1],np.hanning(patch+2)[1:-1])
    for z,(y,x) in zip(patches,extraction_grid(n,patch,stride)):
        out[:,y:y+patch,x:x+patch]+=z.reshape(c,patch,patch)*w[None,:,:]
        denom[y:y+patch,x:x+patch]+=w
    return out/np.maximum(denom[None,:,:],1e-12)


class PhaseVocabulary:
    """Complex spherical dictionary; fit one complex gain and one identity per patch.

    Complexity per scene: number_of_patches complex gains + integer IDs, not
    freely varying phases for every pixel and every coefficient.
    """
    def __init__(self,n_atoms=32,seed=0,iterations=10):
        self.n_atoms=n_atoms;self.seed=seed;self.iterations=iterations

    def fit(self, patch_matrix):
        x=np.asarray(patch_matrix,dtype=np.complex128)
        rng=np.random.default_rng(self.seed)
        self.train_rms=np.sqrt(np.mean(np.abs(x)**2,axis=0)+1e-7)
        # Equalize frequency/channel energy. It also suppresses coarse DC dominance.
        self.train_rms=np.maximum(self.train_rms,np.max(self.train_rms)*.04)
        x=x/self.train_rms[None,:]
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-10)
        ids=rng.choice(len(x),size=self.n_atoms,replace=False)
        d=x[ids].copy()
        for _ in range(self.iterations):
            sim=x@d.conj().T
            choice=np.argmax(np.abs(sim),axis=1)
            for k in range(self.n_atoms):
                mask=choice==k
                if not np.any(mask):
                    d[k]=x[rng.integers(len(x))];continue
                # Align all examples' global phases before finding their centroid.
                phases=np.conj(sim[mask,k])/np.maximum(np.abs(sim[mask,k]),1e-10)
                centroid=np.sum(x[mask]*phases[:,None],axis=0)
                d[k]=centroid/np.maximum(np.linalg.norm(centroid),1e-10)
        self.atoms=d
        return self

    def infer(self, patches,phase=True, scrambled=False, seed=19):
        x=np.asarray(patches,dtype=np.complex128)/self.train_rms[None,:]
        d=self.atoms
        if scrambled:
            rng=np.random.default_rng(seed)
            d=np.abs(d)*np.exp(1j*rng.uniform(-np.pi,np.pi,d.shape))
            d=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-10)
        sim=x@d.conj().T
        if phase:
            idx=np.argmax(np.abs(sim),axis=1)
            gains=sim[np.arange(len(x)),idx]
        else:
            idx=np.argmax(np.maximum(sim.real,0),axis=1)
            gains=np.maximum(sim.real[np.arange(len(x)),idx],0)
        fit=(gains[:,None]*d[idx])*self.train_rms[None,:]
        return fit,idx,gains


def soft_threshold_coeff(coeff, threshold, frame, only_high=True):
    result=coeff.copy()
    start=1 if only_high else 0
    z=result[start:]
    mag=np.abs(z)
    result[start:]=z*np.maximum(0,1-threshold/np.maximum(mag,1e-10))
    return frame.synthesize(result)


def mse(x,y):
    return float(np.mean((np.asarray(x)-np.asarray(y))**2))
