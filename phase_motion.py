"""Experiment 002: measure geometric phase prediction, without privileged motion.

The complex frame is a redundant linear transform. Moving a scene translates its
complex coefficients; phase rotation alone is a *narrow-band approximation*, not
an exact shift for localized edges. We test this distinction rather than assume it.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import shift as ndshift
from scipy.optimize import minimize
from phase_world import ComplexDirectionalFrame


def translate_fft(image, dx, dy):
    """Exact circular subpixel translation of a sampled, bandlimited 2-D image."""
    n=image.shape[0]
    ky,kx=np.meshgrid(np.fft.fftfreq(n)*2*np.pi,np.fft.fftfreq(n)*2*np.pi,indexing='ij')
    return np.fft.ifft2(np.fft.fft2(image)*np.exp(-1j*(kx*dx+ky*dy))).real


def channel_phase_stats(frame,previous,current):
    """Temporal cross phase and data-adaptive frequency centroid by channel.

    The phase of the global coefficient cross product is a circular weighted
    average of Fourier-phase advances *within* each channel, not exactly the
    advance at its nominal centre. Centroids approximate small shifts.
    """
    a=frame.analyze(previous)[1:]
    b=frame.analyze(current)[1:]
    cross=np.sum(b*np.conj(a),axis=(-2,-1))
    energy_a=np.sum(abs(a)**2,axis=(-2,-1))
    energy_b=np.sum(abs(b)**2,axis=(-2,-1))
    coherence=np.abs(cross)/np.maximum(np.sqrt(energy_a*energy_b),1e-12)
    n=frame.size
    ky,kx=np.meshgrid(2*np.pi*np.fft.fftfreq(n),2*np.pi*np.fft.fftfreq(n),indexing='ij')
    power=(abs(np.fft.fft2(previous))**2)[None] * frame.filters[1:]**2
    denom=power.sum(axis=(-2,-1))
    k=np.stack([(power*kx).sum(axis=(-2,-1))/np.maximum(denom,1e-10),
                (power*ky).sum(axis=(-2,-1))/np.maximum(denom,1e-10)],axis=-1)
    # Ignore channels carrying barely any signal; limit the contribution of strong DC leakage.
    w=np.sqrt(energy_a*energy_b)*coherence
    return cross,k,w,coherence


def phase_motion(frame,previous,current,search=1.5,step=.25,refine=True,scramble_geometry=False):
    """Infer translation (dx,dy) using channel cross phases and centroids.

    No ground-truth motion, labels, future frames, or clean reference are used.
    For an actual translation, phase_{new}-phase_{old} ~= -k dot d.
    """
    cross,k,w,_=channel_phase_stats(frame,previous,current)
    if scramble_geometry:
        # Fixed wrong association between directional responses and wavenumbers.
        k = k[np.random.default_rng(23).permutation(len(k))]
    unit=cross/np.maximum(np.abs(cross),1e-12)
    w=w/np.maximum(w.sum(),1e-12)
    candidates=np.arange(-search,search+step*.25,step)
    dx,dy=np.meshgrid(candidates,candidates)
    grid=np.stack([dx.ravel(),dy.ravel()],axis=-1)
    def score_many(displacements):
        pred=np.exp(-1j*(displacements@k.T))
        return np.real((pred*np.conj(unit)[None,:])@w)
    scores=score_many(grid)
    initial=grid[np.argmax(scores)]
    if not refine:
        return initial
    def objective(v):return -float(score_many(np.asarray(v).reshape(1,2))[0])
    result=minimize(objective,initial,method='L-BFGS-B',bounds=[(-search,search)]*2,
                    options={'maxiter':35,'ftol':1e-11})
    return result.x


def phase_shift_only(frame,reference,dx,dy):
    """Approximate motion: rotate channel coefficients *at fixed pixels*.

    Predicted output is not generally a translation: amplitude envelopes stay
    where they were. This is a falsification control for the waveguide metaphor.
    """
    coeff=frame.analyze(reference)
    n=frame.size
    ky,kx=np.meshgrid(2*np.pi*np.fft.fftfreq(n),2*np.pi*np.fft.fftfreq(n),indexing='ij')
    power=(abs(np.fft.fft2(reference))**2)[None]*frame.filters**2
    den=power.sum(axis=(-2,-1))
    kxc=(power*kx).sum(axis=(-2,-1))/np.maximum(den,1e-10)
    kyc=(power*ky).sum(axis=(-2,-1))/np.maximum(den,1e-10)
    pred=coeff*np.exp(-1j*(kxc*dx+kyc*dy))[:,None,None]
    return frame.synthesize(pred)


def classical_phase_correlation(reference,current,search=1.5):
    """Classical global phase correlation, with subpixel DFT-plane fitting.

    Uses the same noisy pair; search restricted to the known benchmark range
    in the *same way* as phase_motion. Solves the Fourier shift relation.
    """
    ft=np.fft.fft2(reference)
    gt=np.fft.fft2(current)
    cross=gt*np.conj(ft)
    unit=cross/np.maximum(np.abs(cross),1e-8)
    n=reference.shape[0]
    ky,kx=np.meshgrid(2*np.pi*np.fft.fftfreq(n),2*np.pi*np.fft.fftfreq(n),indexing='ij')
    # Phase-only cross-power: maximizing real inverse Fourier transform.
    candidates=np.arange(-search,search+1e-8,.25)
    dx,dy=np.meshgrid(candidates,candidates)
    grid=np.stack([dx.ravel(),dy.ravel()],axis=-1)
    flatk=np.stack([kx.ravel(),ky.ravel()],axis=-1)
    v=unit.ravel()
    score=np.real(np.exp(1j*(grid@flatk.T))@v)/len(v)
    initial=grid[np.argmax(score)]
    def loss(z):return -float(np.real(np.sum(v*np.exp(1j*(flatk@z))))/len(v))
    result=minimize(loss,initial,method='L-BFGS-B',bounds=[(-search,search)]*2,options={'maxiter':40})
    return result.x


def pixel_translation_fit(reference,current,search=1.5):
    """Strong baseline: optimize shifted noisy reference against current in pixel MSE.

    This is expensive but explicitly separates benefits of complex phase access
    from ordinary geometric image registration.
    """
    # Circular FFT shift matches the simulator, not interpolation artifacts.
    # No clean images enter this registration baseline.
    x=np.fft.fft2(reference)
    n=reference.shape[0]
    ky,kx=np.meshgrid(2*np.pi*np.fft.fftfreq(n),2*np.pi*np.fft.fftfreq(n),indexing='ij')
    candidates=np.arange(-search,search+1e-8,.50)
    def loss(d):
        pred=np.fft.ifft2(x*np.exp(-1j*(kx*d[0]+ky*d[1]))).real
        return float(np.mean((pred-current)**2))
    initial=min(([dx,dy] for dx in candidates for dy in candidates),key=loss)
    return minimize(loss,initial,method='L-BFGS-B',bounds=[(-search,search)]*2,
                    options={'maxiter':50,'ftol':1e-11}).x
