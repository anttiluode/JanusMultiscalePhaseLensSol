import unittest
import numpy as np
from phase_world import (ComplexDirectionalFrame,PhaseVocabulary,extract_patches,
                         patches_to_coeff,generate_scene)

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=ComplexDirectionalFrame(size=32,scales=3,orientations=4)

    def test_exact_analysis_synthesis(self):
        rng=np.random.default_rng(10)
        for _ in range(3):
            x=rng.normal(size=(32,32))
            self.assertLess(np.max(np.abs(self.f.synthesize(self.f.analyze(x))-x)),1e-10)
        self.assertLess(self.f.frame_error(),1e-12)

    def test_complex_directionality(self):
        im=generate_scene(42,size=32)
        co=self.f.analyze(im)
        self.assertGreater(np.max(np.abs(co.imag)),1e-4)

    def test_patch_overlap_identity(self):
        rng=np.random.default_rng(8)
        z=rng.normal(size=(self.f.channels,32,32))+1j*rng.normal(size=(self.f.channels,32,32))
        patches=extract_patches(z)
        reconstructed=patches_to_coeff(patches,self.f)
        self.assertLess(np.max(np.abs(z-reconstructed)),1e-10)

    def test_global_complex_phase_optimization(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(30,10))+1j*rng.normal(size=(30,10))
        vocab=PhaseVocabulary(8,iterations=4).fit(x)
        patch=(2*np.exp(1.3j)*vocab.atoms[0]*vocab.train_rms)[None,:]
        fit,index,gain=vocab.infer(patch)
        self.assertLess(np.linalg.norm(fit-patch),1e-8)
        fit_real,_,_=vocab.infer(patch,phase=False)
        self.assertGreater(np.linalg.norm(fit_real-patch),1e-5)

    def test_motion_is_scale_and_orientation_dependent(self):
        n=32
        yy,xx=np.mgrid[0:n,0:n]
        k=4
        wave=np.cos(2*np.pi*k*xx/n)
        a=self.f.analyze(wave)
        b=self.f.analyze(np.roll(wave,1,axis=1))
        # Direction-0 filter near +kx isolates an analytic sinusoid.
        j=1
        actual=np.angle(b[j,7,7]/a[j,7,7])
        expected=-2*np.pi*k/n
        self.assertLess(abs(np.angle(np.exp(1j*(actual-expected)))),.02)
        self.assertLess(np.max(np.abs(b-np.roll(a,1,axis=2))),1e-10)

    def test_unrelated_generator_is_distinct(self):
        a=generate_scene(777,size=32,domain='familiar')
        b=generate_scene(777,size=32,domain='unrelated')
        self.assertGreater(np.mean((a-b)**2),.005)

if __name__=='__main__':unittest.main()
