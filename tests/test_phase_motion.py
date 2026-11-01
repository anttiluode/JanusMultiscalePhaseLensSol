import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase_world import ComplexDirectionalFrame, generate_scene, mse
from phase_motion import (translate_fft,phase_motion,classical_phase_correlation,
                          channel_phase_stats,phase_shift_only)

class PhaseMotionTests(unittest.TestCase):
    def setUp(self):
        self.im=generate_scene(202,64)
        self.frame=ComplexDirectionalFrame(64,4,4)

    def test_translation_inverse(self):
        dx,dy=.41,-.67
        self.assertLess(mse(self.im,translate_fft(translate_fft(self.im,dx,dy),-dx,-dy)),1e-5)

    def test_complex_frame_motion_is_spatial_shift(self):
        # Exact under integer motion (periodic boundary conditions).
        moved=np.roll(self.im,(2,-3),axis=(0,1))
        c=self.frame.analyze(self.im)
        self.assertLess(np.max(abs(self.frame.analyze(moved)-np.roll(c,(2,-3),axis=(1,2)))),1e-9)

    def test_phase_motion_noiseless_subpixel(self):
        truth=np.array([.71,-.43])
        moved=translate_fft(self.im,*truth)
        est=phase_motion(self.frame,self.im,moved)
        self.assertLess(float(np.linalg.norm(est-truth)),.08)

    def test_classical_correlation_noiseless_subpixel(self):
        truth=np.array([.71,-.43])
        moved=translate_fft(self.im,*truth)
        est=classical_phase_correlation(self.im,moved)
        self.assertLess(float(np.linalg.norm(est-truth)),.1)

    def test_phase_only_is_not_an_exact_shift(self):
        d=(.82,-.55)
        moved=translate_fft(self.im,*d)
        rotated=phase_shift_only(self.frame,self.im,*d)
        self.assertGreater(mse(moved,rotated),1e-5)

    def test_scrambled_frequency_geometry_destroys_noiseless_alignment(self):
        true=np.array([.71,-.43])
        moved=translate_fft(self.im,*true)
        correct=phase_motion(self.frame,self.im,moved)
        wrong=phase_motion(self.frame,self.im,moved,scramble_geometry=True)
        self.assertGreater(np.linalg.norm(wrong-true),np.linalg.norm(correct-true)+.1)

    def test_noiseless_shift_is_better_than_no_alignment(self):
        truth=(.81,-.57)
        shifted=translate_fft(self.im,*truth)
        est=phase_motion(self.frame,self.im,shifted)
        self.assertLess(mse(shifted,.5*(shifted+translate_fft(self.im,*est))),
                        mse(shifted,.5*(shifted+self.im)))

if __name__=='__main__':unittest.main()
