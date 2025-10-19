import nbimporter
import unittest
import cv2
import numpy as np
from utils import *
import part1 as module
import pdb

class TestFilters(unittest.TestCase):
    def setUp(self):
        
        self.kernel = np.array([
                    [1, 0, -1], 
                    [2, 0, -2],
                    [1, 0, -1]
                ])
        
        self.image = cv2.imread("main.jpg", cv2.IMREAD_GRAYSCALE)
        
        self.reference = cv2.filter2D(self.image, ddepth=cv2.CV_32F, kernel= np.flip(self.kernel), borderType=cv2.BORDER_CONSTANT)
        
        writeDoubleImage(self.reference, "ref.jpg")

    def test_filter_scipy_convolve2d(self):
        
        out = module.filter_scipy_convolve2d(self.image, self.kernel)
        err = out - self.reference
        norm = np.linalg.norm(err, 'fro')
                        
        self.assertTrue(norm<1e-3, "Matrices are not close!")
        
    def test_filter_numpy_for_loop(self):
        
        out = module.filter_numpy_for_loop(self.image, self.kernel)
        err = out - self.reference
        norm = np.linalg.norm(err, 'fro')
                        
        self.assertTrue(norm<1e-3, "Matrices are not close!")
        
    def test_filter_torch(self):
        
        out = module.filter_torch(self.image, self.kernel)
        err = out - self.reference
        norm = np.linalg.norm(err, 'fro')/err.shape[0]/err.shape[1]
                        
        self.assertTrue(norm<1e-3, "Matrices are not close!")
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestFilters("test_filter_scipy_convolve2d"))
    suite.addTest(TestFilters("test_filter_numpy_for_loop"))
    suite.addTest(TestFilters("test_filter_torch"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
