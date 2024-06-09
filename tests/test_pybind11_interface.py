import unittest

import numpy as np

from surface_roughness import (
    Surface, 
    SampleWindow
)

from surface_roughness.roughness_impl import (
    _cppTINBasedRoughness,
    _cppTINBasedRoughness_Settings_impl,
    _cppDirectionalRoughness
)
from surface_roughness.sampling import RoughnessMap

class TestDirectionalSetting(unittest.TestCase):
    def setUp(self):
        self.surface = Surface('tests/example_surface.stl')
        self.surface.preprocess()
        self.points = self.surface.points
        self.triangles = self.surface.triangles
        self.area = self.surface.area
        self.settings = _cppTINBasedRoughness_Settings_impl()
    
    def test_setting_initialize(self):
        self.assertAlmostEqual(72,self.settings['n_az'])
        self.assertAlmostEqual(0,self.settings['az_offset'])
        self.assertAlmostEqual(200,self.settings['min_triangles'])

    def test_setting_set(self):
        self.settings['n_az'] = 38
        self.assertAlmostEqual(38,self.settings['n_az'])

class TestTINBasedRoughness(unittest.TestCase):
    def setUp(self):
        self.surface = Surface('tests/example_surface.stl')
        self.surface.preprocess()
        self.cppimpl = _cppTINBasedRoughness(self.surface.points, self.surface.triangles)
        self.cppimpl.evaluate()        

    def test_setting_initialize(self):
        self.assertAlmostEqual(72,self.cppimpl.settings['n_az'])
        self.assertAlmostEqual(0,self.cppimpl.settings['az_offset'])
        self.assertAlmostEqual(200,self.cppimpl.settings['min_triangles'])

    def test_result_keys(self):
        self.assertListEqual(
            sorted([
                'delta*_t',
                'delta_t',
                'az',
                'n_tri'
            ]),
            sorted(self.cppimpl.impl.result_keys())
        )
    
    def test_final_orientation(self):
        vec = self.cppimpl.final_orientation
        print(vec)
        test_vec = [0,0,1]
        for i in range(3):
            self.assertAlmostEqual(test_vec[i],vec[i])
        
    def test_min_bounds(self):
        minvec = self.cppimpl.min_bounds
        print(minvec)

        test_min = [
            3.72695227,
            7.00697149,
            -0.95610574]

        for i in range(3):
            self.assertAlmostEqual(test_min[i], minvec[i])
        
    def test_max_bounds(self):
        maxvec = self.cppimpl.max_bounds
        print(maxvec)
        test_max = [
            25.42814906,
            34.59745339,
            1.00586474]
        for i in range(3):
            self.assertAlmostEqual(test_max[i], maxvec[i])
            
    def test_area(self):
        self.assertAlmostEqual(self.cppimpl.total_area, self.surface.area)
    
    def test_result(self):
        self.surface.evaluate_delta_t(impl='py')
        pydelta_t = self.surface.delta_t('delta_t')
        cppdelta_t = np.array(self.cppimpl['delta_t'])[:,0]
        
        self.assertEqual(len(cppdelta_t), 72)
        
        for pyresult, cppresult in zip(pydelta_t, cppdelta_t):
            self.assertAlmostEqual(pyresult, cppresult)
            
        az = np.array(self.cppimpl['az'])[:,0]
        for i in range(72):
            self.assertAlmostEqual(np.radians(i*5),az[i])

class TestDirectionalRoughness(unittest.TestCase):
    def setUp(self):
        self.surface = Surface('tests/example_surface.stl')
        self.surface.preprocess()
        self.cppimpl = _cppDirectionalRoughness(self.surface.points, self.surface.triangles)
        self.cppimpl.evaluate()        

    def test_setting_initialize(self):
        self.assertAlmostEqual(72,self.cppimpl.settings['n_az'])
        self.assertAlmostEqual(0,self.cppimpl.settings['az_offset'])
        self.assertAlmostEqual(200,self.cppimpl.settings['min_triangles'])

    def test_result_keys(self):
        self.assertListEqual(
            sorted([
                'c',
                'theta_max',
                'a0',
                'gof',
                'dip_bin_data',
                'thetamax_cp1',
                'az',
                'n_tri'
            ]),
            sorted(self.cppimpl.impl.result_keys())
        )
    
    def test_final_orientation(self):
        vec = self.cppimpl.final_orientation
        print(vec)
        test_vec = [0,0,1]
        for i in range(3):
            self.assertAlmostEqual(test_vec[i],vec[i])
        
    def test_min_bounds(self):
        minvec = self.cppimpl.min_bounds
        print(minvec)

        test_min = [
            3.72695227,
            7.00697149,
            -0.95610574]

        for i in range(3):
            self.assertAlmostEqual(test_min[i], minvec[i])
        
    def test_max_bounds(self):
        maxvec = self.cppimpl.max_bounds
        print(maxvec)
        test_max = [
            25.42814906,
            34.59745339,
            1.00586474]
        for i in range(3):
            self.assertAlmostEqual(test_max[i], maxvec[i])
            
    def test_area(self):
        print(self.surface.area)
        self.assertAlmostEqual(self.cppimpl.total_area, self.surface.area)
        
    
    def test_result(self):
        self.surface.evaluate_thetamax_cp1(impl='py')
        pythetamax_cp1 = np.array(self.surface.thetamax_cp1('thetamax_cp1'))[:,0]
        cppthetamax_cp1 = np.array(self.cppimpl['thetamax_cp1'])[:,0]
        
        self.assertEqual(len(cppthetamax_cp1), 72)
        
        for i,(pyresult, cppresult) in enumerate(zip(pythetamax_cp1, cppthetamax_cp1)):
            self.assertAlmostEqual(pyresult, cppresult, msg=f'at {i}')
            
        az = np.array(self.cppimpl['az'])[:,0]
        for i in range(72):
            self.assertAlmostEqual(np.radians(i*5),az[i],msg=f'at {i}')

class TestRoughnessMap(unittest.TestCase):
    def setUp(self):
        self.window = SampleWindow(is_circle=True, radius=2.5)
        self.surface = Surface('tests/example_surface.stl')
        self.map = RoughnessMap(
            self.surface,
            'delta_t',
            self.window,
            1,1,0,0)
        
    def test_sampling(self):
        self.map.sample()
        self.assertEqual(518,len(self.map.samples))
        self.assertAlmostEqual(22.726952268535697, self.map.samples[0,0])
        self.assertAlmostEqual(7.00697149, self.map.samples[0,1])
        
    def test_evaluation(self):
        self.map.sample()
        self.map.evaluate()
        self.map.analyze_directional_roughness('delta_t')
        
        for orientation in self.map.final_orientations:
            self.assertAlmostEqual(1., orientation[2],places=5)
        
        self.assertAlmostEqual(6.60086672, self.map.total_areas[0])
        self.assertAlmostEqual(5.38515774, self.map.roughness_data['delta_t'][0,0])
        self.assertAlmostEqual(5.77771409, self.map.roughness_data['delta_t'][517,71])