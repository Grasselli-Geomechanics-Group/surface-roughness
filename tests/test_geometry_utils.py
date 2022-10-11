import unittest
import numpy as np
from surface_roughness._geometry_utils import (
    line_intersection,
    segments_intersecting,
    create_event_list,
    ray_intersection,
    point_in_polygon,
    get_max_lefthand,
    SegmentTree
)

class TestLineIntersection(unittest.TestCase):
    def setUp(self):
        self.p1 = np.array([-1.,-1.])
        self.p2 = np.array([1.,1.])
        self.q1 = np.array([-1.,1.])
        self.q2 = np.array([1.,-1.])
        self.r1 = np.array([300,2])
        self.r2 = np.array([300,-2])
        
    def test_is_intersecting(self):
        self.assertTrue(line_intersection(self.p1,self.p2,self.q1,self.q2))
        self.assertTrue(line_intersection(self.p2,self.p1,self.q2,self.q1))
        
    def test_is_non_intersecting(self):
        self.assertFalse(line_intersection(self.p1,self.q1,self.p2,self.q2))
        self.assertFalse(line_intersection(self.q2,self.p2,self.q1,self.p1))
        self.assertFalse(line_intersection(self.p1,self.p2,self.r1,self.r2))
        self.assertFalse(line_intersection(self.r1,self.r2,self.q1,self.q2))

class TestSegmentsIntersecting(unittest.TestCase):
    def setUp(self):
        t1 = np.array([
            [0.,1.],
            [1.,0.],
            [-1.,0.]
        ])
        t2 = np.array([
            [0.,0.1],
            [-0.5,0.4],
            [0.5,0.4]
        ])
        s1 = np.vstack([t1,np.array([0.,-1.])])
        self.segments1 = create_event_list(np.array([0,1,2]),t1)
        self.segments2 = create_event_list(np.array([0,1,2]),t2)
        self.segments3 = create_event_list(np.array([0,1,2]),t2+np.array([0,0.5]))
        self.segments4 = create_event_list(np.array([0,1,2]),t2+np.array([0.2,0.3]))
        self.segments5 = create_event_list(np.array([0,1,2,3]),s1)
        
    def test_is_intersecting(self):
        self.assertTrue(segments_intersecting(self.segments1,self.segments3))
        self.assertTrue(segments_intersecting(self.segments2,self.segments4))
    
    def test_not_intersecting(self):
        self.assertFalse(segments_intersecting(self.segments2,self.segments3))
        self.assertFalse(segments_intersecting(self.segments1,self.segments2))

class TestRayIntersection(unittest.TestCase):
    def setUp(self):
        # left to right segment
        self.segment = np.array([
            [-1.,0.],
            [1.,0.]
        ])
        self.right_diagonal_segment = np.array([
            [1.,0.],
            [0.,1.]
        ])
        
        self.left_diagonal_segment = np.array([
            [-1.,0.],
            [0.,1.]
        ])
        self.below = np.array([0,-1.])
        self.farbelow = np.array([0,-400.])
        self.above = np.array([0,1.])
        self.left = np.array([-2.,0.])
        self.right = np.array([2.,0.])
        self.lowerleft = np.array([-2.,-1.])
        self.lowerright = np.array([2.,-1.])
        self.upperleft = np.array([-2.,1.])
        self.upperright = np.array([2.,1.])
        self.centre_point = np.array([0.,0.5])
        self.below_left = np.array([-1.,0.])
        self.below_right = np.array([1.,0.])
        
    def test_intersecting_left_right(self):
        self.assertTrue(ray_intersection(self.below,np.array([0.,1.]),self.segment[0],self.segment[1]))
        self.assertTrue(ray_intersection(self.below,np.array([0.,1.]),self.segment[1],self.segment[0]))
        self.assertTrue(ray_intersection(self.farbelow, np.array([0,1.]), self.segment[0],self.segment[1]))
        self.assertTrue(ray_intersection(self.above,np.array([0.,-1.]),self.segment[0],self.segment[1]))
        self.assertTrue(ray_intersection(self.above,np.array([0.,-1.]),self.segment[1],self.segment[0]))
    
    def test_non_intersecting_left_right(self):
        self.assertFalse(ray_intersection(self.above,np.array([0.,1.]),self.segment[0],self.segment[1]))
        self.assertFalse(ray_intersection(self.above,np.array([0.,1.]),self.segment[1],self.segment[0]))
        self.assertFalse(ray_intersection(self.below,np.array([0.,-1.]),self.segment[0],self.segment[1]))
        self.assertFalse(ray_intersection(self.below,np.array([0.,-1.]),self.segment[1],self.segment[0]))
        self.assertFalse(ray_intersection(self.upperleft,np.array([0.,1.]),self.segment[0],self.segment[1]))
        self.assertFalse(ray_intersection(self.upperright,np.array([0.,1.]),self.segment[1],self.segment[0]))
        self.assertFalse(ray_intersection(self.lowerleft,np.array([0.,-1.]),self.segment[0],self.segment[1]))
        self.assertFalse(ray_intersection(self.lowerright,np.array([0.,-1.]),self.segment[1],self.segment[0]))

    def test_intersecting_leftright_left_endpoint(self):
        self.assertFalse(ray_intersection(self.below_left,np.array([0.,1.]),self.segment[0],self.segment[1]))

    def test_intersecting_leftright_right_endpoint(self):
        self.assertTrue(ray_intersection(self.below_right,np.array([0.,1.]),self.segment[0],self.segment[1]))
    
    def test_intersecting_rightleft_left_endpoint(self):
        self.assertTrue(ray_intersection(self.below_left,np.array([0.,1.]),self.segment[1],self.segment[0]))

    def test_intersecting_rightleft_right_endpoint(self):
        self.assertFalse(ray_intersection(self.below_right,np.array([0.,1.]),self.segment[1],self.segment[0]))
    
    def test_intersecting_right_diagonal(self):
        self.assertTrue(ray_intersection(self.centre_point,np.array([1.,0.]),self.right_diagonal_segment[0],self.right_diagonal_segment[1]))

    def test_not_intersecting_right_diagonal(self):
        self.assertFalse(ray_intersection(self.centre_point,np.array([-1.,0.]),self.right_diagonal_segment[0],self.right_diagonal_segment[1]))

    def test_intersecting_left_diagonal(self):
        self.assertTrue(ray_intersection(self.centre_point,np.array([-1.,0.]),self.left_diagonal_segment[0],self.left_diagonal_segment[1]))
    
    def test_not_intersecting_left_diagonal(self):
        self.assertFalse(ray_intersection(self.centre_point,np.array([1.,0.]),self.left_diagonal_segment[0],self.left_diagonal_segment[1]))

class TestPointInPolygon(unittest.TestCase):
    def setUp(self):
        self.polygon = np.array([
            [0.,1.],
            [1.,0.],
            [-1.,0.]
        ])
        self.reverse_polygon = np.array([
            [0.,1.,],
            [-1.,0.],
            [1.,0.]
        ])
        self.complex_polygon = np.array(
            [[ -3.00347664,  10.59417122],
       [ -3.00347664,  11.85079755],
       [ 17.59640177,  11.85079755],
       [ 22.74637137,  11.85079755],
       [ 25.32135617,  10.59417122],
       [ 25.32135617, -10.76847639],
       [ -0.42849184, -10.76847639],
       [ -3.00347664,   9.33754489]])
        
        self.inside_point = np.array([0.,0.5])
        self.inside_offset_point = np.array([0.1,0.5])
        self.outside_point = np.array([2.,0.5])

        self.complex_in_point = np.array([0,10])
        self.complex_out_point = np.array([0,30])

    def test_point_in_polygon(self):
        self.assertTrue(point_in_polygon(self.inside_point,self.polygon))
    
    def test_point_in_reverse_polygon(self):
        self.assertTrue(point_in_polygon(self.inside_point,self.reverse_polygon))
    
    def test_point_in_polygon_offset(self):
        self.assertTrue(point_in_polygon(self.inside_offset_point,self.polygon))
    
    def test_point_in_reverse_polygon_offset(self):
        self.assertTrue(point_in_polygon(self.inside_offset_point,self.polygon))

    def test_point_not_in_polygon(self):
        self.assertFalse(point_in_polygon(self.outside_point,self.polygon))
    
    def test_point_not_in_reverse_polygon(self):
        self.assertFalse(point_in_polygon(self.outside_point,self.reverse_polygon))

    def test_complex_in_polygon(self):
        self.assertTrue(point_in_polygon(self.complex_in_point, self.complex_polygon))

    def test_complex_not_in_polygon(self):
        self.assertFalse(point_in_polygon(self.complex_out_point, self.complex_polygon))

class TestGetMaxLefthand(unittest.TestCase):
    def setUp(self):
        # left to right segment
        self.segment = np.array([
            [-1.,0.],
            [1.,0.]
        ])
        self.points = np.array([
            [0,-2.],
            [1.,-1],
            [100,-1],
            [0,-3],
            [0,2.],
            [1.,1],
            [100,1],
            [0,3]
        ])
        self.all_ynegative = np.array([
            [0,-2.],
            [1.,-1],
            [100,-1],
            [0,-3]
        ])
        self.all_ypositive = np.array([
            [0,2.],
            [1.,1],
            [100,1],
            [0,3]
        ])
    
    def test_bisecting(self):
        self.assertTrue(
            np.array_equal(np.array([0,3]),
                            get_max_lefthand(self.segment[0],
                                            self.segment[1],
                                            self.points)))
    
    def test_find_positive(self):
        self.assertTrue(
            np.array_equal(np.array([0,3]),
                            get_max_lefthand(self.segment[0],
                                            self.segment[1],
                                            self.all_ypositive)))
    def test_find_none(self):
        self.assertFalse(get_max_lefthand(self.segment[0],
                                          self.segment[1],
                                          self.all_ynegative))
    
    def test_flipped_bisecting(self):
        self.assertTrue(
            np.array_equal(np.array([0,-3]),
                            get_max_lefthand(self.segment[1],
                                            self.segment[0],
                                            self.points)))
        
    def test_flipped_find_positive(self):
        self.assertTrue(
            np.array_equal(np.array([0,-3]),
                            get_max_lefthand(self.segment[1],
                                            self.segment[0],
                                            self.all_ynegative)))
    def test_flipped_find_none(self):
        self.assertFalse(get_max_lefthand(self.segment[1],
                                          self.segment[0],
                                          self.all_ypositive))

class TestSegmentTree(unittest.TestCase):
    def setUp(self):
        points = np.array([
            [0.,0],
            [2,2],
            [1,-0.5],
            [-1,-2],
            [-1.75,-1],
            [-2,1]
        ])
        self.points = points[points[:,0].argsort()]
        # Expected order:
        # seg 0:
        #   leftpoint = [-2,1]
        #   rightpoint = [2,2]
        #   farpoint = False
        #   no segments
        self.nochildtree = SegmentTree(self.points[0],
                                        self.points[-1],
                                        get_max_lefthand(self.points[0],
                                                        self.points[-1],
                                                        self.points))

        self.nochildtree.check_children(self.points)
        # Expected order:
        # seg 0:
        #   leftpoint = [2,2]
        #   rightpoint = [-2,1]
        #   farpoint = [-2,-1]
        #   segments = [seg 1, seg 2]
        # seg 1:
        #   leftpoint = [2,2]
        #   rightpoint = [-2,-1]
        #   farpoint = [1,-0.5]
        #   segments = None
        # seg 2:
        #   leftpoint = [-1,-2]
        #   rightpoint = [-2,1]
        #   farpoint = [-1.75,-1]
        # segments = None
        self.childtree = SegmentTree(self.points[-1],
                           self.points[0],
                           get_max_lefthand(self.points[-1],
                                            self.points[0],
                                            self.points))
        self.childtree.check_children(self.points)
        
    def test_nochild_initialization(self):
        self.assertTrue(len(self.nochildtree.child_segments) == 0)

    def test_child_initialization(self):
        self.assertTrue(len(self.childtree.child_segments) == 2)
    
    def test_child_left_initialized(self):
        self.assertTrue(self.childtree.left_used)
        self.assertTrue(np.array_equal(self.childtree.child_segments[0].leftpoint,np.array([2,2])))
        self.assertTrue(np.array_equal(self.childtree.child_segments[0].farpoint,np.array([1,-0.5])))
        
    def test_child_right_initialized(self):
        self.assertTrue(self.childtree.right_used)
        self.assertTrue(np.array_equal(self.childtree.child_segments[1].leftpoint,np.array([-1,-2])))
        self.assertTrue(np.array_equal(self.childtree.child_segments[1].farpoint,np.array([-1.75,-1])))
        
    def test_extract_points_nochild(self):
        points = self.nochildtree.extract_points()
        self.assertTrue(np.array_equal(np.array([-2,1]),points[0]))
        self.assertTrue(np.array_equal(np.array([2,2]),points[1]))
        self.assertEqual(len(points),2)
    
    def test_extract_points_child(self):
        points = self.childtree.extract_points()
        self.assertEqual(len(points),5)
        self.assertTrue(np.array_equal(np.array([2,2]),points[0]))
        self.assertTrue(np.array_equal(np.array([1,-0.5]),points[1]))
        self.assertTrue(np.array_equal(np.array([-1,-2]),points[2]))
        self.assertTrue(np.array_equal(np.array([-1.75,-1.]),points[3]))
        self.assertTrue(np.array_equal(np.array([-2,1]),points[-1]))
        

class TestSortBoundary(unittest.TestCase):
    def setUp(self):
        pass

    def test_sorting(self):
        pass
    
if __name__ == '__main__':
    unittest.main()