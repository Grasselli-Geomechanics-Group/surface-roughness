from dataclasses import dataclass
from copy import deepcopy

import numpy as np

@dataclass
class Event:
    p1: np.ndarray
    p2: np.ndarray
    type_:int
    
def loop_points(loop,points) -> np.ndarray:
    return np.vstack([points[index] for index in loop])

def create_event_list(loop,points):
    poly = loop_points(loop,points)
    events = []
    for i in range(poly.shape[0]):
        p1 = poly[i-1]
        p2 = poly[i]
        if np.isclose(p1[0],p2[0]):
            events.append(Event(p1,p2,2)) # Vertical line ignored
        elif p1[0] < p2[0]:
            events.append(Event(p1,p2,0)) # Left point
            events.append(Event(p2,p1,1)) # Right point
        else:
            events.append(Event(p2,p1,0))
            events.append(Event(p1,p2,1))
    events.sort(key=lambda x: x.p1[0])
    return events

def line_intersection(p1,p2,q1,q2):
    if np.allclose(p1,q1) or np.allclose(p1,q2) or\
        np.allclose(p2,q1) or np.allclose(p2,q2):
        return False
    x1,y1 = p1[:2]
    x2,y2 = p2[:2]
    x3,y3 = q1[:2]
    x4,y4 = q2[:2]
    tnum = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    tdem = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    unum = (x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)
    udem = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    if tnum*tdem >= 0 and abs(tnum) <= abs(tdem):
        return unum*udem >= 0 and abs(unum) <= abs(udem)
    return False

def segments_intersecting(segments1,segments2):
    pq: list[Event] = deepcopy(segments1)
    pq.extend(segments2)
    pq = sorted(pq,key=lambda x: x.p1[0])
    sl: list[Event] = []
    def add_segment(segment:Event):
        if len(sl) == 0:
            sl.append(segment)
            return False
        
        for i,sl_item in enumerate(sl):
            if sl_item.p1[1] > segment.p1[1]:
                sl.insert(i,segment)
                break
            elif i == len(sl)-1:
                if sl_item.p1[1] < segment.p1[1]:
                    sl.append(segment)
                else:
                    sl.insert(i,segment)
                break
                
        
        # Test item above
        if i < len(sl)-2:
            if line_intersection(segment.p1,segment.p2,sl[i+1].p1,sl[i+1].p2):
                return True
        elif i > 0:
            if line_intersection(segment.p1,segment.p2,sl[i-1].p1,sl[i-1].p2):
                return True
        elif i < len(sl)-2 and i > 0:
            if line_intersection(sl[i-1].p1,sl[i-1].p2,sl[i+1].p1,sl[i+1].p2):
                return True
        return False
    
    def remove_segment(point:np.ndarray):
        for i,sl_item in enumerate(sl):
            if np.array_equal(sl_item.p1, point):
                del sl[i]
                break
        if i != len(sl):
            result = line_intersection(sl[i-1].p1,sl[i-1].p2,sl[i].p1,sl[i].p2)
        else:
            return False
        if result:
            return True
        return False
    
    for segment_end in pq:
        if segment_end.type_ == 0:
            if add_segment(segment_end):
                return True
            
        elif segment_end.type_ == 1:
            if remove_segment(segment_end.p2):
                return True
        del segment_end
    
    return False


def original_segments_intersecting(points1,points2):
    for s1 in range(points1.shape[0]):
        p1 = points1[s1-1]
        p2 = points1[s1]
        for s2 in range(points2.shape[0]):
            q1 = points2[s2-1]
            q2 = points2[s2]
            if line_intersection(p1,p2,q1,q2):
                return True
    return False
        

def ray_intersection(point,direction,p1,p2):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    p1 = p1[:2]
    p2 = p2[:2]
    point = point[:2]
    q = p1
    s = p2-p1
    r = direction
    p = point[:2]
    if np.cross(r,s) != 0:
        t = np.cross(q-p,s)/np.cross(r,s)
        u = np.cross(q-p,r)/np.cross(r,s)
        if t >= 0:
            # Create imbalance for when ray passes directly through points
            return u > 0 and u <= 1
    return False

def point_in_polygon(p:np.ndarray,segments):
    count = 0
    for i in range(segments.shape[0]):
        p1 = segments[i-1]
        p2 = segments[i]
        if ray_intersection(p,np.array([0,1.]),p1,p2):
            count += 1
    # intersection if count is odd
    return count % 2 == 1

def get_max_lefthand(p0,p1,points):
    distances = np.cross(points-p0,points-p1,axis=1)
    if np.any(distances > 0):
        return points[np.argmax(distances)]
    else:
        return False

@dataclass
class SegmentTree:
    leftpoint:np.ndarray
    rightpoint:np.ndarray
    farpoint:np.ndarray
    is_parent:bool = True
    
    def __post_init__(self):
        self.child_segments: list[SegmentTree] = []
        self.left_used = False
        self.right_used = False
        self.points = []
        
    def check_children(self,points):
        if self.farpoint is not False:
            s1_farpoint = get_max_lefthand(self.leftpoint,self.farpoint,points)
            if s1_farpoint is not False:
                self.child_segments.append(SegmentTree(self.leftpoint,self.farpoint,s1_farpoint,False))
                self.child_segments[0].check_children(points)
                self.left_used = True
                
            s2_farpoint = get_max_lefthand(self.farpoint,self.rightpoint,points)
            if s2_farpoint is not False:
                self.child_segments.append(SegmentTree(self.farpoint,self.rightpoint,s2_farpoint,False))
                self.child_segments[-1].check_children(points)
                self.right_used = True

    def extract_points(self):
        if not self.is_parent or not self.left_used:
            # left point not added unless it is a parent
            self.points.append(self.leftpoint)
        # get midpoint data
        if self.left_used:
            self.points.extend(self.child_segments[0].extract_points())

            if self.right_used:
                self.points.extend(self.child_segments[-1].extract_points())
            else:
                self.points.append(self.farpoint)
        elif self.right_used:
            # right will automatically provide farpoint
            self.points.extend(self.child_segments[0].extract_points())
        else:
            # no right segment to provide farpoint
            if self.farpoint is not False:
                self.points.append(self.farpoint)
        if self.is_parent:
            self.points.append(self.rightpoint)
        idx = np.unique(self.points,axis=0,return_index=True)[1]
        return np.array([self.points[ix] for ix in sorted(idx)])
    
def sort_boundary(points:np.ndarray):
    points = points[points[:,0].argsort()]
    
    # Point 3 as maximum left-hand distance from line with min and max X
    point3 = get_max_lefthand(points[0],points[-1],points)

    if point3 is False:
        raise RuntimeError("cannot sort colinear boundary")
    segment_tree = SegmentTree(points[0],points[-1],point3)
    segment_tree.check_children(points)
    
    segment_tree2 = SegmentTree(points[-1],points[0],get_max_lefthand(points[-1],points[0],points))
    segment_tree2.check_children(points)
    convex_hull = segment_tree.extract_points()
    convex_hull = np.vstack([convex_hull,segment_tree2.extract_points()])
    convex_hull = np.array(convex_hull)
    return convex_hull

def points_in_polygon(p:np.ndarray,segments):
    # grid search
    x_grid = np.linspace(p[:,0].min()-.5,p[:,0].max()+.5,20)
    y_grid = np.linspace(p[:,1].min()-.5,p[:,1].max()+.5,20)
    
    # Group point indices based on grid location
    storage:dict[list[int]] = {}
    for x_i in range(x_grid.shape[0]-1):
        for y_i in range(y_grid.shape[0]-1):
            x_check = np.logical_and(x_grid[x_i] <= p[:,0],p[:,0] < x_grid[x_i+1])
            y_check = np.logical_and(y_grid[y_i] <= p[:,1],p[:,1] < y_grid[y_i+1])
            
            storage[(x_i,y_i)] = np.where(np.logical_and(x_check,y_check))[0]
    
    # Find grids intersected by boundary
    boundary_grids = set()
    for x_i in range(x_grid.shape[0]-1):
        for y_i in range(y_grid.shape[0]-1):
            x_check = np.logical_and(x_grid[x_i] <= segments[:,0],segments[:,0] < x_grid[x_i+1])
            y_check = np.logical_and(y_grid[y_i] <= segments[:,1],segments[:,1] < y_grid[y_i+1])
            
            if np.any(np.logical_and(x_check,y_check)):
                boundary_grids.add((x_i,y_i))
    
    # Find grids fully inside boundary grids
    rough_boundaries = np.array([[(x_grid[x_i]+x_grid[x_i+1])/2,(y_grid[y_i]+y_grid[y_i+1])/2] for x_i,y_i in list(boundary_grids)])

    # Sort rough_boundaries   
    rough_boundaries = sort_boundary(rough_boundaries)
    rough_boundaries = np.vstack([rough_boundaries,rough_boundaries[0]])
    inside_grid = set()
    for gridpoint in storage:
        x = (x_grid[gridpoint[0]]+x_grid[gridpoint[0]+1])/2
        y = (y_grid[gridpoint[1]] + y_grid[gridpoint[1]+1])/2
        if point_in_polygon(np.array([x,y]),rough_boundaries):
            inside_grid.add(gridpoint)
    
    # Initialize in_polygon matrix with inner grid point indices
    in_polygon = np.zeros(p.shape[0],dtype=np.bool_)
    for grid in inside_grid:
        in_polygon[storage[grid]] = True

    
    # Generate points to test within boundary grids
    test_points_idx = np.zeros([0],dtype=int)
    for grid in boundary_grids:
        test_points_idx = np.hstack([test_points_idx,storage[grid]])
    
    test_points = p[test_points_idx]

    # Winding number test for boundary points
    # https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html
    winding_numbers = np.zeros([test_points.shape[0]])
    for i,point in enumerate(segments):
        prev_point = segments[i-1]
        
        left_checks = (point[0] - prev_point[0]) * (test_points[:,1]-prev_point[1]) - (test_points[:,0]-prev_point[0]) * (point[1] - prev_point[1])
        
        upward_check = np.logical_and(prev_point[1] <= test_points[:,1],point[1] > test_points[:,1])
        winding_numbers[np.logical_and(upward_check,left_checks > 0)] += 1
        
        
        downward_check = np.logical_and(prev_point[1] > test_points[:,1],point[1] <= test_points[:,1])
        winding_numbers[np.logical_and(downward_check,left_checks < 0)] -= 1

    # Odd winding number means it's in polygon
    in_polygon[test_points_idx[(winding_numbers % 2) == 1]] = True
    
    return in_polygon
