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
    
    if tnum >= 0 and tnum <= tdem:
        return unum >= 0 and unum <= udem
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
        if t >= 0 and t <= 1:
            return u >= 0 and u <= 1
    return False

def point_in_polygon(p:np.ndarray,segments):
    count = 0
    for i in range(segments.shape[0]):
        p1 = segments[i-1]
        p2 = segments[i]
        if ray_intersection(p,np.array([0,1]),p1,p2):
            count += 1
    # intersection if count is odd
    return count % 2 == 1
