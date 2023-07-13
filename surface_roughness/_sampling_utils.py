from collections import deque

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import numexpr as ne

from surface_roughness._geometry_utils import (
    points_in_polygon,
    point_in_polygon
)
from shapely import geometry

def _resample_polyline(positions,spacing:float):
    # Adjust list of points to fixed spacing
    new_positions = [positions[0]]
    current_point_idx = 0
    current_position = positions[0]

    while current_point_idx + 1 != len(positions):
        traverse_dt = spacing
        current_segment_length = np.linalg.norm(positions[current_point_idx+1] - current_position)
        
        if current_segment_length < spacing:
            current_position = positions[current_point_idx+1]
            current_point_idx = current_point_idx + 1
            if current_point_idx + 1 == len(positions):
                break
            traverse_dt = traverse_dt - current_segment_length
            current_segment_length = np.linalg.norm(positions[current_point_idx+1] - current_position)
        
        direction = positions[current_point_idx+1] - current_position
        direction = direction / np.linalg.norm(direction)
        new_point = direction * traverse_dt + current_position
        if np.any(np.isnan(new_point)):
            break
        new_positions.append(new_point)
        current_position = new_point
    
    return np.array(new_positions)

def _next_point(current_position,current_velocity,field,positions,dt):
    # Orient field along current velocity
    current_velocity = current_velocity / np.linalg.norm(current_velocity)
    orientation = field @ current_velocity
    orientation = orientation / np.abs(orientation)
    newfield = field * orientation[:,np.newaxis]
    
    # RK4 streamline method
    k1 = dt * current_velocity
    k2 = dt * griddata(positions,newfield,current_position + k1/2)[0]
    k3 = dt * griddata(positions,newfield,current_position + k2/2)[0]
    k4 = dt * griddata(positions,newfield,current_position + k3)[0]
    return current_position + (k1 + 2*k2 + 2*k3 + k4)/6

def _comb_vectorfield(positions,field):
    tree = KDTree(positions)
    dd,ii = tree.query(positions,k=8)
    dd,ii = dd[:,1:],ii[:,1:]
    average_distance = np.mean(dd[:,0])
    traverse_distance = average_distance/2
    
    minpoint = np.array([np.min(positions[:,0]),np.min(positions[:,1])])
    n = (np.max(positions,axis=0) - np.min(positions,axis=0))//traverse_distance + 1
    
    for x_i in range(int(n[0])):
        x_pos = minpoint[0] + x_i * traverse_distance
        for y_i in range(int(n[1])):
            y_pos = minpoint[1] + y_i * traverse_distance
            
            idx = tree.query_ball_point((x_pos,y_pos), average_distance*2)
            local_field = field[idx]
            if local_field.size == 0:
                continue
            local_average = np.mean(local_field,axis=0)
            flip_unit = np.sum(local_field * local_average,axis=1)
            flip_unit = flip_unit / np.abs(flip_unit)
            field[idx] = field[idx] * flip_unit[:,np.newaxis]
            
    return field 

def _generate_vtkdir_data(raw_dir_data,magnitudes,centroids,normals,samples):
    dir = raw_dir_data
    dir = np.hstack([np.cos(dir[:,np.newaxis]),np.sin(dir[:,np.newaxis])])
    vtk_dir = griddata(samples,dir,centroids[:,:2])
    vtk_mag = griddata(samples,magnitudes,centroids[:,:2])
    vtk_dir /= np.linalg.norm(vtk_dir,axis=1)[:,np.newaxis]
    vtk_dir = np.hstack([vtk_dir,np.zeros([vtk_dir.shape[0],1])])
    vtk_perp = np.vstack([-vtk_dir[:,1],vtk_dir[:,0],vtk_dir[:,2]]).T
    vtk_dir = np.cross(vtk_perp,normals)*vtk_mag[:,np.newaxis]
    
    return [vtk_dir]

def _offset_bounds(bounds,distance_offset):
    newbounds = []
    for bound in bounds:
        polygon = geometry.Polygon(bound[:,:2])
        polygon = polygon.buffer(-distance_offset)
        if polygon.geom_type == 'MultiPolygon':
            current_poly = polygon.geoms[0]
            for poly in polygon.geoms:
                if poly.area > current_poly.area:
                    current_poly = poly
            polygon = current_poly
        x = np.array(polygon.exterior.xy[0])[:,np.newaxis]
        y = np.array(polygon.exterior.xy[1])[:,np.newaxis]
        newbounds.append(np.hstack([x,y]))
    
    return newbounds

def _centroids_in_offset_bounds(centroids,bounds,distance_from_boundary):
    offset_bounds = np.vstack(_offset_bounds(bounds,distance_from_boundary))
    return points_in_polygon(centroids,offset_bounds)
        
def _triangles_from_p_in_sample(triangles,point_in_sample):
    return np.all(point_in_sample[triangles],axis=1)

def _in_circle(x,y,cx,cy,r):
    # https://stackoverflow.com/questions/59963642/is-there-a-fast-python-algorithm-to-find-all-points-in-a-dataset-which-lie-in-a
    return ne.evaluate('(x - cx)**2 + (y-cy)**2 < r**2')

def _streamline(positions,field,offset_bounds,starter,max_length,dt):
    sample_positions = deque()
    velocity = griddata(positions,field,starter)[0]
    current_point = starter
    length = 0
    
    max_n_samples = max_length/dt*3
    while True:
        new_point = _next_point(current_point,velocity,field,positions,dt)
        if np.isnan(new_point[0]):
            break
        sample_positions.append(new_point)
        length = length + np.linalg.norm(new_point - current_point)
        current_point = new_point
        if not point_in_polygon(current_point,offset_bounds):
            break
        if length > max_length:
            break
        if len(sample_positions) > max_n_samples:
            break
        if new_point[0] < offset_bounds[0,0] and new_point[0] > offset_bounds[1,0]:
            if new_point[1] < offset_bounds[0,1] and new_point[1] > offset_bounds[1,1]:
                break
    current_point = sample_positions[0]
    while True:
        new_point = _next_point(current_point,velocity,field,positions,-dt)
        if np.isnan(new_point[0]):
            break
        sample_positions.appendleft(new_point)
        length = length + np.linalg.norm(new_point - current_point)
        current_point = new_point
        if not point_in_polygon(current_point,offset_bounds):
            break
        if length > max_length:
            break
        if len(sample_positions) > 2*max_n_samples:
            break
        if new_point[0] < offset_bounds[0,0] and new_point[0] > offset_bounds[1,0]:
            if new_point[1] < offset_bounds[0,1] and new_point[1] > offset_bounds[1,1]:
                break
    return sample_positions