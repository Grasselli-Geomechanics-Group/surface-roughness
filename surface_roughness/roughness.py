
from meshio import read
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull

from .roughness_impl import (
    _rs,
    _cppDirectionalRoughness,
    _PyDirectionalRoughness,
    _PyTINBasedRoughness,
    _cppTINBasedRoughness,
    _cppTINBasedRoughness_bestfit,
    _cppTINBasedRoughness_againstshear,
    _cppMeanDipRoughness
)


class Surface:
    """This class loads a surface for surface roughness analysis.
    
    """
    def __init__(self,path=None,mesh=None,preprocess=True,verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
        self._mesh = mesh
        if path:
            if verbose:
                print("Loading mesh at path")
            self._mesh = read(path)
        self.n_triangles = len(self._mesh.cells_dict['triangle'])
        self.triangles : np.ndarray = self._mesh.cells_dict['triangle']
        self.points : np.ndarray = self._mesh.points
        self._roughness_data = {}
        self.is_run = False
        self.is_preprocessed = preprocess

        self._thetamax_cp1 = None
        self._delta_t = None
        self._delta_a = None
        self._delta_n = None
        self._meandip = None

        self._roughness_map = None
        
        if preprocess:
            self.preprocess()
        if verbose:
            print("Calculating normals...")
        self._calculate_normals()
    
    @property
    def original_points(self):
        return self._mesh.points

    @property
    def original_normals(self):
        original_points = self.original_points
        v1v0 = np.array([original_points[tri_i[1]] - original_points[tri_i[0]] for tri_i in self.triangles])
        v2v0 = np.array([original_points[tri_i[2]] - original_points[tri_i[0]] for tri_i in self.triangles])
        normals = np.cross(v1v0,v2v0,axisa=1,axisb=1)
        normals /= np.linalg.norm(self.normals,axis=1)[:,np.newaxis]

        return normals

    def convex_bounds(self):
        h = ConvexHull(self.points[:,:2])

        return self.points[h.vertices]

    def bounds(self):
        return np.vstack([self.points.min(axis=0),self.points.max(axis=0)])
    
    def plot(self):
        # return pptk.viewer(self.points)
        pass
    
    def preprocess(self):
        if self.verbose:
            print("Aligning to best fit...")
        self._align_best_fit()
        if self.verbose:
            print("Calculating areas...")
        self._calculate_areas()

    @property
    def area(self):
        return np.sum(self._areas)

    @property
    def resolution(self):
        return np.sqrt(4/np.sqrt(3)*np.mean(self._areas))

    @property
    def B_exp(self):
        return 1.34*self.resolution**0.058

    @staticmethod
    def _find_rotmatrix_2_z_pos(v_orig):
        #https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
        #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        v_orig /= np.linalg.norm(v_orig)
        up = np.array([0,0,1.])
        v = np.cross(v_orig,up)
        c = np.dot(v_orig,up)
        s = np.linalg.norm(v)

        v_skew = np.array([
            [0,-v[2],v[1]],
            [v[2],0,-v[0]],
            [-v[1],v[0],0]
        ])
        return np.eye(3) + v_skew + v_skew.dot(v_skew)*(1-c)/s**2
    
    @staticmethod
    def _calculate_surface_normal(points):
        p_temp = points - points.mean(axis=0)
        u,s,_ = np.linalg.svd(p_temp.T,full_matrices=False)
        normal = u[:,np.argmin(s)]
        normal /= np.linalg.norm(normal)
        return normal
    
    @property
    def surface_normal(self):
        return Surface._calculate_surface_normal(self.points)

    def _align_best_fit(self):
        self.centroid = np.mean(self.points,axis=0)
        zeroed_points = self.points - self.centroid
        self.initial_orientation = self.surface_normal
        orientation = self.surface_normal

        self.x_rot = np.arcsin(orientation[0])
        self.y_rot = np.arcsin(orientation[1])

        rot_mat = Surface._find_rotmatrix_2_z_pos(orientation)
        self.points = zeroed_points @ rot_mat.T + self.centroid

    def _calculate_normals(self):
        v1v0 = np.array([self.points[tri_i[1]] - self.points[tri_i[0]] for tri_i in self.triangles])
        v2v0 = np.array([self.points[tri_i[2]] - self.points[tri_i[0]] for tri_i in self.triangles])
        self.normals = np.cross(v1v0,v2v0,axisa=1,axisb=1)
        self.normals /= np.linalg.norm(self.normals,axis=1)[:,np.newaxis]

    def _calculate_areas(self):
        self._areas = np.zeros([self.n_triangles])
        self._nominal_areas = np.zeros([self.n_triangles])
        v1v0 = np.array([self.points[tri_i[1]] - self.points[tri_i[0]] for tri_i in self.triangles])
        v2v0 = np.array([self.points[tri_i[2]] - self.points[tri_i[0]] for tri_i in self.triangles])
        self._areas = np.linalg.norm(np.cross(v1v0,v2v0,axis=1),axis=1)
        v1v0[:,2] = 0
        v2v0[:,2] = 0
        self._nominal_areas = np.linalg.norm(np.cross(v1v0,v2v0,axis=1),axis=1)

    def __getitem__(self,key):
        return self._roughness_data[key]

    def rs(self):
        if not 'rs' in self._roughness_data:
            self._roughness_data['rs'] = _rs(self._nominal_areas,self._areas)
        return self._roughness_data['rs']
    
    def evaluate_thetamax_cp1(self,verbose=False,impl='cpp',**dr_kwargs):
        dr_kwargs.setdefault('n_directions',72)
        dr_kwargs.setdefault('n_offset',0)
        dr_kwargs.setdefault('n_dip_bins',90)
        dr_kwargs.setdefault('fit_initialguess',1)
        dr_kwargs.setdefault('fit_precision',6)
        dr_kwargs.setdefault('fit_regularization',10e-10)
        dr_kwargs.setdefault('fit_alpha',0.01)
        dr_kwargs.setdefault('fit_beta',0.5)
        dr_kwargs.setdefault('min_triangles',200)

        if impl == 'cpp':
            self._thetamax_cp1 = _cppDirectionalRoughness(self.points,self.triangles,**dr_kwargs)
        elif impl == 'py':
            self._thetamax_cp1 = _PyDirectionalRoughness(self.points,self.triangles,self.normals,self._areas,**dr_kwargs)
        else:
            raise ValueError("evaluate_dr argument impl must be either 'cpp' or 'py'")
        self._thetamax_cp1.evaluate(verbose)

    def thetamax_cp1(self,value,**dr_kwargs):
        if (not self._thetamax_cp1) or dr_kwargs:
            self.evaluate_thetamax_cp1(**dr_kwargs)
        dr_kwargs.setdefault('return_az',False)
        if dr_kwargs['return_az']:
            return self._thetamax_cp1[value],self._thetamax_cp1['az']

        return self._thetamax_cp1[value]

    def evaluate_delta_t(self,verbose=False,impl='cpp',**tin_kwargs):
        tin_kwargs.setdefault('n_directions',72)
        tin_kwargs.setdefault('n_offset',0)
        tin_kwargs.setdefault('min_triangles',200)
        if impl == 'cpp':
            self._delta_t = _cppTINBasedRoughness(self.points,self.triangles,**tin_kwargs)
        elif impl == 'py':
            self._delta_t = _PyTINBasedRoughness(self.points,self.triangles,self._areas,self.normals,**tin_kwargs)
        else:
            raise ValueError("evaluate_TIN argument impl must be either 'cpp' or 'py'")
        self._delta_t.evaluate(verbose)

    def delta_t(self,value=None,**tin_kwargs):
        if value == None:
            return self._delta_t
        if (not self._delta_t) or tin_kwargs:
            self.evaluate_delta_t(**tin_kwargs)
        tin_kwargs.setdefault('return_az',False)
        if tin_kwargs['return_az']:
            return self._delta_t[value],self._delta_t['az']

        return self._delta_t[value]

    def evaluate_delta_a(self,verbose=False,impl='cpp',**tin_kwargs):
        tin_kwargs.setdefault('n_directions',72)
        tin_kwargs.setdefault('n_offset',0)
        tin_kwargs.setdefault('min_triangles',200)
        if impl == 'cpp':
            self._delta_a = _cppTINBasedRoughness_bestfit(self.points,self.triangles,**tin_kwargs)
        elif impl == 'py':
            self._delta_a = _PyTINBasedRoughness_bestfit(self.points,self.triangles,self._areas,self.normals,**tin_kwargs)
        else:
            raise ValueError("evaluate_TIN argument impl must be either 'cpp' or 'py'")
        self._delta_a.evaluate(verbose)

    def delta_a(self,value=None,**tin_kwargs):
        if value == None:
            return self._delta_a
        if (not self._delta_a) or tin_kwargs:
            self.evaluate_delta_a(**tin_kwargs)
        tin_kwargs.setdefault('return_az',False)
        if tin_kwargs['return_az']:
            return self._delta_a[value],self._delta_a['az']

        return self._delta_a[value]
    
    def evaluate_delta_n(self,verbose=False,impl='cpp',**tin_kwargs):
        tin_kwargs.setdefault('n_directions',72)
        tin_kwargs.setdefault('n_offset',0)
        tin_kwargs.setdefault('min_triangles',200)
        if impl == 'cpp':
            self._delta_n = _cppTINBasedRoughness_againstshear(self.points,self.triangles,**tin_kwargs)
        elif impl == 'py':
            self._delta_n = _PyTINBasedRoughness_againstshear(self.points,self.triangles,self._areas,self.normals,**tin_kwargs)
        else:
            raise ValueError("evaluate_TIN argument impl must be either 'cpp' or 'py'")
        self._delta_n.evaluate(verbose)

    def delta_n(self,value=None,**tin_kwargs):
        if value == None:
            return self._delta_n
        if (not self._delta_n) or tin_kwargs:
            self.evaluate_delta_n(**tin_kwargs)
        tin_kwargs.setdefault('return_az',False)
        if tin_kwargs['return_az']:
            return self._delta_n[value],self._delta_n['az']

        return self._delta_n[value]

    def evaluate_meandip(self,verbose=False,impl='cpp',**tin_kwargs):
        tin_kwargs.setdefault('n_directions',72)
        tin_kwargs.setdefault('n_offset',0)
        tin_kwargs.setdefault('min_triangles',200)
        if impl == 'cpp':
            self._meandip = _cppMeanDipRoughness(self.points,self.triangles,**tin_kwargs)
        elif impl == 'py':
            self._meandip = _PyMeanDipRoughness(self.points,self.triangles,self._areas,self.normals,**tin_kwargs)
        else:
            raise ValueError("evaluate_TIN argument impl must be either 'cpp' or 'py'")
        self._meandip.evaluate(verbose)

    def meandip(self,value=None,**tin_kwargs):
        if value == None:
            return self._meandip
        if (not self._meandip) or tin_kwargs:
            self.evaluate_meandip(**tin_kwargs)
        tin_kwargs.setdefault('return_az',False)
        if tin_kwargs['return_az']:
            return self._meandip[value],self._meandip['az']

        return self._meandip[value]

    def to_pandas(self, metric):
        metric_methods:dict[pd.DataFrame] = {
            'delta_t':self._delta_t,
            'thetamax_cp1':self._thetamax_cp1
        }
        metric_methods[metric].to_pandas()

