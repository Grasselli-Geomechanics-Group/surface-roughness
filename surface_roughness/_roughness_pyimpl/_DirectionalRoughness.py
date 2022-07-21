import numpy as np
from tqdm.contrib import tenumerate
from scipy.optimize import curve_fit

class _DRTriangle:
    def __init__(self,normal,area,points):
        self.normal = normal
        self.area = area
        self.points = points
    
    def set_normal(self,normal):
        self.normal = normal

    def set_apparent_dip(self,shear_dir):
        prx = self.normal[0] - (shear_dir[1]*shear_dir[1]*self.normal[0] - shear_dir[1]*shear_dir[0]*self.normal[1])
        pry = self.normal[1] - (-shear_dir[0]*shear_dir[1]*self.normal[0] + shear_dir[1]*shear_dir[1]*self.normal[1])
        prz = self.normal[2]

        norm = np.linalg.norm(np.array([prx,pry,prz]))
        prx /= norm
        pry /= norm

        self.apparent_dip_angle = np.arccos(shear_dir[0]*prx + shear_dir[1]*pry) - np.pi/2

def theta_cp1_curve(theta_curve,C):
    return theta_curve**C

class _PyDirectionalRoughness:
    def __init__(self,points,triangles,normals,areas,**kwargs):
        self.points = points
        self.triangles = triangles
        self.normals = normals
        self._settings = kwargs
        self._az = np.linspace(self._settings['n_directions'],self._settings['n_directions']+2*np.pi,self._settings['n_directions'],endpoint=False)
        self._parameter = {}
        self.evaluated = False
        self.triangle_container = [_DRTriangle(n,a,self.points[t,:]) for n,t,a in zip(normals,triangles,areas)]
        self._result_keys = ['theta_max','c','c_cov','a_0','thetamax_cp1']
    def __getitem__(self,key):
        if not self.evaluated:
            self.evaluate()
        return self._parameter[key]

    def result_keys(self):
        return self._result_keys

    def evaluate(self,verbose=False,filename=None):
        shear_dirs = [np.array([np.cos(az),np.sin(az)]) for az in self._az]
        self._parameter['az'] = [az*180./np.pi for az in self._az]
        for parameter in self._result_keys:
            self._parameter[parameter] = len(shear_dirs)*[None]
        total_area = sum([t.area for t in self.triangle_container])
        for i,shear_dir in tenumerate(shear_dirs,total=len(shear_dirs)):

            for t in self.triangle_container:
                t.set_apparent_dip(shear_dir)
            triangles = [t for t in self.triangle_container if t.apparent_dip_angle >= 0]
            if len(triangles) > 0:
                dips = [t.apparent_dip_angle for t in triangles]
                areas = [t.area for t in triangles]
                bin_area_pdf, bins = np.histogram(dips,self._settings['n_dip_bins'],(0,np.pi),weights=areas)

                bin_area_cdf = np.cumsum(bin_area_pdf[::-1])[::-1]/total_area
                bin_area_cdf = np.trim_zeros(bin_area_cdf,'b')
                bin_area_cdf = np.append(bin_area_cdf,0)
                bins = bins[:len(bin_area_cdf)-1]
                
                self._parameter['theta_max'][i] = max(triangles,key=lambda x: x.apparent_dip_angle).apparent_dip_angle
                bins = np.append(bins, self._parameter['theta_max'][i])
                self._parameter['a_0'][i] = bin_area_cdf[0]
                x = (1.-bins/self._parameter['theta_max'][i])
                y = bin_area_cdf/self._parameter['a_0'][i]

                self._parameter['c'][i],self._parameter['c_cov'][i] = curve_fit(theta_cp1_curve,x,y,bounds=(0,np.inf))
                self._parameter['thetamax_cp1'][i] = self._parameter['theta_max'][i]/(self._parameter['c'][i]+1)*180/np.pi
        
        self._parameter['theta_max'] = [tmax*180./np.pi for tmax in self._parameter['theta_max']]
        self.evaluated=True

