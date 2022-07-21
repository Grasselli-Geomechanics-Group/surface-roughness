import numpy as np
from tqdm.contrib import tenumerate

class _PyTINTriangle:
    def __init__(self,normal,area,points):
        self.normal = normal
        self.area = area
        self.points = points
    
    def set_normal(self,normal):
        self.normal = normal

    def apparent_dip(self,shear_dir):
        prx = self.normal[0] - (shear_dir[1]*shear_dir[1]*self.normal[0] - shear_dir[1]*shear_dir[0]*self.normal[1])
        pry = self.normal[1] - (-shear_dir[0]*shear_dir[1]*self.normal[0] + shear_dir[0]*shear_dir[0]*self.normal[1])
        prz = self.normal[2]

        norm = np.linalg.norm(np.array([prx,pry,prz]))
        prx /= norm
        pry /= norm

        return np.arccos(shear_dir[0]*prx + shear_dir[1]*pry) - np.pi/2

class _PyTINBasedRoughness:
    def __init__(self,points,triangles,areas,normals,**kwargs):
        self.points = points
        self.triangles = triangles
        self.normals = normals
        self.areas = areas
        self._settings = kwargs
        self._az = np.linspace(self._settings['n_offset'],self._settings['n_offset']+2*np.pi,self._settings['n_directions'],endpoint=False)
        self._parameter = {}
        self.result_keys = ['delta_t','delta*_t','n_tri']
        self.evaluated = False
        self.triangle_container = [_PyTINTriangle(n,a,self.points[t,:]) for n,t,a in zip(normals,triangles,areas)]

    def __getitem__(self,key):
        if not self.evaluated:
            self.evaluate()
        return np.array(self._parameter[key])

    def evaluate(self,verbose):
        shear_dirs = [np.array([np.cos(az),np.sin(az)]) for az in self._az]
        self._parameter['az'] = self._az
        for parameter in self.result_keys:
            self._parameter[parameter] = len(shear_dirs)*[None]

        if verbose:
            shear_list = tenumerate(shear_dirs,total=len(shear_dirs))
        else:
            shear_list = enumerate(shear_dirs)
        
        for i, shear_dir in shear_list:
            apparent_dip = [t.apparent_dip(shear_dir) for t in self.triangle_container]
            filtered = [dip >= 0 for dip in apparent_dip]

            if any(filtered):
                dips = np.degrees(np.array([dip for dip,isin in zip(apparent_dip,filtered) if isin]))
                areas = np.array([t.area for t,isin in zip(self.triangle_container,filtered) if isin])
                self._parameter['delta_t'][i] = np.sum(dips*areas)/areas.sum()
                self._parameter['delta*_t'][i] = np.sqrt(np.sum(areas*dips**2)/areas.sum())
                self._parameter['n_tri'][i] = sum(filtered)

        self.evaluated = True
        