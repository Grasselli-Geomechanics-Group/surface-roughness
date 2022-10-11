import numpy as np

from scipy.optimize import curve_fit

def power_law(x,a,b):
    return a*np.power(x,b)

def powerlaw_fitting(powerlaw_data):
    parameters,cov = curve_fit(f=power_law,xdata=powerlaw_data[:,0],ydata=powerlaw_data[:,1],p0 = [0,0],bounds=(-np.inf,np.inf))
    stdevs = np.sqrt(np.diag(cov))
    return parameters, stdevs

class Profile:
    def __init__(self,point_table=None):
        if point_table is not None:
            self.points = point_table
            self.spacing = self.points[1,0] - self.points[0,0]

            centroid = np.mean(self.points,axis=0)
            temp_p = self.points - centroid
            u,s,_ = np.linalg.svd(temp_p.T,full_matrices=False)
            normal = u[:,np.argmin(s)]
            normal /= np.linalg.norm(normal)
            rotation = -np.arctan2(normal[0],normal[1])
            self.points = temp_p @ np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)]
            ]).T + centroid
            
    def __getitem__(self,key):
        return self.points[key]
    
    @property
    def n_points(self):
        return self.points.shape[0]
    
    def hhcorr(self):
        delta_h = []
        delta_x = []
        for i in range(1,self.n_points):
            delta_h.append(np.sqrt(np.sum((self.points[i:,1] - self.points[:-i,1])**2)))
            delta_x.append(i*self.spacing)
        return np.hstack([np.array(delta_x)[:,np.newaxis],np.array(delta_h)[:,np.newaxis]])