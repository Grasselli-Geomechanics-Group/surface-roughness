from copy import deepcopy
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
            rotation = np.arctan2(normal[0],normal[1])
            self.points = temp_p @ np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)]
            ]).T

            # Normalize profile points
            mean_spacing = np.mean(np.diff(self.points[:,0]))
            yrange =  np.arange(0,(self.points.shape[0]+1)*mean_spacing,mean_spacing)
            self.points[:,0] = yrange[:self.points.shape[0]]
            
    def __getitem__(self,key):
        return self.points[key]
    
    def center(self):
        return self.points - np.mean(self.points,axis=0)
    
    @property
    def Rq(self):
        return np.sqrt(np.mean((self.points[:,1]-np.mean(self.points[:,1]))**2))

    def gaussian_filter(self,cutoff_wavelength):
        
        def kernel(lc,spacing):
            alpha = np.sqrt(np.log(2)/np.pi)
            l = int(lc/spacing)
            l = l + 1 - l % 2
            ax = np.linspace(-(l-1)/2,(l-1)/2,l)*spacing
            kernel = np.exp(-np.pi*np.square(ax/(alpha*lc))) /(alpha*lc) * spacing
            return kernel
        k = kernel(cutoff_wavelength,self.spacing)
        k_length = k.shape[0]
        
        # Mirror profile at boundaries by k_length to treat end effects
        beginning = self.points[:k_length,1]
        x_begin = np.arange(-self.spacing*(beginning.shape[0]+1),0,self.spacing)[:beginning.shape[0],np.newaxis]
        begin_slope,_,_,_ = np.linalg.lstsq(
            np.hstack([np.ones_like(x_begin),x_begin]),
            beginning,
            rcond=None
            )
        beginning = begin_slope[1]*x_begin + beginning[0]
    
        ending = self.points[-k_length:,1]
        x_end = np.arange(self.points[-1,0],self.points[-1,0]+self.spacing*(ending.shape[0]+1),self.spacing)[:ending.shape[0],np.newaxis]
        end_slope,_,_,_ = np.linalg.lstsq(
            np.hstack([np.ones_like(x_end),x_end]),
            ending,
            rcond=None
        )
        ending = end_slope[1]*(x_end-self.points[-1,0]) + self.points[-1,1]
        complete_profile = np.hstack([beginning.flatten(),self.points[:,1],ending.flatten()])
        w = np.convolve(complete_profile,k,'same')
        return w[k_length:-k_length]
    


    def robust_gaussian_filter(self,cutoff_wavelength):
        gamma = 0.7309
        c = self.Rq*3
        c_prev = 0
        
        def regression_x(k,delta_x,n):
            xl_vec = (np.arange(n)[:,np.newaxis] - k) * delta_x

            return np.hstack([np.ones([n,1]),xl_vec,xl_vec**2])
        
        def space_variant_weighting(k,n,cut_wavelength,profile_heights,Rq,delta_x):
            xl_vec = (np.arange(n)[:,np.newaxis] - k) * delta_x
            sk = 1/(gamma*cut_wavelength)*np.exp(-np.pi*(xl_vec/(gamma*cut_wavelength))**2)
            delta = (1-(profile_heights/c)**2)**2
            delta[profile_heights > Rq*3] = 0
            return np.diag(sk[:,0]*delta)
        n = 0
        while np.abs(c-c_prev) > c_prev*.05:
            if n > 5: 
                break
            w = np.zeros([self.points.shape[0]])
            for i,k in enumerate(range(self.points.shape[0])):
                xk = regression_x(k,self.spacing,self.points.shape[0])
                sk = space_variant_weighting(k, self.points.shape[0], cutoff_wavelength, self.points[:,1], self.Rq,self.spacing)

                wk = np.array([1,0,0]) @ np.linalg.inv(xk.T @ sk @ xk) @ xk.T @ sk @ self.points[:,1]
                w[i] = wk
            c_prev = deepcopy(c)
            c = 4.4478*np.median(np.abs(self.points[:,1]-w))
            n = n + 1
        return w

    @staticmethod
    def from_csv(file):
        data = np.genfromtxt(file,delimiter=',')
        return Profile(data)
    
    @property
    def n_points(self):
        return self.points.shape[0]
    
    def hhcorr(self):
        delta_h = []
        delta_x = []
        # for i in np.geomspace(1,self.n_points):
        # for i in range(1,self.n_points):
        i = 1
        while i < self.n_points:
            delta_h.append(np.sqrt(np.sum((self.points[i:,1] - self.points[:-i,1])**2)))
            delta_x.append(i*self.spacing)
            # if i < 10:
            #     i = i * 2
            # else:
            #     i = i * 1.25
            #     i = int(i)
            i += 1
        return np.hstack([np.array(delta_x)[:,np.newaxis],np.array(delta_h)[:,np.newaxis]])
    
    def z2(self):
        diffs = np.diff(self.points[:,1])
        return np.sqrt(np.sum(diffs**2)/(diffs.shape[0]*self.spacing**2))