import numpy as np
from ._roughness_cppimpl import (
    _cppDirectionalRoughness_impl,
    _cppDirectionalRoughness_Settings_impl,
    _cppTINBasedRoughness_impl,
    _cppTINBasedRoughness_bestfit_impl,
    _cppTINBasedRoughness_againstshear_impl,
    _cppTINBasedRoughness_Settings_impl,
    _cppMeanDipRoughness_impl,
    _cppMeanDipRoughness_Settings_impl
)
from ._roughness_pyimpl import (
    _PyTINBasedRoughness,
    _PyDirectionalRoughness
)
from pandas import DataFrame

def _rs(nominal_areas,total_areas):
    return total_areas.sum()/nominal_areas.sum()

class DirRoughnessBase:
    """Base roughness class for directional roughness
    """
    def __getitem__(self,key):
        """Return a list of roughness parameters indexed by azimuth

        :param key: Roughness parameter to be returned
        :type key: str
        :raises ValueError: Error if key is not a roughness parameter
        :return: Array of roughness parameters indexed by azimuth
        :rtype: numpy.ndarray
        """
        if key in self.impl.result_keys():
            return np.array(self.impl[key])
        else:
            raise ValueError('Parameter not in DR')

    def evaluate(self,verbose=False,filename=None):
        """Start surface roughness processing

        :param verbose: Enable command line , defaults to False
        :type verbose: bool
        :param filename: Write processed surface to new file if not None, defaults to None
        :type filename: str, optional
        """
        if filename is not None:
            self.impl.evaluate(self.settings,verbose,filename)
        else:
            self.impl.evaluate(self.settings,verbose,"")
    
    def to_pandas(self):
        """Returns a pandas dataframe with azimuth indexed parameters

        :return: Pandas Dataframe containing roughness parameters 
        :rtype: pandas.Dataframe
        """
        df_data = {key:self[key] for key in self.impl.result_keys() if not key is 'az'}
        return DataFrame(df_data,index=self['az'])
    
    def to_csv(self,*args,**kwargs):
        """Save pandas dataframe generated from roughness results as csv
        Arguments are the same as pandas.DataFrame.to_csv
        """
        self.to_pandas().to_csv(*args,**kwargs)

    @property
    def final_orientation(self):
        """Final orientation after aligning to best-fit plane

        :return: Normal vector of best-fit plane
        :rtype: numpy.ndarray
        """
        return np.array(self.impl.final_orientation)

    @property
    def min_bounds(self):
        """Negative XYZ bounds of surface

        :return: Array of negative XYZ bounds
        :rtype: numpy.ndarray
        """
        return np.array(self.impl.min_bounds)
    
    @property
    def max_bounds(self):
        """Positive XYZ bounds of surface

        :return: Array of positive XYZ bounds
        :rtype: numpy.ndarray
        """
        return np.array(self.impl.max_bounds)
    
    @property
    def centroid(self):
        """Centroid of surface area

        :return: Array of centroid
        :rtype: numpy.ndarray
        """
        return np.array(self.impl.centroid)

    @property
    def shape_size(self):
        """XYZ size of surface area

        :return: Array of sizes
        :rtype: numpy.ndarray
        """
        return np.array(self.impl.shape_size)

    @property
    def total_area(self):
        """Total area of surface 

        :return: Surface area
        :rtype: numpy.ndarray
        """
        return self.impl.total_area


class _cppDirectionalRoughness(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        if triangle_mask is None:
            self.impl = _cppDirectionalRoughness_impl(points,triangles)
        else:
            self.impl = _cppDirectionalRoughness_impl(points,triangles,triangle_mask)
        self.settings = _cppDirectionalRoughness_Settings_impl()
        for key,value in kwargs.items():
            if key in [
                'n_az','az_offset',
                'n_dip_bins','fit_initialguess',
                'fit_precision','fit_regularization',
                'fit_alpha','fit_beta','min_triangles']:
                self.settings[key] = value
    
class _cppTINBasedRoughness(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        if triangle_mask is None:
            self.impl = _cppTINBasedRoughness_impl(points,triangles)
        else:
            self.impl = _cppTINBasedRoughness_impl(points,triangles,triangle_mask)
        self.settings = _cppTINBasedRoughness_Settings_impl()
        for key,value in kwargs.items():
            if key in ['n_az','az_offset','min_triangles']:
                self.settings[key] = value

class _cppTINBasedRoughness_bestfit(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        if triangle_mask is None:
            self.impl = _cppTINBasedRoughness_bestfit_impl(points,triangles)
        else:
            self.impl = _cppTINBasedRoughness_bestfit_impl(points,triangles,triangle_mask)
        self.settings = _cppTINBasedRoughness_Settings_impl()
        for key,value in kwargs.items():
            if key in ['n_az','az_offset','min_triangles']:
                self.settings[key] = value

class _cppTINBasedRoughness_againstshear(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        if triangle_mask is None:
            self.impl = _cppTINBasedRoughness_againstshear_impl(points,triangles)
        else:
            self.impl = _cppTINBasedRoughness_againstshear_impl(points,triangles,triangle_mask)
        self.settings = _cppTINBasedRoughness_Settings_impl()
        for key,value in kwargs.items():
            if key in ['n_az','az_offset','min_triangles']:
                self.settings[key] = value

class _cppMeanDipRoughness(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        if triangle_mask is None:
            self.impl = _cppMeanDipRoughness_impl(points,triangles)
        else:
            self.impl = _cppMeanDipRoughness_impl(points,triangles,triangle_mask)
        self.settings = _cppMeanDipRoughness_Settings_impl()
        for key,value in kwargs.items():
            if key in ['n_az','az_offset','min_triangles']:
                self.settings[key] = value


class pyDirectionalRoughness(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        pass

class pyTINBasedRoughness(DirRoughnessBase):
    def __init__(self,points:np.ndarray,triangles:np.ndarray,triangle_mask:np.ndarray=None,**kwargs):
        pass