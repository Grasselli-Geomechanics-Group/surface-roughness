import numpy as np
from surface_roughness import Surface

surface = Surface(path=r'X:\Git\src_calculator_v2\example_surface.stl')
surface.evaluate_dr()

x = np.array([1.        , 0.94302884, 0.88605769, 0.82908653, 0.77211538,
       0.71514422, 0.65817307, 0.60120191, 0.54423076, 0.4872596 ,
       0.43028845, 0.37331729, 0.31634614, 0.25937498, 0.20240383,
       0.14543267, 0.08846152, 0.03149036])
y = np.array([1.00000000e+00, 8.01729402e-01, 6.27712070e-01, 4.81505241e-01,
       3.61672764e-01, 2.59157072e-01, 1.75732963e-01, 1.16110356e-01,
       7.19314442e-02, 4.33150493e-02, 2.30797273e-02, 1.11164741e-02,
       5.75058342e-03, 3.17587373e-03, 1.65439156e-03, 8.33811453e-04,
       5.54868226e-04, 2.25196163e-04])
delta_max = 3

# https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
nu1 = 0.1
nu2 = 0.25
nu3 = 0.75
t1 = 0.25
t2 = 2

c_0 = 1
k = 1
def powerlaw(c) -> np.ndarray:
    return x**c

def powerlaw_dx(c,degree) -> np.ndarray:
    return powerlaw(c)*np.log(x)**degree

def f(c):
    return np.sum((powerlaw(c) - y)**2)

def df(c):
    curve = powerlaw(c)
    return np.sum(2*curve*np.log(x)*(curve-y))

def ddf(c):
    curve = powerlaw(c)
    return np.sum(2*curve*np.log(x)**2*(2*curve-y))

def m(c,dc):
    return f(c) + df(c) * dc + dc**2 * ddf(c) /2

def rho_k(current_c, dc):
    return (f(current_c) - f(current_c+dc))/(f(current_c)-m(current_c,dc))

def find_dc(c_now,delta_k):
    b = df(c_now)
    a = ddf(c_now)
    # Vertex find quadratic eqn derived taylor approx. of cost fcn
    v = -b/(2*a)
    if abs(v) > delta_k:
        return delta_k*v/abs(v)
    else:
        return v
    
def trust_region_stepping(current_c,delta_current):
    current_dc = find_dc(current_c,delta_current)
    current_rho = rho_k(current_c,current_dc)
    while current_dc > 10e-10:
        if current_rho > nu3:
            delta_current *= t2
        elif current_rho < nu1:
            delta_current *= t1
        current_c += current_dc
        print(current_c,current_dc,delta_current)
        current_dc = find_dc(current_c,delta_current)
    return current_dc

trust_region_stepping(c_0,delta_max)