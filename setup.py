import glob
from setuptools import setup
from pybind11.setup_helpers import (
    Pybind11Extension, 
    build_ext, 
    ParallelCompile, 
    naive_recompile
)
from pathlib import Path

import platform

debug = False
openmp = True

ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

cpp_args = []
linkargs = []
libs = []

if platform.system() == "Windows":
    cpp_args.extend(['/std:c++20', '/MD'])
    
    if debug:
        cpp_args.extend(['/Od','/Zi'])
        linkargs.extend(['/DEBUG'])
        
    else:
        cpp_args.extend(['/O2', '/Ot'])
        
    if openmp:
        cpp_args.append('/openmp')
        
elif platform.system() == "Linux":
    cpp_args.extend(['-std=c++20'])
    
    if debug:
        cpp_args.extend(['-O0'])
        
    else:
        cpp_args.extend(['-O3'])
        
    if openmp:
        cpp_args.append('-fopenmp')
        libs.append('gomp')
        
else:
    # disable openmp for non-linux/windows systems
    cpp_args.extend(['-std=c++20', '-O3'])
    
 
roughness_cppimpl_sources = [
    "surface_roughness/_roughness_cppimpl.cpp",
    "surface_roughness/_roughness_cpp/Directional.cpp",
    "surface_roughness/_roughness_cpp/DirectionalRoughness.cpp",
    "surface_roughness/_roughness_cpp/TINBasedRoughness.cpp",
    "surface_roughness/_roughness_cpp/TINBasedRoughness_bestfit.cpp",
    "surface_roughness/_roughness_cpp/TINBasedRoughness_againstshear.cpp",
    "surface_roughness/_roughness_cpp/MeanApparentDip.cpp"
]
roughness_cppimpl_includes = [
        'surface_roughness/_roughness_cpp',
        'eigen'
]
headers = []
[headers.extend(glob.glob(f+"/*.h")) for f in roughness_cppimpl_includes]
roughness_cppimpl_depends = roughness_cppimpl_sources+headers
roughness_cppimpl = Pybind11Extension(
    "surface_roughness._roughness_cppimpl",
    sources=roughness_cppimpl_sources,
    depends=roughness_cppimpl_depends,
    include_dirs=roughness_cppimpl_includes,
    language='c++',
    extra_compile_args=cpp_args,
    extra_link_args=linkargs,
    libraries=libs
)

setup(
    name="surface-roughness",
    version="0.0.3",
    description="Surface roughness calculation with Python",
    long_description=(Path(__file__).parent/"README.md").read_text(),
    long_description_content_type='text/markdown',
    author="Earl Magsipoc",
    author_email="e.magsipoc@mail.utoronto.ca",
    url="https://github.com/e-mags/pysurfaceroughness",
    license="MIT",
    package_dir = {
        'surface_roughness':'surface_roughness',
        'surface_roughness._roughness_pyimpl':'surface_roughness/_roughness_pyimpl'},
    packages=['surface_roughness','surface_roughness._roughness_pyimpl'],
    ext_modules=[roughness_cppimpl],
    install_requires=[
        'scipy',
        'meshio',
        'tqdm',
        'numpy',
        'numexpr',
        'pandas',
        'matplotlib',
        'mplstereonet',
        'shapely'
    ],
    cmdclass={"build_ext":build_ext},
    zip_safe=False,
    python_requires=">=3.9"
)
