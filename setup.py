import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pathlib import Path

import platform

debug = False

if debug:
    if platform.system() == "Windows":
        cpp_args=['/Od','/Zi','/openmp']
        linkargs = ['/DEBUG']
    else:
        cpp_args = []
        linkargs = []
else:
    cpp_args = ['/openmp']
    linkargs = []
 
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
        'surface_roughness/_roughness_cpp/include',
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
    extra_link_args=linkargs
)

setup(
    name="surface-roughness",
    version="0.0.1",
    description="Surface roughness calculation with Python",
    long_description=(Path(__file__).parent/"README.md").read_text(),
    author="Earl Magsipoc",
    author_email="e.magsipoc@mail.utoronto.ca",
    url="https://github.com/e-mags/pysurfaceroughness",
    license="MIT",
    package_dir = {
        'surface_roughness':'surface_roughness',
        'surface_roughness._roughness_pyimpl':'surface_roughness/_roughness_pyimpl'},
    packages=['surface_roughness','surface_roughness._roughness_pyimpl'],
    # ext_package='surface_roughness',
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
    python_requires=">=3.7"
)