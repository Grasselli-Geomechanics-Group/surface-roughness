from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os
import numpy

import platform

debug = False

if platform.system() == "Windows":
    cpp_args=['/Od' if debug else '/O2', 
              '/Oi',
              '/W3',
              '/sdl',
              '/MD',
              '/Gy',
              '/std:c++17',
              '/Zi' if debug else '' # Remove after debugging
            ]
else:
    cpp_args = ['-std=c++17']
 
def env_path(env_var,relative_path):
    return os.path.join(os.environ[env_var],relative_path)
root_folder = os.getcwd()

if platform.system() == "Windows":
    armadillo_include = env_path('ARMADILLO_ROOT','include')
    openblas_lib = env_path('ARMADILLO_ROOT','examples/lib_win64')
    roughness_cppimpl_sources = [
        "surface_roughness/_roughness_cppimpl.cpp",
        "surface_roughness/_roughness_cpp/DirectionalRoughness.cpp",
        "surface_roughness/_roughness_cpp/TINBasedRoughness.cpp",
        "surface_roughness/_roughness_cpp/TINBasedRoughness_bestfit.cpp",
        "surface_roughness/_roughness_cpp/TINBasedRoughness_againstshear.cpp",
        "surface_roughness/_roughness_cpp/MeanApparentDip.cpp"

    ]
    roughness_cppimpl_includes = [
            numpy.get_include(),
            armadillo_include,
            'surface_roughness/_roughness_cpp/include'
    ]
    roughness_cppimpl_depends = roughness_cppimpl_sources+roughness_cppimpl_includes
    roughness_cppimpl = Pybind11Extension(
        "_roughness_cppimpl",
        sources=roughness_cppimpl_sources,
        depends=roughness_cppimpl_depends,
        include_dirs=roughness_cppimpl_includes,
        libraries=[
            'libopenblas'
        ],
        library_dirs=[openblas_lib],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=['/DEBUG'] if debug else [] 
    )

    setup(
        name="surface_roughness",
        version="0.0.1",
        description="Surface roughness calculation with Python",
        author="Earl Magsipoc",
        author_email="e.magsipoc@mail.utoronto.ca",
        license="MIT",
        package_dir = {
            'surface_roughness':'surface_roughness',
            'surface_roughness._roughness_pyimpl':'surface_roughness/_roughness_pyimpl'},
        packages=['surface_roughness','surface_roughness._roughness_pyimpl'],
        ext_package='surface_roughness',
        ext_modules=[roughness_cppimpl],
        package_data={
            'surface_roughness':['libopenblas.dll']},
        install_requires=[
            'scipy',
            'meshio',
            'tqdm',
            'numpy',
            'numexpr',
            'pandas',
            'pptk',
            'matplotlib',
            'pyevtk'
        ]
    )