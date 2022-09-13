#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include "Directional.h"

#include "DirectionalRoughness.h"
#include "TINBasedRoughness_againstshear.h"
#include "TINBasedRoughness_bestfit.h"
#include "TINBasedRoughness.h"
#include "MeanApparentDip.h"

namespace py = pybind11;

class PyDirectional : public Directional {
    public:
    using Directional::Directional;
    void evaluate(DirectionalSetting settings,bool verbose, std::string file) override {
        PYBIND11_OVERRIDE_PURE(
            void,
            Directional,
            evaluate,
            settings,
            verbose,
            file
        );
    }
};

PYBIND11_MODULE(_roughness_cppimpl,m) {
    typedef py::buffer_info info;
    py::class_<DirectionalSetting>(m,"_cppDirectionalSetting_impl")
        .def(py::init<>())
        .def("__setitem__",&DirectionalSetting::set)
        .def("__getitem__",&DirectionalSetting::get);

    py::class_<Directional,PyDirectional>(m,"_cppDirectional_impl")
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>())
        .def("evaluate",&Directional::evaluate)
        .def("__getitem__",&Directional::operator[])
        .def("points",&Directional::get_points)
        .def("normals",&Directional::get_normals)
        .def("result_keys",&Directional::result_keys)
        .def_property_readonly("final_orientation",&Directional::get_final_orientation)
        .def_property_readonly("min_bounds",&Directional::get_min_bounds)
        .def_property_readonly("max_bounds",&Directional::get_max_bounds)
        .def_property_readonly("centroid",&Directional::get_centroid)
        .def_property_readonly("shape_size",&Directional::get_size)
        .def_property_readonly("total_area",&Directional::get_area);

    m.def("_cppDirectionalRoughness_Settings_impl",[](){return DirectionalRoughness::Setting();});
    m.def("_cppTINBasedRoughness_Settings_impl",[](){return TINBasedRoughness::Setting();});
    m.def("_cppMeanDipRoughness_Settings_impl",[](){return MeanDipRoughness::Setting();});


    py::class_<DirectionalRoughness,Directional>(m,"_cppDirectionalRoughness_impl")
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>());
;
    py::class_<TINBasedRoughness,Directional>(m,"_cppTINBasedRoughness_impl")
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>());

    py::class_<TINBasedRoughness_bestfit,Directional>(m,"_cppTINBasedRoughness_bestfit_impl")
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>());

    py::class_<TINBasedRoughness_againstshear,Directional>(m,"_cppTINBasedRoughness_againstshear_impl")
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>());

    py::class_<MeanDipRoughness,Directional>(m,"_cppMeanDipRoughness_impl")
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
        .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>());

}