#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

// #include "DirectionalRoughness.h"
// #include "TINBasedRoughness_againstshear.h"
// #include "TINBasedRoughness_bestfit.h"
#include "TINBasedRoughness.h"
// #include "MeanApparentDip.h"

namespace py = pybind11;

PYBIND11_MODULE(_roughness_cppimpl,m) {
    typedef py::buffer_info info;
    // py::class_<DirectionalRoughness>(m,"_cppDirectionalRoughness_impl")
    //     .def(py::init([](
    //         const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
	// 		const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles
    //     )
    //     {
    //         info p_info = points.request();
	// 	    info t_info = triangles.request();

    //         std::vector<double> arma_points(p_info.size);
    //         for (size_t i = 0; i < arma_points.size(); ++i)
    //             arma_points.at(i) = *((double*)p_info.ptr + i);

    //         std::vector<uint64_t> arma_triangles(t_info.size);
    //         for (size_t i = 0; i < arma_triangles.size(); ++i)
	// 		    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);

    //         return DirectionalRoughness(arma_points,arma_triangles);
    //     }))
    //     .def(py::init([](
    //         const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
	// 		const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles,
    //         const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangle_mask
    //     ) {
    //         info p_info = points.request();
	// 	    info t_info = triangles.request();
    //         info m_info = triangle_mask.request();

    //         std::vector<double> arma_points(p_info.size);
    //         for (size_t i = 0; i < arma_points.size(); ++i)
    //             arma_points.at(i) = *((double*)p_info.ptr + i);

    //         std::vector<uint64_t> arma_triangles(t_info.size);
    //         for (size_t i = 0; i < arma_triangles.size(); ++i)
	// 		    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
    //         std::vector<uint64_t> arma_mask(m_info.size);
    //         for (size_t i = 0; i < arma_mask.size(); ++i)
	// 		    arma_mask.at(i) = *((uint64_t*)m_info.ptr + i);

    //         return DirectionalRoughness(arma_points,arma_triangles, arma_mask);
    //     }))
    //     .def("evaluate",&DirectionalRoughness::evaluate)
    //     .def("__getitem__",&DirectionalRoughness::operator[])
    //     .def("points",&DirectionalRoughness::get_points)
    //     .def("normals",&DirectionalRoughness::get_normals)
    //     .def("result_keys",&DirectionalRoughness::result_keys)
    //     .def_property_readonly("final_orientation",&DirectionalRoughness::get_final_orientation)
    //     .def_property_readonly("min_bounds",&DirectionalRoughness::get_min_bounds)
    //     .def_property_readonly("max_bounds",&DirectionalRoughness::get_max_bounds)
    //     .def_property_readonly("centroid",&DirectionalRoughness::get_centroid)
    //     .def_property_readonly("shape_size",&DirectionalRoughness::get_size)
    //     .def_property_readonly("total_area",&DirectionalRoughness::get_area);


    // py::class_<DirectionalRoughness_settings>(m,"_cppDirectionalRoughness_Settings_impl")
    // .def(py::init<>())
    // .def("__setitem__",&DirectionalRoughness_settings::set)
    // .def("__getitem__",&DirectionalRoughness_settings::get);

    py::class_<TINBasedRoughness>(m,"_cppTINBasedRoughness_impl")
    .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>())
    .def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>())
    .def("evaluate",&TINBasedRoughness::evaluate)
    .def("__getitem__",&TINBasedRoughness::operator[])
    .def("points",&TINBasedRoughness::get_points)
    .def("normals",&TINBasedRoughness::get_normals)
    .def("result_keys",&TINBasedRoughness::result_keys)
    .def_property_readonly("final_orientation",&TINBasedRoughness::get_final_orientation)
    .def_property_readonly("min_bounds",&TINBasedRoughness::get_min_bounds)
    .def_property_readonly("max_bounds",&TINBasedRoughness::get_max_bounds)
    .def_property_readonly("centroid",&TINBasedRoughness::get_centroid)
    .def_property_readonly("shape_size",&TINBasedRoughness::get_size)
    .def_property_readonly("total_area",&TINBasedRoughness::get_area);

//     py::class_<TINBasedRoughness_bestfit>(m,"_cppTINBasedRoughness_bestfit_impl")
//     .def(py::init([](
//             const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
// 			const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles
//         ) {
//             info p_info = points.request();
// 		    info t_info = triangles.request();

//             std::vector<double> arma_points(p_info.size);
//             for (size_t i = 0; i < arma_points.size(); ++i)
//                 arma_points.at(i) = *((double*)p_info.ptr + i);
 
//             std::vector<uint64_t> arma_triangles(t_info.size);
//             for (size_t i = 0; i < arma_triangles.size(); ++i)
// 			    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
//             return TINBasedRoughness_bestfit(arma_points,arma_triangles);
//         }))
//     .def(py::init([](
//             const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
// 			const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles,
//             const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& selected_triangles
//         ) {
//             info p_info = points.request();
// 		    info t_info = triangles.request();
//             info m_info = selected_triangles.request();

//             std::vector<double> arma_points(p_info.size);
//             for (size_t i = 0; i < arma_points.size(); ++i)
//                 arma_points.at(i) = *((double*)p_info.ptr + i);

//             std::vector<uint64_t> arma_triangles(t_info.size);
//             for (size_t i = 0; i < arma_triangles.size(); ++i)
// 			    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
//             std::vector<uint64_t> arma_mask(m_info.size);
//             for (size_t i = 0; i < arma_mask.size(); ++i)
// 			    arma_mask.at(i) = *((uint64_t*)m_info.ptr + i);
//             return TINBasedRoughness_bestfit(arma_points,arma_triangles,arma_mask);
//         }))
//     .def("evaluate",&TINBasedRoughness_bestfit::evaluate)
//     .def("__getitem__",&TINBasedRoughness_bestfit::operator[])
//     .def("points",&TINBasedRoughness_bestfit::get_points)
//     .def("normals",&TINBasedRoughness_bestfit::get_normals)
//     .def("result_keys",&TINBasedRoughness_bestfit::result_keys)
//     .def_property_readonly("final_orientation",&TINBasedRoughness_bestfit::get_final_orientation)
//     .def_property_readonly("min_bounds",&TINBasedRoughness_bestfit::get_min_bounds)
//     .def_property_readonly("max_bounds",&TINBasedRoughness_bestfit::get_max_bounds)
//     .def_property_readonly("centroid",&TINBasedRoughness_bestfit::get_centroid)
//     .def_property_readonly("shape_size",&TINBasedRoughness_bestfit::get_size)
//     .def_property_readonly("total_area",&TINBasedRoughness_bestfit::get_area);

//     py::class_<TINBasedRoughness_againstshear>(m,"_cppTINBasedRoughness_againstshear_impl")
//     .def(py::init([](
//             const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
// 			const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles
//         ) {
//             info p_info = points.request();
// 		    info t_info = triangles.request();

//             std::vector<double> arma_points(p_info.size);
//             for (size_t i = 0; i < arma_points.size(); ++i)
//                 arma_points.at(i) = *((double*)p_info.ptr + i);
 
//             std::vector<uint64_t> arma_triangles(t_info.size);
//             for (size_t i = 0; i < arma_triangles.size(); ++i)
// 			    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
//             return TINBasedRoughness_againstshear(arma_points,arma_triangles);
//         }))
//     .def(py::init([](
//             const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
// 			const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles,
//             const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& selected_triangles
//         ) {
//             info p_info = points.request();
// 		    info t_info = triangles.request();
//             info m_info = selected_triangles.request();

//             std::vector<double> arma_points(p_info.size);
//             for (size_t i = 0; i < arma_points.size(); ++i)
//                 arma_points.at(i) = *((double*)p_info.ptr + i);

//             std::vector<uint64_t> arma_triangles(t_info.size);
//             for (size_t i = 0; i < arma_triangles.size(); ++i)
// 			    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
//             std::vector<uint64_t> arma_mask(m_info.size);
//             for (size_t i = 0; i < arma_mask.size(); ++i)
// 			    arma_mask.at(i) = *((uint64_t*)m_info.ptr + i);
//             return TINBasedRoughness_againstshear(arma_points,arma_triangles,arma_mask);
//         }))
//     .def("evaluate",&TINBasedRoughness_againstshear::evaluate)
//     .def("__getitem__",&TINBasedRoughness_againstshear::operator[])
//     .def("points",&TINBasedRoughness_againstshear::get_points)
//     .def("normals",&TINBasedRoughness_againstshear::get_normals)
//     .def("result_keys",&TINBasedRoughness_againstshear::result_keys)
//     .def_property_readonly("final_orientation",&TINBasedRoughness_againstshear::get_final_orientation)
//     .def_property_readonly("min_bounds",&TINBasedRoughness_againstshear::get_min_bounds)
//     .def_property_readonly("max_bounds",&TINBasedRoughness_againstshear::get_max_bounds)
//     .def_property_readonly("centroid",&TINBasedRoughness_againstshear::get_centroid)
//     .def_property_readonly("shape_size",&TINBasedRoughness_againstshear::get_size)
//     .def_property_readonly("total_area",&TINBasedRoughness_againstshear::get_area);

//     py::class_<MeanDipRoughness>(m,"_cppMeanDipRoughness_impl")
//     .def(py::init([](
//             const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
// 			const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles
//         ) {
//             info p_info = points.request();
// 		    info t_info = triangles.request();

//             std::vector<double> arma_points(p_info.size);
//             for (size_t i = 0; i < arma_points.size(); ++i)
//                 arma_points.at(i) = *((double*)p_info.ptr + i);
 
//             std::vector<uint64_t> arma_triangles(t_info.size);
//             for (size_t i = 0; i < arma_triangles.size(); ++i)
// 			    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
//             return MeanDipRoughness(arma_points,arma_triangles);
//         }))
//     .def(py::init([](
//             const py::array_t<double, py::array::f_style | py::array::forcecast>& points,
// 			const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& triangles,
//             const py::array_t<uint64_t, py::array::f_style | py::array::forcecast>& selected_triangles
//         ) {
//             info p_info = points.request();
// 		    info t_info = triangles.request();
//             info m_info = selected_triangles.request();

//             std::vector<double> arma_points(p_info.size);
//             for (size_t i = 0; i < arma_points.size(); ++i)
//                 arma_points.at(i) = *((double*)p_info.ptr + i);

//             std::vector<uint64_t> arma_triangles(t_info.size);
//             for (size_t i = 0; i < arma_triangles.size(); ++i)
// 			    arma_triangles.at(i) = *((uint64_t*)t_info.ptr + i);
                
//             std::vector<uint64_t> arma_mask(m_info.size);
//             for (size_t i = 0; i < arma_mask.size(); ++i)
// 			    arma_mask.at(i) = *((uint64_t*)m_info.ptr + i);
//             return MeanDipRoughness(arma_points,arma_triangles,arma_mask);
//         }))
//     .def("evaluate",&MeanDipRoughness::evaluate)
//     .def("__getitem__",&MeanDipRoughness::operator[])
//     .def("points",&MeanDipRoughness::get_points)
//     .def("normals",&MeanDipRoughness::get_normals)
//     .def("result_keys",&MeanDipRoughness::result_keys)
//     .def_property_readonly("final_orientation",&MeanDipRoughness::get_final_orientation)
//     .def_property_readonly("min_bounds",&MeanDipRoughness::get_min_bounds)
//     .def_property_readonly("max_bounds",&MeanDipRoughness::get_max_bounds)
//     .def_property_readonly("centroid",&MeanDipRoughness::get_centroid)
//     .def_property_readonly("shape_size",&MeanDipRoughness::get_size)
//     .def_property_readonly("total_area",&MeanDipRoughness::get_area);

//     py::class_<MeanDipRoughness_settings>(m,"_cppMeanDipRoughness_Settings_impl")
//     .def(py::init<>())
//     .def("__setitem__",&MeanDipRoughness_settings::set)
//     .def("__getitem__",&MeanDipRoughness_settings::get);

    py::class_<TINBasedRoughness_settings>(m,"_cppTINBasedRoughness_Settings_impl")
    .def(py::init<>())
    .def("__setitem__",&TINBasedRoughness_settings::set)
    .def("__getitem__",&TINBasedRoughness_settings::get);
}