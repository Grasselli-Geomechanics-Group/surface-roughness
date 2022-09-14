#include <unordered_map>
#include <functional>
#include <vector>
#include <chrono>
#include <string>
#include <format>

#include <omp.h>

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

#define RegisterDirectional(directional_class,name,module) \
    py::class_<directional_class,Directional>(module,"_cpp"##name"_impl").def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>()).def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i,Eigen::ArrayXi>());\
    module.def("_cpp"##name"_Settings_impl",[](){return directional_class::Setting();});\
    py::class_<Evaluator<directional_class>>(module,"_"##name"_Evaluator").def(py::init<Eigen::MatrixX3d,Eigen::MatrixX3i>()).def("evaluate",&Evaluator<directional_class>::evaluate)\


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

template <typename T>
class Evaluator {
public:
    Evaluator(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles) :
    points(points),triangles(triangles) {}
    std::vector<std::unique_ptr<T>> evaluate(std::vector<Eigen::ArrayXi> t_in_circle) {
        using namespace pybind11::literals;
        int n_samples = (int)t_in_circle.size();
        std::vector<std::unique_ptr<T>> output(n_samples);
        int progress = 0;
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        auto timer_format = [&]<typename LeftType, typename RightType>(LeftType minutes,RightType seconds)->std::string {
            std::string minute_string = std::to_string(minutes);
            std::string second_string = std::to_string(seconds);
            if (minutes < 10) {
                minute_string = "0"+minute_string;
            }
            if (seconds < 10) {
                second_string = "0"+second_string;
            }
            return minute_string + ":" + second_string;
        };
        auto print_progress = [&](int progress, seconds duration) {
            auto minutes = duration.count() / 60;
            auto seconds = duration.count() % 60;
            float production_time = (float)duration.count()/(progress+1);
            std::string speed = (production_time > 1)? std::to_string(production_time)+"s/it" : std::to_string(1/production_time) + "it/s";
            int remaining = int(production_time * (n_samples - progress));
            int remaining_minutes = remaining / 60;
            int remaining_seconds = remaining % 60;
            std::string print_string = 
                std::to_string(progress)+"/"+
                std::to_string(n_samples) +" ["+ 
                timer_format(minutes,seconds)+ "<" + 
                timer_format(remaining_minutes,remaining_seconds) + 
                ", " + speed;
            py::print("\r",print_string,"end"_a="]\t","sep"_a="");
        };
        seconds duration;
        #pragma omp parallel for schedule(dynamic) 
        for (int i = 0; i < n_samples; ++i) {
            output[i] = std::make_unique<T>(T(points,triangles,t_in_circle[i]));
            output[i]->evaluate();
            progress++;
            
            // Print every 1%
            if ((i*2/2 * 100 / n_samples )% 2 == 0) {
                if (omp_get_thread_num() == 0) {
                    #pragma omp critical
                    {
                        duration = duration_cast<seconds>(high_resolution_clock::now() - start);
                        print_progress(progress,duration);
                    }
                }
            }
            
        }
        print_progress(progress,duration);
        py::print("\n");
        
        return output;
    }

private:
    Eigen::MatrixX3d points;
    Eigen::MatrixX3i triangles;
};

PYBIND11_MODULE(_roughness_cppimpl,m) {
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
        .def("current_settings",&Directional::current_settings)
        .def_property_readonly("final_orientation",&Directional::get_final_orientation)
        .def_property_readonly("min_bounds",&Directional::get_min_bounds)
        .def_property_readonly("max_bounds",&Directional::get_max_bounds)
        .def_property_readonly("centroid",&Directional::get_centroid)
        .def_property_readonly("shape_size",&Directional::get_size)
        .def_property_readonly("total_area",&Directional::get_area);

    RegisterDirectional(TINBasedRoughness,"TINBasedRoughness",m);
    RegisterDirectional(DirectionalRoughness,"DirectionalRoughness",m);
    RegisterDirectional(TINBasedRoughness_bestfit,"TINBasedRoughness_bestfit",m);
    RegisterDirectional(TINBasedRoughness_againstshear,"TINBasedRoughness_againstshear",m);
    RegisterDirectional(MeanDipRoughness,"MeanDipRoughness",m);
}