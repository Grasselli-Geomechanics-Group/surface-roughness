#ifndef _TINBASEDROUGHNESS_BESTFIT_H
#define _TINBASEDROUGHNESS_BESTFIT_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "TINBasedRoughness.h"


class TINBasedRoughness_bestfit
{
public:
	TINBasedRoughness_bestfit(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles);
	TINBasedRoughness_bestfit(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles, Eigen::ArrayXi selected_triangles);
	void evaluate(TINBasedRoughness_settings settings = TINBasedRoughness_settings(),bool verbose=false, std::string file=std::string());
    Eigen::ArrayXd operator[](std::string key) {return parameters[key];}
    Eigen::MatrixX3d get_points() {return points;}
    Eigen::MatrixX3d get_normals() {return normals;}

	Eigen::Vector3d get_min_bounds() {return min_bounds;}
	Eigen::Vector3d get_max_bounds() {return max_bounds;}
	Eigen::Vector3d get_centroid() {return centroid;}
	std::vector<double> get_size() {return size_;}
	double get_area() {return total_area;}

	Eigen::Vector3d get_final_orientation() { return final_orientation; }
    std::vector<std::string> result_keys();

private:
    Eigen::MatrixX3d points;
    Eigen::MatrixX3i triangles;
    Eigen::MatrixX3d normals;
	std::vector<uint8_t> triangle_mask;

    std::vector<double> areas;
    std::vector<double> nominal_areas;
    double total_area;

	bool save_file(std::string file_path);
	
	TINBasedRoughness_settings settings_;
    std::unordered_map<std::string,Eigen::ArrayXd> parameters;
	
    void alignBestFit();

	Eigen::Vector3d initial_orientation;
	Eigen::Vector3d final_orientation;

	Eigen::Vector3d min_bounds;
	Eigen::Vector3d max_bounds;
	Eigen::Vector3d centroid;
	std::vector<double> size_;

	bool aligned;

	// Collected parameters
	Eigen::ArrayXd azimuths_;
	Eigen::ArrayXd delta_a_;
	Eigen::ArrayXd delta_star_a_;
	Eigen::ArrayXd n_facing_triangles_;
};

#endif //_TINBASEDROUGHNESS_BESTFIT_H