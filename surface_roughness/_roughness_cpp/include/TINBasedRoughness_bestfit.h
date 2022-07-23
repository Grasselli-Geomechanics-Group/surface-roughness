#ifndef _TINBASEDROUGHNESS_BESTFIT_H
#define _TINBASEDROUGHNESS_BESTFIT_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>

#include <armadillo>

#include "TINBasedRoughness.h"


class TINBasedRoughness_bestfit
{
public:
	TINBasedRoughness_bestfit(const std::vector<double>& points, const std::vector<uint64_t>& triangles);
	TINBasedRoughness_bestfit(const std::vector<double>& points, const std::vector<uint64_t>& triangles, const std::vector<uint64_t>& selected_triangles);
	void evaluate(TINBasedRoughness_settings settings = TINBasedRoughness_settings(),bool verbose=false, std::string file=std::string());
    std::vector<double> operator[](std::string key) {return parameters[key];}
    std::vector<double> get_points();
    std::vector<double> get_normals();

	std::vector<double> get_min_bounds() {return min_bounds;}
	std::vector<double> get_max_bounds() {return max_bounds;}
	std::vector<double> get_centroid() {return centroid;}
	std::vector<double> get_size() {return size_;}
	double get_area() {return total_area;}

	std::vector<double> get_final_orientation() { return final_orientation; }
    std::vector<std::string> result_keys();

private:
    arma::mat points;
    arma::Mat<arma::uword> triangles;
    arma::mat normals;
	std::vector<uint8_t> triangle_mask;

    std::vector<double> areas;
    std::vector<double> nominal_areas;
    double total_area;

	bool save_file(std::string file_path);
	
	TINBasedRoughness_settings settings_;
    std::unordered_map<std::string,std::vector<double>> parameters;

	
	arma::mat pol2cart(arma::vec azimuths);
    void alignBestFit();
    void calculateNormals();
    void calculateAreas();

    arma::vec plane_fit(const arma::mat& xyz);
    arma::vec plane_normal(const arma::mat& xyz);
	std::vector<double> initial_orientation;
	std::vector<double> final_orientation;

	std::vector<double> min_bounds;
	std::vector<double> max_bounds;
	std::vector<double> centroid;
	std::vector<double> size_;

	bool aligned;

	// Collected parameters
	arma::vec azimuths_;
	arma::vec delta_a_;
	arma::vec delta_star_a_;
	arma::uvec n_facing_triangles_;
};

#endif //_TINBASEDROUGHNESS_BESTFIT_H