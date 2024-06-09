#ifndef _DIRECTIONALROUGHNESS_H
#define _DIRECTIONALROUGHNESS_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "Directional.h"

class DirectionalRoughness : public Directional
{
public:
	using Directional::Directional;
	void evaluate(DirectionalSetting settings = Directional::Setting(),bool verbose=false, std::string file=std::string());
	static DirectionalSetting Setting() {
		return DirectionalSetting({
			std::make_pair("n_az", 72.),
			std::make_pair("az_offset",0.),
			std::make_pair("n_dip_bins",90.),
			std::make_pair("fit_initialguess",1.),
			std::make_pair("fit_precision",6.),
			std::make_pair("fit_regularization",10e-10),
			std::make_pair("fit_alpha",0.01),
			std::make_pair("fit_beta",0.5),
			std::make_pair("min_triangles",200)
		});
	}

private:
	double C_param_GGFit_3D(Eigen::Index az_i);
	double C_param_Newton_opt_backtracking_Fit(Eigen::Index az_i);
	double calculateGOFPValue(Eigen::VectorXd& observed, double C, double theta_max, double a_0);

	// Collected parameters
	Eigen::ArrayXXd C_fitting_;
	Eigen::ArrayXXd A_0_parameter_;
	Eigen::ArrayXXd theta_max_;
	Eigen::ArrayXXd n_facing_triangles_;
	Eigen::ArrayXXd dip_bins_data_;
	Eigen::ArrayXXd bins_;
	Eigen::ArrayXXd chi_gof_;
};
#endif //_DIRECTIONALROUGHNESS_H