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
		DirectionalSetting setting;
		setting.set("n_az",72.);
		setting.set("az_offset",0.);
		setting.set("n_dip_bins",90);
		setting.set("fit_initialguess",1);
		setting.set("fit_precision",6);
		setting.set("fit_regularization",10e-10);
		setting.set("fit_alpha",0.01);
		setting.set("fit_beta",0.5);
		setting.set("min_triangles",200);
		return setting;
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