#ifndef _TINBASEDROUGHNESS_BESTFIT_H
#define _TINBASEDROUGHNESS_BESTFIT_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "TINBasedRoughness.h"
#include "Directional.h"


class TINBasedRoughness_bestfit : public Directional
{
public:
	using Directional::Directional;
	void evaluate(DirectionalSetting settings = Directional::Setting(),bool verbose=false, std::string file=std::string());
	static DirectionalSetting Setting() {
		return DirectionalSetting({
			std::make_pair("n_az",72.),
			std::make_pair("az_offset",0.),
			std::make_pair("min_triangles",200)
		});
	}

private:
	std::vector<double> nominal_areas;

	// Collected parameters
	Eigen::ArrayXXd delta_a_;
	Eigen::ArrayXXd delta_star_a_;
};

#endif //_TINBASEDROUGHNESS_BESTFIT_H