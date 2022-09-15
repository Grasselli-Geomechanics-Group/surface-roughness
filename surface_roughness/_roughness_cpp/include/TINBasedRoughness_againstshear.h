#ifndef _TINBASEDROUGHNESS_AGAINSTSHEAR_H
#define _TINBASEDROUGHNESS_AGAINSTSHEAR_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "TINBasedRoughness.h"
#include "Directional.h"

class TINBasedRoughness_againstshear : public Directional
{
public:
	using Directional::Directional;
	void evaluate(DirectionalSetting settings = Directional::Setting(),bool verbose=false, std::string file=std::string());
	static DirectionalSetting Setting() {
		return DirectionalSetting({
			{"n_az",72.},
			{"az_offset",0.},
			{"min_triangles",200}
		});
	}

private:
	// Collected parameters
	Eigen::ArrayXXd delta_n_;
	Eigen::ArrayXXd delta_star_n_;
};

#endif //_TINBASEDROUGHNESS_AGAINSTSHEAR_H