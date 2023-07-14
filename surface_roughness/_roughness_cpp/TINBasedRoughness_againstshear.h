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
		DirectionalSetting setting;
        setting.set("n_az",72.);
        setting.set("az_offset",0.);
        setting.set("min_triangles",200);

        return setting;
	}

private:
	// Collected parameters
	Eigen::ArrayXXd delta_n_;
	Eigen::ArrayXXd delta_star_n_;
};

#endif //_TINBASEDROUGHNESS_AGAINSTSHEAR_H