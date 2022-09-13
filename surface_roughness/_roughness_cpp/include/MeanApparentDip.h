#ifndef _MEANAPPARENTDIP_H
#define _MEANAPPARENTDIP_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "Directional.h"

class MeanDipRoughness : public Directional
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
	Eigen::ArrayXXd mean_dip_;
	Eigen::ArrayXXd std_dip_;
};

#endif //_MEANAPPARENTDIP_H