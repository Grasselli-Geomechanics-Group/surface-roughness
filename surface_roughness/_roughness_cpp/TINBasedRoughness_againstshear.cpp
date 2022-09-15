#include "TINBasedRoughness_againstshear.h"
#include "TINBasedRoughness.h"

#include <math.h>
#include <numeric>
#include <execution>
#include <iterator>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "DirectionalUtil.h"
#include "Directional.h"

void TINBasedRoughness_againstshear::evaluate(DirectionalSetting settings, bool verbose_,std::string file_path)
{
    settings_ = settings;
    using namespace Eigen;
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
    this->total_area = calculateAreas(this->areas,this->points,this->triangles);
	typedef std::vector<Triangle> TriangleContainer;
    if (verbose_) std::cout << "Calculated areas\n";
	// 1.0 Calculate analysis directions;
	azimuths_.resize((Index)settings_.at("n_az"),1);
	double step = 0;
	for (auto az:azimuths_.rowwise()) {
		az = Matrix<double,1,1>(step);
		step += 2*M_PI/settings_.at("n_az") + settings.at("az_offset")*M_PI/180;
	}		size_t n_directions = azimuths_.size();
	delta_n_ = ArrayXXd::Zero(n_directions,1);
    delta_star_n_ = ArrayXXd::Zero(n_directions,1);
    n_facing_triangles_ = ArrayXXd::Zero(n_directions,1);

    if (verbose_) std::cout << "Calculated analysis directions\n";
	// 2.0 Create triangles for analysis
	TriangleContainer dir_triangle(triangles.rows());
	unsigned int counter = 0;

	std::transform(
		this->normals.rowwise().begin(),this->normals.rowwise().end(),
		this->areas.begin(),
		dir_triangle.begin(),
		[&](const auto& norm_row,const double& area) -> Triangle {
			Triangle triangle;
			triangle.index = counter++;
			triangle.area = area;
			triangle.set_normal(norm_row);
			return triangle;
		});

	if (verbose_) std::cout << "Created triangle containers\n";

    if (this->triangles.rows() < settings_["min_triangles"]) {
        this->parameters.insert({"az",azimuths_});
        this->parameters.insert({"n_tri", n_facing_triangles_});
		this->parameters.insert({"delta_n",delta_n_});
		this->parameters.insert({"delta*_n",delta_star_n_});
		return;
	}

	// MatrixX2d cartesian_az = pol2cart(azimuths_);
	std::pair<ArrayXd,ArrayXd> az = pol2cart(azimuths_);

	for (Index az_i = 0; az_i < azimuths_.size(); ++az_i) {
		if (verbose_) std::cout << "Calculated az" << std::to_string(az_i) << "\n";
        TriangleContainer evaluation;
		evaluation.reserve(dir_triangle.size());
		std::copy(dir_triangle.begin(), dir_triangle.end(), std::back_inserter(evaluation));

		// Calculate apparent dip for each analysis direction
		std::for_each(evaluation.begin(), evaluation.end(),
			[&az_i, &az](Triangle& triangle) {
			triangle.set_apparent_dip(az.first(az_i), az.second(az_i), true);
		});
        // Filter negative apparent dip
		auto remove_it = std::remove_if(evaluation.begin(), evaluation.end(),
			[&](const Triangle& triangle) {
			return triangle.apparent_dip_angle < 0;
		});
		evaluation.erase(remove_it, evaluation.end());
        n_facing_triangles_(az_i) = (double) evaluation.size();
        if (n_facing_triangles_(az_i) > 0) {
            auto n_facing_triangles = n_facing_triangles_(az_i);

			// By projected area against shear (Delta_N)
			auto eval_againstshear_area = get_area_vector(
				evaluation,[](const auto& tri) {
					return tri.against_shear_area;
				}
			);

			// Delta_N params
			auto delta_n_pair = area_params(evaluation, eval_againstshear_area);
			delta_n_(az_i) = delta_n_pair.first;
			delta_star_n_(az_i) = delta_n_pair.second;
        }
    }
    this->parameters.insert({"az",azimuths_});
    this->parameters.insert({"n_tri", n_facing_triangles_});
    this->parameters.insert({"delta_n",delta_n_});
    this->parameters.insert({"delta*_n", delta_star_n_});
    if (!file_path.empty()) save_file(file_path);
    
    points.resize(0,0);
    triangles.resize(0,0);
    normals.resize(0,0);
    triangle_mask.clear();
    areas.clear();
}
