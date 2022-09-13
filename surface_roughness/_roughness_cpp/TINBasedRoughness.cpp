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

TINBasedRoughness::TINBasedRoughness(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles) :
	aligned(false),
	points(points),
	triangles(triangles)
{
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
}

TINBasedRoughness::TINBasedRoughness(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles,
    Eigen::ArrayXi selected_triangles):
	aligned(false)
{
	select_triangles(this->points,points,this->triangles,triangles,selected_triangles);
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
}

void TINBasedRoughness::alignBestFit()
{
	if (!this->aligned) {
		BestFitResult result = align(this->points,this->triangles);
		this->final_orientation = result.final_orientation;
		this->min_bounds = result.min_bounds;
		this->max_bounds = result.max_bounds;
		this->centroid = result.centroid;
		this->aligned = true;
	}
}

void TINBasedRoughness::evaluate(TINBasedRoughness_settings settings, bool verbose_,std::string file_path)
{
    settings_ = settings;
    using namespace Eigen;
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
    this->total_area = calculateAreas(this->areas,this->points,this->triangles);
	typedef std::vector<Triangle> TriangleContainer;
    if (verbose_) std::cout << "Calculated areas\n";
	// 1.0 Calculate analysis directions;
	azimuths_ = M_PI / 180. * ArrayXd::LinSpaced((Index)settings_.at("n_az"),0., 360.-360./settings_.at("n_az"));
	size_t n_directions = azimuths_.size();
	azimuths_ += settings_.at("az_offset") * M_PI / 180.;
    delta_t_ = ArrayXd::Zero(n_directions);
    delta_star_t_ = ArrayXd::Zero(n_directions);
    n_facing_triangles_ = ArrayXd::Zero(n_directions);

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
        this->parameters.insert({"delta_t",delta_t_});
        this->parameters.insert({"delta*_t",delta_star_t_});
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
			triangle.set_apparent_dip(az.first(az_i), az.second(az_i));
		});
        // Filter negative apparent dip
		auto remove_it = std::remove_if(evaluation.begin(), evaluation.end(),
			[&](const Triangle& triangle) {
			return triangle.apparent_dip_angle < 0;
		});
		evaluation.erase(remove_it, evaluation.end());
        n_facing_triangles_(az_i) = (double)evaluation.size();
        if (n_facing_triangles_(az_i) > 0) {
            auto n_facing_triangles = n_facing_triangles_(az_i);

			// By total area (Delta_T)
			auto eval_area = get_area_vector(
				evaluation,[](const auto& tri) {
				return tri.area;
			});

			// Delta_T params
			auto delta_t_pair = area_params(evaluation, eval_area);
			delta_t_(az_i) = delta_t_pair.first;
			delta_star_t_(az_i) = delta_t_pair.second;
        }
    }
    this->parameters.insert({"az",azimuths_});
    this->parameters.insert({"n_tri", n_facing_triangles_});
    this->parameters.insert({"delta_t",delta_t_});
    this->parameters.insert({"delta*_t", delta_star_t_});
    if (!file_path.empty()) save_file(file_path);
    
    points.resize(0,0);
    triangles.resize(0,0);
    normals.resize(0,0);
    triangle_mask.clear();
    areas.clear();
}

bool TINBasedRoughness::save_file(std::string path) 
{
	return create_file(path,this->points,this->triangles,this->normals);
}

std::vector<std::string> TINBasedRoughness::result_keys()
{
    using namespace std;
    vector<string> keys(parameters.size());
    std::transform(parameters.begin(),parameters.end(),keys.begin(),
    [](const auto& param) { return param.first;});
    return keys;
}