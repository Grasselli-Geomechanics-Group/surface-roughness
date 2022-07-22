#include "TINBasedRoughness.h"

#include <math.h>
#include <numeric>
#include <execution>
#include <iterator>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

TINBasedRoughness::TINBasedRoughness(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles) :
	aligned(false),
	points(points),
	triangles(triangles)
{
    this->alignBestFit();
    this->calculateNormals();
}

// TINBasedRoughness::TINBasedRoughness(
//     const std::vector<double>& points, 
//     const std::vector<uint64_t>& triangles,
//     const std::vector<uint64_t>& selected_triangles):
// 	aligned(false)
// {
// 	using namespace arma;
	
// 	arma::uword n_triangles = (arma::uword)selected_triangles.size();
// 	size_t triangles_in_n_rows = triangles.size()/3;
// 	this->triangles.resize(n_triangles,3);
// 	for (auto tri_it = selected_triangles.begin(); tri_it != selected_triangles.end(); ++tri_it) {
// 		arma::uword index = (arma::uword) std::distance(selected_triangles.begin(),tri_it);
// 		this->triangles.row(index) = 
// 			{triangles.at(*tri_it),
// 			triangles.at(*tri_it + triangles_in_n_rows),
// 			triangles.at(*tri_it + 2*triangles_in_n_rows)};
// 	}

// 	// Get vector of all unique points
// 	arma::Col<uint64_t> point_indices = vectorise(this->triangles);
// 	point_indices = sort(unique(point_indices));
// 	std::vector<std::pair<uint64_t,uint64_t>> p_init;
// 	std::vector<uint64_t> new_vals(point_indices.n_rows); std::iota(new_vals.begin(),new_vals.end(), 0);
// 	std::transform(
// 		point_indices.begin(),point_indices.end(),
// 		new_vals.begin(),std::back_inserter(p_init),
// 		[](const auto& a, const auto& b) 
// 		{ return std::make_pair(a,b); });

// 	std::unordered_map<uint64_t,uint64_t> pindex_find(p_init.begin(),p_init.end());
	
// 	// Copy points
// 	arma::uword n_points = points.size()/3;
// 	this->points.resize(point_indices.n_rows, 3);
// 	for (auto point_index = point_indices.begin(); point_index != point_indices.end(); ++point_index)
// 		this->points.row(point_index - point_indices.begin()) = 
// 		{points.at(*point_index),
// 		points.at(*point_index + n_points),
// 		points.at(*point_index + 2*n_points)};
	
// 	// Reconfigure triangles to current point index
// 	this->triangles.for_each([&](Mat<uint64_t>::elem_type& val) {
// 		val = pindex_find.at(val);
// 	});

//     this->alignBestFit();
//     this->calculateNormals();
// }

Eigen::Vector3d TINBasedRoughness::plane_fit(const Eigen::MatrixX3d& xyz) {
    using namespace Eigen;
    // Plane fit methodology
    // https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    Matrix3d U;
    Vector3d s;
    Matrix3d V;
    
	MatrixX3d data = xyz;
    RowVector3d centroid = data.colwise().mean();
    data.rowwise() -= centroid;

	JacobiSVD<Matrix3Xd, ComputeThinU | ComputeThinV> svd(data.transpose());
    
	Vector3d normal = svd.matrixU().col(2);
	if (normal(2) < 0) normal *= -1.;
    // [ n_x ][ n_y ][ n_z ]
    return normal;
}

Eigen::Vector3d TINBasedRoughness::plane_normal(const Eigen::MatrixX3d& xyz) {
    using namespace Eigen;
    Vector3d fit = plane_fit(xyz);
    return fit.normalized();
}

void TINBasedRoughness::alignBestFit()
{
	if (!this->aligned) {
		using namespace Eigen;
		Vector3d centroid = this->points.colwise().mean();

		this->initial_orientation = plane_normal(this->points);
		Vector3d current_orientation = initial_orientation;
		// Rotate 3 times to improve accuracy
		for (int rep = 0; rep < 3; ++rep) {
			double theta = -std::asin(
				current_orientation(0)*current_orientation(0) +
				current_orientation(1)*current_orientation(1));

			Vector3d rot_axis(current_orientation(1),-current_orientation(0),0);
			rot_axis = rot_axis.normalized();

			// Calculate rotation matrix
			AngleAxis<double> rotation(theta,rot_axis);
			
			// Rotate points
			this->points = (rotation.matrix() * this->points.transpose()).transpose();
			
			current_orientation = plane_normal(points);
		}
		typedef std::vector<double> std_dvec;
		this->final_orientation = current_orientation;

		this->min_bounds = points.colwise().minCoeff();
		this->max_bounds = points.colwise().maxCoeff();
		this->centroid = points.colwise().mean();
		this->aligned = true;
	}
}

void TINBasedRoughness::calculateNormals()
{
    using namespace Eigen;
	normals.resize(triangles.rows(),NoChange);
	std::transform(
		triangles.rowwise().cbegin(),triangles.rowwise().cend(),
		this->normals.rowwise().begin(),
		[&](const auto& row){
			Vector3d V1V2 = points.row(row(1)) - points.row(row(0));
			Vector3d V1V3 = points.row(row(2)) - points.row(row(0));
			return V1V2.cross(V1V3).normalized();
		}
	);
}

void TINBasedRoughness::calculateAreas()
{
    using namespace Eigen;
	areas.resize(triangles.rows());
	std::transform(
		triangles.rowwise().cbegin(),triangles.rowwise().cend(),
		areas.begin(),
		[&](const auto& row) -> double {
			Vector3d V1V2 = points.row(row(1)) - points.row(row(0));
			Vector3d V1V3 = points.row(row(2)) - points.row(row(0));
			return 0.5*V1V2.cross(V1V3).norm();
		});

    this->total_area = std::accumulate(areas.begin(), areas.end(), 0.0);
}

std::vector<double> get_area_vector(
	std::vector<TIN_triangle> evaluation, 
	std::function<double(TIN_triangle)> area_op)
{
	std::vector<double> area_vec;
	std::transform(evaluation.begin(),evaluation.end(),
	std::back_inserter(area_vec),area_op);
	return area_vec;
}

std::pair<double,double> area_params(
	std::vector<TIN_triangle> evaluation,
	std::vector<double> area_vector)
{
	// Convert apparent dip to degrees
	std::vector<double> eval_appdip;
	std::transform(
		evaluation.begin(),evaluation.end(),
		std::back_inserter(eval_appdip),[](const auto& tri) {
		return tri.apparent_dip_angle*180.0/M_PI;
	});

	// Collect total area
	double total_area = std::reduce(area_vector.begin(),area_vector.end(),0.,std::plus<double>());

	// Multiply and sum apparent dip and area  (theta*A)
	std::vector<double> areadip;
	std::transform(
		eval_appdip.begin(),eval_appdip.end(),
		area_vector.begin(),std::back_inserter(areadip),
		std::multiplies<double>());
	double delta_t_top = std::reduce(areadip.begin(),areadip.end(),0.,std::plus<double>());

	// Multiply areadip with apparent dip again (theta^2*A)
	std::vector<double> areadipdip;
	std::transform(areadip.begin(),areadip.end(),eval_appdip.begin(),std::back_inserter(areadipdip),std::multiplies<double>());
	double delta_star_t_top = std::reduce(areadipdip.begin(),areadipdip.end(),0.,std::plus<double>());

	double delta_star= sqrt(delta_star_t_top/total_area);
	double delta = delta_t_top/total_area;
	return std::make_pair(delta,delta_star);
}

void TINBasedRoughness::evaluate(TINBasedRoughness_settings settings, bool verbose_,std::string file_path)
{
    settings_ = settings;
    using namespace Eigen;
    this->alignBestFit();
    this->calculateNormals();
    this->calculateAreas();
	typedef std::vector<TIN_triangle> TriangleContainer;
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
		[&](const auto& norm_row,const double& area) -> TIN_triangle {
			TIN_triangle triangle;
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

	MatrixX2d cartesian_az = pol2cart(azimuths_);

	for (int az_i = 0; az_i < azimuths_.size(); ++az_i) {
		if (verbose_) std::cout << "Calculated az" << std::to_string(az_i) << "\n";
        TriangleContainer evaluation;
		std::copy(dir_triangle.begin(), dir_triangle.end(), std::back_inserter(evaluation));

		// Calculate apparent dip for each analysis direction
		std::for_each(evaluation.begin(), evaluation.end(),
			[&az_i, &cartesian_az](TIN_triangle& triangle) {
			triangle.set_apparent_dip(cartesian_az(az_i, 0), cartesian_az(az_i, 1));
		});
        // Filter negative apparent dip
		auto remove_it = std::remove_if(evaluation.begin(), evaluation.end(),
			[&](const TIN_triangle& triangle) {
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
	using namespace std;
	ofstream stlfile(path);
	if (stlfile.is_open()) {
		stlfile << "solid " + path + "\n";

		for (Eigen::Index triangle = 0; triangle < triangles.rows(); ++triangle) {
			stlfile << "facet normal " +
				to_string(normals(triangle, 0)) + " " +
				to_string(normals(triangle, 1)) + " " +
				to_string(normals(triangle, 2)) + "\n";

			stlfile << "\touter loop\n";
			for (Eigen::Index vertex = 0; vertex < 3; ++vertex) {
				stlfile << "\t\tvertex " +
					to_string(points(triangles(triangle, vertex), 0)) + " " +
					to_string(points(triangles(triangle, vertex), 1)) + " " +
					to_string(points(triangles(triangle, vertex), 2)) + "\n";
			}
			stlfile << "\tendloop\n";
			stlfile << "endfacet\n";
		}
		stlfile << "endsolid " + path;
		return true;
	}
	else {
		cerr << "Unable to open " + path + " for STL write\n";
		return false;
	}
}

std::vector<std::string> TINBasedRoughness::result_keys()
{
    using namespace std;
    vector<string> keys(parameters.size());
    std::transform(parameters.begin(),parameters.end(),keys.begin(),
    [](const auto& param) { return param.first;});
    return keys;
}

Eigen::MatrixX2d TINBasedRoughness::pol2cart(Eigen::ArrayXd azimuth) 
{
	using namespace Eigen;
	// Polar to cartesian transformation by vector
	ArrayXd Tx = azimuth.cos();
	ArrayXd Ty = azimuth.sin();

	MatrixX2d T(azimuth.size(),2);
	T << Tx,Ty;
	return T;
}