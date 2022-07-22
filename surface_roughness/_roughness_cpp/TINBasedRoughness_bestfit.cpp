#include "TINBasedRoughness_bestfit.h"
#include "TINBasedRoughness.h"

#include <math.h>
#include <numeric>
#include <execution>
#include <iterator>
#include <functional>

#include <armadillo>

TINBasedRoughness_bestfit::TINBasedRoughness_bestfit(
    const std::vector<double>& points, 
    const std::vector<uint64_t>& triangles) :
	aligned(false)
{
    using namespace arma;
    this->points = conv_to<mat>::from(points);
    this->points.reshape(points.size()/3,3);
    this->triangles = conv_to<u64_mat>::from(triangles);
    this->triangles.reshape(triangles.size()/3,3);
    this->alignBestFit();
    this->calculateNormals();
}

TINBasedRoughness_bestfit::TINBasedRoughness_bestfit(
    const std::vector<double>& points, 
    const std::vector<uint64_t>& triangles,
    const std::vector<uint64_t>& selected_triangles):
	aligned(false)
{
	using namespace arma;
	
	arma::uword n_triangles = (arma::uword)selected_triangles.size();
	size_t triangles_in_n_rows = triangles.size()/3;
	this->triangles.resize(n_triangles,3);
	for (auto tri_it = selected_triangles.begin(); tri_it != selected_triangles.end(); ++tri_it) {
		arma::uword index = (arma::uword) std::distance(selected_triangles.begin(),tri_it);
		this->triangles.row(index) = 
			{triangles.at(size_t(*tri_it)),
			triangles.at(size_t(*tri_it) + triangles_in_n_rows),
			triangles.at(size_t(*tri_it) + 2*triangles_in_n_rows)};
	}

	// Get vector of all unique points
	arma::Col<uint64_t> point_indices = vectorise(this->triangles);
	point_indices = sort(unique(point_indices));
	std::vector<std::pair<uint64_t,uint64_t>> p_init;
	std::vector<uint64_t> new_vals(point_indices.n_rows); std::iota(new_vals.begin(),new_vals.end(), 0);
	std::transform(
		point_indices.begin(),point_indices.end(),
		new_vals.begin(),std::back_inserter(p_init),
		[](const auto& a, const auto& b) 
		{ return std::make_pair(a,b); });

	std::unordered_map<uint64_t,uint64_t> pindex_find(p_init.begin(),p_init.end());
	
	// Copy points
	arma::uword n_points = points.size()/3;
	this->points.resize(point_indices.n_rows, 3);
	for (auto point_index = point_indices.begin(); point_index != point_indices.end(); ++point_index)
		this->points.row(point_index - point_indices.begin()) = 
		{points.at(*point_index),
		points.at(*point_index + n_points),
		points.at(*point_index + 2*n_points)};
	
	// Reconfigure triangles to current point index
	this->triangles.for_each([&](Mat<uint64_t>::elem_type& val) {
		val = pindex_find.at(val);
	});

    this->alignBestFit();
    this->calculateNormals();
}

std::vector<double> TINBasedRoughness_bestfit::get_points()
{
    arma::rowvec pout = points.as_row();
    return arma::conv_to<std::vector<double>>::from(pout);
}

std::vector<double> TINBasedRoughness_bestfit::get_normals()
{
    arma::rowvec nout = normals.as_row();
    return arma::conv_to<std::vector<double>>::from(nout);
}

arma::vec TINBasedRoughness_bestfit::plane_fit(const arma::mat& xyz) {
    using namespace arma;
    // Plane fit methodology
    // https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    dmat U;
    dvec s;
    dmat V;
    dmat data;
    if (xyz.n_cols == 3) {
        data = xyz.t();
    }
    else {
        data = xyz;
    }
    dvec centroid = mean(data,1);
    data.each_col() -= centroid;

    svd_econ(U, s, V, data,"left");
    dvec normal = U.col(s.index_min());
    double d = -1.0*sum(centroid % normal);
    dvec plane_norm = { normal(0),normal(1),d };
    plane_norm /= -normal(2);
    // Return  (vector order not same as MATLAB)
    // vector z = n_x * x + n_y * y + n_0
    // [ n_x ]
    // [ n_y ]
    // [ n_0 ]
    return plane_norm;
}

arma::vec TINBasedRoughness_bestfit::plane_normal(const arma::mat& xyz) {
    using namespace arma;
    dvec fit = plane_fit(xyz);
    dvec normal = { -fit(0), -fit(1), 1 };
    return normalise(normal);
}

void TINBasedRoughness_bestfit::alignBestFit()
{
	if (!aligned) {
		using namespace arma;
		arma::rowvec centroid = mean(this->points);

		arma::vec initial_orientation = plane_normal(points);
		vec current_orientation = initial_orientation;
		// Rotate 3 times to improve accuracy
		for (int rep = 0; rep < 3; ++rep) {
			
			double sin_theta = std::sqrt(
				current_orientation(0)*current_orientation(0) +
				current_orientation(1)*current_orientation(1));
			double cos_theta = std::sqrt(1.0 - sin_theta * sin_theta);

			arma::vec rot_axis = { current_orientation(1),-current_orientation(0),0 };
			rot_axis = normalise(rot_axis);
			double l = rot_axis(0);
			double m = rot_axis(1);
			double n = rot_axis(2);

			// Calculate rotation matrix
			arma::mat rot_matrix = {
				{l*l*(1.0 - cos_theta) + cos_theta, m*l*(1.0 - cos_theta) - n * sin_theta, n*l*(1.0 - cos_theta) + m * sin_theta},
				{l*m*(1.0 - cos_theta) + n * sin_theta, m*m*(1.0 - cos_theta) + cos_theta, n*m*(1.0 - cos_theta) - l * sin_theta},
				{l*n*(1.0 - cos_theta) - m * sin_theta, m*n*(1.0 - cos_theta) + l * sin_theta, n*n*(1.0 - cos_theta) + cos_theta}
			};
			
			points.each_row();
			// Rotate points
			points.each_row([&rot_matrix](rowvec& point) {
				point = trans(rot_matrix * point.t());
			});
			
			current_orientation = plane_normal(points);
		}
		typedef std::vector<double> std_dvec;
		final_orientation = arma::conv_to<std_dvec>::from(current_orientation);

		min_bounds = arma::conv_to<std_dvec>::from(trans(min(points)));
		max_bounds = arma::conv_to<std_dvec>::from(trans(max(points)));
		this->centroid = arma::conv_to<std_dvec>::from(trans(mean(points)));
		std::transform(
			max_bounds.begin(),max_bounds.end(),
			min_bounds.begin(),std::back_inserter(size_),
			std::minus<double>());
		aligned = true;
	}
}

void TINBasedRoughness_bestfit::calculateNormals()
{
    using namespace arma;
    normals.zeros(triangles.n_rows,3);
    for (arma::uword triangle_i = 0; triangle_i < triangles.n_rows; ++triangle_i) {
        rowvec V1V2 = points.row(triangles(triangle_i, 1)) - points.row(triangles(triangle_i, 0));
        rowvec V1V3 = points.row(triangles(triangle_i, 2)) - points.row(triangles(triangle_i, 0));
        normals.row(triangle_i) = normalise(cross(V1V2, V1V3).t()).t();
    }
}

void TINBasedRoughness_bestfit::calculateAreas()
{
    using namespace arma;

    triangles.each_row([&](Row<uint64_t>& triangle) {
        rowvec V1V2(3);
        rowvec V1V3(3);
        V1V2 = points.row(triangle(1)) - points.row(triangle(0));
        V1V3 = points.row(triangle(2)) - points.row(triangle(0));
        areas.push_back(0.5*norm(cross(V1V2, V1V3)));
        V1V2(2) = 0;
        V1V3(2) = 0;
        nominal_areas.push_back(0.5*norm(cross(V1V2, V1V3)));
    });
    total_area = std::accumulate(areas.begin(), areas.end(), 0.0);
}

std::vector<double> get_area_vector(
	std::vector<TIN_triangle_bestfit> evaluation, 
	std::function<double(TIN_triangle_bestfit)> area_op)
{
	std::vector<double> area_vec;
	std::transform(evaluation.begin(),evaluation.end(),
	std::back_inserter(area_vec),area_op);
	return area_vec;
}

std::pair<double,double> area_params(
	std::vector<TIN_triangle_bestfit> evaluation,
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

void TINBasedRoughness_bestfit::evaluate(TINBasedRoughness_settings settings, bool verbose_,std::string file_path)
{
    settings_ = settings;
    using namespace arma;
    this->alignBestFit();
    this->calculateNormals();
    this->calculateAreas();
	typedef std::vector<TIN_triangle_bestfit> TriangleContainer;
    if (verbose_) std::cout << "Calculated areas\n";
	// 1.0 Calculate analysis directions;
	azimuths_ = datum::pi / 180 * linspace(0, 360-360/settings_.at("n_az"), (arma::uword)settings_.at("n_az"));
	auto size_az = size(azimuths_);
	arma::uword n_directions = azimuths_.n_elem;
	azimuths_ += settings_.at("az_offset") * datum::pi / 180;
	delta_a_.zeros(n_directions);
    delta_star_a_.zeros(n_directions);
    n_facing_triangles_.zeros(n_directions);

    if (verbose_) std::cout << "Calculated analysis directions\n";
	// 2.0 Create triangles for analysis
	TriangleContainer dir_triangle(triangles.n_rows);
	auto normal_x_iter = this->normals.begin_col(0);
	auto normal_y_iter = this->normals.begin_col(1);
	auto normal_z_iter = this->normals.begin_col(2);
	auto area_iter = this->nominal_areas.begin(); // Track best-fit area instead of total area
	unsigned int counter = 0;
	auto tri_0_iter = this->triangles.begin_col(0);
	auto tri_1_iter = this->triangles.begin_col(1);
	auto tri_2_iter = this->triangles.begin_col(2);

	std::for_each(dir_triangle.begin(), dir_triangle.end(), 
	[&](TIN_triangle_bestfit& triangle) {
        triangle.index = counter++;
        triangle.area = *area_iter++;	
        triangle.set_normal(*normal_x_iter++, *normal_y_iter++, *normal_z_iter++);
	});

	if (verbose_) std::cout << "Created triangle containers\n";

    if (this->triangles.n_rows < settings_["min_triangles"]) {
		typedef std::vector<double> stdvec;
        this->parameters.insert({"az",conv_to<stdvec>::from(azimuths_)});
        this->parameters.insert({"n_tri", conv_to<stdvec>::from(n_facing_triangles_)});
		this->parameters.insert({"delta_a",conv_to<stdvec>::from(delta_a_)});
		this->parameters.insert({"delta*_a", conv_to<stdvec>::from(delta_star_a_)});
		return;
	}

	mat cartesian_az = pol2cart(azimuths_);

	for (size_t az_i = 0; az_i < azimuths_.n_elem; ++az_i) {
		if (verbose_) std::cout << "Calculated az" << std::to_string(az_i) << "\n";
        TriangleContainer evaluation;
		std::copy(dir_triangle.begin(), dir_triangle.end(), std::back_inserter(evaluation));

		// Calculate apparent dip for each analysis direction
		std::for_each(evaluation.begin(), evaluation.end(),
			[&az_i, &cartesian_az](TIN_triangle_bestfit& triangle) {
			triangle.set_apparent_dip(cartesian_az(az_i, 0), cartesian_az(az_i, 1));
		});
        // Filter negative apparent dip
		auto remove_it = std::remove_if(evaluation.begin(), evaluation.end(),
			[&](const TIN_triangle_bestfit& triangle) {
			return triangle.apparent_dip_angle < 0;
		});
		evaluation.erase(remove_it, evaluation.end());
        n_facing_triangles_(az_i) = (uword) evaluation.size();
        if (n_facing_triangles_(az_i) > 0) {
            auto n_facing_triangles = n_facing_triangles_(az_i);

			// By best-fit projected area (Delta_A)
			auto eval_bestfit_area = get_area_vector(
				evaluation,[](const auto& tri) {
					return tri.area;
				}
			);

			// Delta_A params
			auto delta_a_pair = area_params(evaluation, eval_bestfit_area);
			delta_a_(az_i) = delta_a_pair.first;
			delta_star_a_(az_i) = delta_a_pair.second;

        }
    }
    typedef std::vector<double> stdvec;
    this->parameters.insert({"az",conv_to<stdvec>::from(azimuths_)});
    this->parameters.insert({"n_tri", conv_to<stdvec>::from(n_facing_triangles_)});
    this->parameters.insert({"delta_a",conv_to<stdvec>::from(delta_a_)});
    this->parameters.insert({"delta*_a", conv_to<stdvec>::from(delta_star_a_)});

    if (!file_path.empty()) save_file(file_path);
    
    points.clear();
    triangles.clear();
    normals.clear();
    triangle_mask.clear();
    areas.clear();
}

bool TINBasedRoughness_bestfit::save_file(std::string path) 
{
	using namespace std;
	ofstream stlfile(path);
	if (stlfile.is_open()) {
		stlfile << "solid " + path + "\n";

		for (arma::uword triangle = 0; triangle < triangles.n_rows; ++triangle) {
			stlfile << "facet normal " +
				to_string(normals(triangle, 0)) + " " +
				to_string(normals(triangle, 1)) + " " +
				to_string(normals(triangle, 2)) + "\n";

			stlfile << "\touter loop\n";
			for (arma::uword vertex = 0; vertex < 3; ++vertex) {
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

std::vector<std::string> TINBasedRoughness_bestfit::result_keys()
{
    using namespace std;
    vector<string> keys(parameters.size());
    std::transform(parameters.begin(),parameters.end(),keys.begin(),
    [](const auto& param) { return param.first;});
    return keys;
}

arma::mat TINBasedRoughness_bestfit::pol2cart(arma::vec azimuth) 
{
	using namespace arma;
	// Polar to cartesian transformation by vector
	vec Tx = cos(azimuth);
	vec Ty = sin(azimuth);

	mat T;
	T = join_horiz(Tx, Ty);
	return T;
}