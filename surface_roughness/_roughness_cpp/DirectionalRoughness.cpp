#include "DirectionalRoughness.h"

#include <math.h>
#include <numeric>
#include <algorithm>
#include <iterator>

#include <armadillo>

DirectionalRoughness::DirectionalRoughness(
	const std::vector<double>& points, 
	const std::vector<uint64_t>& triangles) :
	aligned(false)
{
	using namespace arma;
	this->points = conv_to<mat>::from(points);
	this->points = reshape(this->points,points.size()/3, 3);
	this->triangles = conv_to<Mat<uint64_t>>::from(triangles);
	this->triangles = reshape(this->triangles,triangles.size()/3, 3);
	this->alignBestFit();
    this->calculateNormals();
}

DirectionalRoughness::DirectionalRoughness(
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
			{triangles.at(*tri_it),
			triangles.at(*tri_it + triangles_in_n_rows),
			triangles.at(*tri_it + 2*triangles_in_n_rows)};
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

std::vector<double> DirectionalRoughness::get_points()
{
    arma::rowvec pout = points.as_row();
    return arma::conv_to<std::vector<double>>::from(pout);
}

std::vector<double> DirectionalRoughness::get_normals()
{
    arma::rowvec nout = normals.as_row();
    return arma::conv_to<std::vector<double>>::from(nout);
}

arma::vec DirectionalRoughness::plane_fit(const arma::mat& xyz) {
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

arma::vec DirectionalRoughness::plane_normal(const arma::mat& xyz) {
    using namespace arma;
    dvec fit = plane_fit(xyz);
    dvec normal = { -fit(0), -fit(1), 1 };
    return normalise(normal);
}

void DirectionalRoughness::alignBestFit()
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

void DirectionalRoughness::calculateNormals()
{
    using namespace arma;
    normals.zeros(triangles.n_rows,3);
    for (arma::uword triangle_i = 0; triangle_i < triangles.n_rows; ++triangle_i) {
        rowvec V1V2 = points.row(triangles(triangle_i, 1)) - points.row(triangles(triangle_i, 0));
        rowvec V1V3 = points.row(triangles(triangle_i, 2)) - points.row(triangles(triangle_i, 0));
        normals.row(triangle_i) = normalise(cross(V1V2, V1V3).t()).t();
    }
}

void DirectionalRoughness::calculateAreas()
{
    using namespace arma;
    triangles.each_row([&](Row<uint64_t>& triangle) {
        rowvec V1V2(3);
        rowvec V1V3(3);
        V1V2 = points.row(triangle(1)) - points.row(triangle(0));
        V1V3 = points.row(triangle(2)) - points.row(triangle(0));
        areas.push_back(0.5*norm(cross(V1V2, V1V3)));
    });
    total_area = std::accumulate(areas.begin(), areas.end(), 0.0);
}

void DirectionalRoughness::evaluate(DirectionalRoughness_settings settings, bool verbose_,std::string file_path) 
{
	settings_ = settings;
	using namespace arma;
    this->alignBestFit();
    this->calculateNormals();
    this->calculateAreas();
	typedef std::vector<Directional_triangle> TriangleContainer;
	if (verbose_) std::cout << "Calculated areas\n";
	// 1.0 Calculate analysis directions;
	azimuths_ = datum::pi / 180 * linspace(0, 360-360/settings_.at("n_az"), (arma::uword)settings_.at("n_az"));
	auto size_az = size(azimuths_);
	uword n_directions = azimuths_.n_elem;
	azimuths_ += settings.at("az_offset") * datum::pi / 180;
	C_fitting_.zeros(n_directions);
	A_0_parameter_.zeros(n_directions);
	theta_max_.zeros(n_directions);
	n_facing_triangles_.zeros(n_directions);
	dip_bins_data_.zeros(n_directions, (uword)settings.at("n_dip_bins")+1);
	chi_gof_.zeros(n_directions);

	if (verbose_) std::cout << "Calculated analysis directions\n";
	// 2.0 Create triangles for analysis
	TriangleContainer dir_triangle(triangles.n_rows);
	auto normal_x_iter = this->normals.begin_col(0);
	auto normal_y_iter = this->normals.begin_col(1);
	auto normal_z_iter = this->normals.begin_col(2);
	auto area_iter = this->areas.begin();
	unsigned int counter = 0;
	std::for_each(dir_triangle.begin(), dir_triangle.end(), [&normal_x_iter, &normal_y_iter, &normal_z_iter, &area_iter,&counter](Directional_triangle& triangle) {
		triangle.index = counter++;
		triangle.area = *area_iter++;	
		triangle.set_normal(*normal_x_iter++, *normal_y_iter++, *normal_z_iter++);
	});

	if (verbose_) std::cout << "Created triangle containers\n";
	
	if (this->triangles.n_rows < settings_["min_triangles"]) {
		typedef std::vector<double> stdvec;
		this->parameters.insert({"az",conv_to<stdvec>::from(azimuths_)});
    	this->parameters.insert({"n_tri", conv_to<stdvec>::from(n_facing_triangles_)});
		this->parameters.insert({"c",conv_to<stdvec>::from(C_fitting_)});
		this->parameters.insert({"theta_max",conv_to<stdvec>::from(theta_max_)});
		this->parameters.insert({"a0",conv_to<stdvec>::from(A_0_parameter_)});
		this->parameters.insert({"thetamax_cp1",conv_to<stdvec>::from(theta_max_ / (C_fitting_ + 1))});
		this->parameters.insert({"gof", conv_to<stdvec>::from(chi_gof_)});
		return;
	}

	mat cartesian_az = pol2cart(azimuths_);
	bins_ = linspace(0, datum::pi / 2, (uword)settings.at("n_dip_bins") + 1);
	for (size_t az_i = 0; az_i < azimuths_.n_elem; ++az_i) {
		if (verbose_) std::cout << "Calculated az" << std::to_string(az_i) << "\n";
		TriangleContainer evaluation;
		std::copy(dir_triangle.begin(), dir_triangle.end(), std::back_inserter(evaluation));

		// Calculate apparent dip for each analysis direction
		std::for_each(evaluation.begin(), evaluation.end(),
			[&az_i, &cartesian_az](Directional_triangle& triangle) {
			triangle.set_apparent_dip(cartesian_az(az_i, 0), cartesian_az(az_i, 1));
		});

		// Filter negative apparent dip
		auto remove_it = std::remove_if(evaluation.begin(), evaluation.end(),
			[&](const Directional_triangle& triangle) {
			return triangle.apparent_dip_angle < 0;
		});
		evaluation.erase(remove_it, evaluation.end());
		std::sort(evaluation.begin(), evaluation.end(),
			[](const Directional_triangle& a, const Directional_triangle& b) {
			return a.apparent_dip_angle < b.apparent_dip_angle;
		});

		auto save_start = evaluation.begin();
		int bin_number = 0;

		if (evaluation.size() > 0) {
			double a_star;
			for (TriangleContainer::iterator eval_it = std::next(evaluation.begin()); eval_it != evaluation.end(); ++eval_it) {
				// This only works because the triangles are sorted by increasing apparent dip. Sort command is crucial at lines above
				// We save the current largest bin as bins_(bin_number+1). This is the limit for bins_ accessible to us.

				// First seek the next largest bin. If we don't have the largest bin, we just increment the save_last iterator
				if (eval_it->apparent_dip_angle > bins_(bin_number + 1)) {
					// Once we find the next largest bin, we do heavier work.
					// Eval_it will correspond to the next largest binsize. It will not be included
					// in addition due to the bounds of for_each --> [begin,end)

					// Check what the last largest bin size should be
					while (eval_it->apparent_dip_angle > bins_(bin_number + 1))
						bin_number++;

					//Add the areas of the previous triangles to
					// a_star. This area will be added to all of the currently accessible bins_.
					a_star = std::accumulate(save_start, eval_it, 0.0,
						[](double a, const Directional_triangle& b) {
						return a + b.area;
					});
		
					// We no longer consider areas of triangles already included
					// Continue from previous
					save_start = eval_it;

					// Add area determined by a_star to accesible bins_
					for (int bin_i = 0; bin_i < bin_number; ++bin_i) dip_bins_data_(az_i, bin_i) += a_star;
				}
			}
			// Get leftovers after for loop
			if (save_start != evaluation.end()) {
				a_star = std::accumulate(save_start, evaluation.end(),0.0,
					[](double a, const Directional_triangle& b) {
					return a + b.area;
				});

				for (int bin_i = 0; bin_i <= bin_number; ++bin_i) dip_bins_data_(az_i, bin_i) += a_star;
			}
			n_facing_triangles_(az_i) = evaluation.size();
			if (n_facing_triangles_(az_i) > 0) {
				// Divide all areas by the total area
				dip_bins_data_.row(az_i) /= this->total_area;

				// Get max theta !!(MUST BE SORTED ASCENDING)
				theta_max_(az_i) = evaluation.back().apparent_dip_angle * 180 / datum::pi;

				A_0_parameter_(az_i) = dip_bins_data_(az_i, 0);

				// Fit power curve
				C_fitting_(az_i) = C_param_Newton_opt_backtracking_Fit(az_i);

				// Calculate chi-sq GOF
				vec observations = trans(dip_bins_data_.row(az_i));
				chi_gof_(az_i) = calculateGOFPValue(observations, C_fitting_(az_i), theta_max_(az_i), A_0_parameter_(az_i));
			}
			else {
				theta_max_(az_i) = 0;
				A_0_parameter_(az_i) = 0;
				C_fitting_(az_i) = 0;
				chi_gof_(az_i) = 99999;
			}
		}
	}
	if (verbose_) std::cout << "Saving data\n";
    typedef std::vector<double> stdvec;
    this->parameters.insert({"az",conv_to<stdvec>::from(azimuths_)});
    this->parameters.insert({"n_tri", conv_to<stdvec>::from(n_facing_triangles_)});
	this->parameters.insert({"c",conv_to<stdvec>::from(C_fitting_)});
    this->parameters.insert({"theta_max",conv_to<stdvec>::from(theta_max_)});
    this->parameters.insert({"a0",conv_to<stdvec>::from(A_0_parameter_)});
    this->parameters.insert({"thetamax_cp1",conv_to<stdvec>::from(theta_max_ / (C_fitting_ + 1))});
    this->parameters.insert({"gof", conv_to<stdvec>::from(chi_gof_)});
	if (!file_path.empty()) save_file(file_path);
	points.clear();
    triangles.clear();
    normals.clear();
    triangle_mask.clear();
    areas.clear();
	
}

bool DirectionalRoughness::save_file(std::string path) 
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

std::vector<std::string> DirectionalRoughness::result_keys()
{
    using namespace std;
    vector<string> keys(parameters.size());
    std::transform(parameters.begin(),parameters.end(),keys.begin(),
    [](const auto& param) { return param.first;});
    return keys;
}

double gamma(double Z)
{
	// Approximation
	using namespace std;
	const double inv_e = 1 / M_E;
	
	double D = 1.0 / (10.0 * Z);
	D = 1 / ((12 * Z) - D);
	D = (D + Z)*inv_e;
	D = pow(D, Z);
	D *= sqrt(M_2_PI / Z);
	return D;
}

double chiPValue(double S, double Z)
{
	using namespace std;
	if (Z < 0.0) return 0;
	double Sc = (1.0 / S);
	Sc *= pow(Z, S);
	Sc *= exp(-Z);
	double sum = 1.0; double nom = 1.0; double denominator = 1.0;
	for (int i = 0; i < 200; ++i) {
		nom *= Z;
		S++;
		denominator *= S;
		sum += nom / denominator;
	}
	return sum * Sc;
}

double chisqr(int DOF, double var)
{
	using namespace std;
	if (var < 0 || DOF < 1) return 0.0;
	double K = (double)DOF * 0.5;
	double X = var * 0.5;
	double p_value = chiPValue(K, X);
	if (isnan(p_value) || isinf(p_value)) return 0.0;
	p_value /= gamma(K);
	return 1 - p_value;
}

double DirectionalRoughness::calculateGOFPValue(arma::vec& observed, double C, double theta_max, double a_0)
{
	using namespace arma;
	// http://www.stat.yale.edu/Courses/1997-98/101/chigf.htm
	// https://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
	vec expected(observed.n_elem);
	for (uword i = 0; i < bins_.n_elem; ++i)
		if (bins_(i)+1 < theta_max)
			expected(i) = a_0 * pow((1 - bins_(i) / theta_max), C);
	double chi_sq = sum((observed - expected)%(observed - expected) / expected);

	return chisqr((unsigned int)bins_.n_elem - 1, chi_sq);
}

arma::mat DirectionalRoughness::pol2cart(arma::vec azimuth) 
{
	using namespace arma;
	// Polar to cartesian transformation by vector
	vec Tx = cos(azimuth);
	vec Ty = sin(azimuth);

	mat T;
	T = join_horiz(Tx, Ty);
	return T;
}

double estimate_dc(double c, arma::vec& x, arma::vec& y, double regularization=10e-7, double reduction=1) {
	using namespace arma;
	vec curve = pow(x,c);
	vec logx = log(x);
	double top = sum(2.*curve%logx%(curve-y));
	double bot = sum(2.*curve%square(logx)%(2.*curve-y));
	return -reduction * top/(bot + regularization);
}

inline double f(arma::vec& x, arma::vec& y, double c)
{
	using namespace arma;
	return sum(square(pow(x,c)-y));
}

inline double dfdc(arma::vec& x, arma::vec& y, double c)
{
	using namespace arma;
	vec curve = pow(x,c);
	return sum(2.*pow(x,c)%log(x)%(pow(x,c)-y));
}

double backtrack_search(arma::vec& x, arma::vec& y,double c, double dc, double reduction, double alpha, double beta)
{
	using namespace arma;
	vec curve = pow(x,c);
	double v = estimate_dc(c,x,y); // v is non-reduced dc
	double f_left = f(x,y,c+reduction*dc);
	double f_right = f(x,y,c) + alpha * reduction * dfdc(x,y,c)*dc;

	return (f_left > f_right)? beta * reduction : reduction;
}

double DirectionalRoughness::C_param_Newton_opt_backtracking_Fit(arma::uword az_i) {
	// https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/14-newton.pdf
	// https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
	using namespace arma;
	vec deg_bins = bins_ * 180 / datum::pi;
	uvec valid_values = find(deg_bins <= theta_max_(az_i));
	deg_bins.resize(valid_values.max());
	vec fit_data = dip_bins_data_.row(az_i).t();
	
	fit_data.resize(valid_values.max());
	vec y = fit_data / A_0_parameter_(az_i);
	vec x = 1 - deg_bins / theta_max_(az_i);
	std::vector<double> x_test = conv_to<std::vector<double>>::from(x);
	std::vector<double> y_test = conv_to<std::vector<double>>::from(y);
	double c = settings_.at("fit_initialguess");
	double dc = estimate_dc(c,x,y,settings_.at("fit_regularization"));
	c += dc;
	double reduction = 1;
	double alpha = settings_.at("fit_alpha");
	double beta = settings_.at("fit_beta");
	int stop = 0;
	while (std::abs(reduction*dc) > std::pow(10,-settings_["fit_precision"])) {
		dc = estimate_dc(c+dc,x,y,settings_.at("fit_regularization"),reduction);
		c += reduction*dc;
		reduction = backtrack_search(x,y,c,dc,reduction,alpha,beta);
		stop += 1;
		if (stop > 500) return c;
	}
	return c;
}

double DirectionalRoughness::C_param_GGFit_3D(arma::uword az_i) {
	using namespace arma;
	/*  Original equation
	*   y = Annot(1 - x / theta_max)^c
	*   Find c for best fit
	*   Convert equation to use theta_max and A_0_parameter_
	*   dip_bins_data_/Annot = (1 - bins_/theta_max)^c
	*   y = x^c
	*   residuals r = y - x^c
	*   https://en.wikipedia.org/wiki/Gauss�Newton_algorithm
	*
	*/
	vec deg_bins = bins_ * 180 / datum::pi;
	uvec valid_values = find(deg_bins <= theta_max_(az_i));
	deg_bins.resize(valid_values.max());
	vec fit_data = dip_bins_data_.row(az_i).t();
	
	fit_data.resize(valid_values.max());
	vec y = fit_data / A_0_parameter_(az_i);
	vec x = 1 - deg_bins / theta_max_(az_i);

	// Using Gauss-newton algorithm
	// Initial guess run
	vec beta(2);
	beta(0) = settings_.at("fit_initialguess");
	// Residuals
	mat r(y.n_elem, 2);
	r.col(0) = y - pow(x, beta(0));
	// Sum of squares
	vec ssq(2);
	ssq(0) = accu(pow(r.col(0), 2));
	mat J_r(y.n_elem, 2);
	J_r.col(0) = -log(x) % pow(x, beta(0));
	{
		// Beta_1 = Beta_0 - (J_r' * J_r)^-1 * J_r' * r(Beta_0);
		vec J_rcol = J_r.col(0); // n x 1
		vec r_col = r.col(0);   // n x 1
		vec result;
		try {
			rowvec pseudoinverse = solve(trans(J_rcol)*J_rcol, trans(J_rcol), solve_opts::no_approx); // (1 x n) * (n x 1) * (1 x n)
			result = pseudoinverse * r_col;
		}
		catch (std::runtime_error e) {
			result = 99999;
		}
		beta(1) = beta(0) - result(0);
		r.col(1) = y - pow(x, beta(1));
		ssq(1) = accu(pow(r.col(1), 2));
		J_r.col(1) = -log(x) % pow(x, beta(1));
	}

	uword beta_i = 1;
	while (true) {
		// Check answers
		// Enforce decreasing residual
		//if (!(ssq(beta_i) < ssq(beta_i-1))) {
		//    beta_i--;
		//    break;
		//}
		// Convergence condition by value change
		if (std::abs(beta(beta_i) - beta(beta_i - 1)) == 0) break;
		// Timeout condition
		else if (beta_i > 500) break;

		++beta_i;// Increment beta_i

		// Preallocating for next step
		r.resize(y.n_elem, beta_i + 1);
		J_r.resize(x.n_elem, beta_i + 1);
		ssq.resize(beta_i + 1);
		beta.resize(beta_i + 1);

		// Beta_1 = Beta_0 - (J_r' * J_r)^-1 * J_r' * r(Beta_0);
		vec J_rcol = J_r.col(beta_i - 1); // n x 1
		vec r_col = r.col(beta_i - 1);   // n x 1
		rowvec pseudoinverse = solve(trans(J_rcol)*J_rcol, trans(J_rcol)); // (1 x n) * (n x 1) * (1 x n)
		vec result = pseudoinverse * r_col;

		beta(beta_i) = beta(beta_i - 1) - result(0); // 1 x 1 - (1 x n) * (n x 1)

		// Set r = y_i - x_i ^Beta_0
		r.col(beta_i) = y - pow(x, beta(beta_i));

		// sum of squares
		ssq(beta_i) = accu(pow(r.col(beta_i), 2));

		// Set Jacobian vector
		// J_r = -ln(x) * [x_i^Beta_0];
		J_r.col(beta_i) = -log(x) % pow(x, beta(beta_i));
	}
	
	return beta(beta_i);
}
