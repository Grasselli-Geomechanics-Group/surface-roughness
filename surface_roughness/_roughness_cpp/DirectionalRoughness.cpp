#include "DirectionalRoughness.h"

#include <math.h>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <vector>

#include <Eigen/Core>

#include "DirectionalUtil.h"

DirectionalRoughness::DirectionalRoughness(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles) :
	aligned(false),
	points(points),
	triangles(triangles)
{
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
}


DirectionalRoughness::DirectionalRoughness(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles,
    Eigen::ArrayXi selected_triangles):
	aligned(false)
{
	select_triangles(this->points,points,this->triangles,triangles,selected_triangles);
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
}

void DirectionalRoughness::alignBestFit()
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

void DirectionalRoughness::evaluate(DirectionalRoughness_settings settings, bool verbose_,std::string file_path) 
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
	}
	size_t n_directions = azimuths_.size();
	C_fitting_ = ArrayXXd::Zero(n_directions,1);
	A_0_parameter_ = ArrayXXd::Zero(n_directions,1);
	theta_max_ = ArrayXXd::Zero(n_directions,1);
	n_facing_triangles_ = ArrayXXd::Zero(n_directions,1);
	dip_bins_data_ = ArrayXXd::Zero(n_directions, (Index)settings.at("n_dip_bins")+1);
	chi_gof_ = ArrayXXd::Zero(n_directions,1);

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
		this->parameters.insert({"c",C_fitting_});
		this->parameters.insert({"theta_max",theta_max_});
		this->parameters.insert({"a0",A_0_parameter_});
		this->parameters.insert({"thetamax_cp1",(theta_max_ / (C_fitting_ + 1))});
		this->parameters.insert({"gof", chi_gof_});
		return;
	}

	MatrixX2d cartesian_az = pol2cart(azimuths_);
	bins_.resize((Index)settings.at("n_dip_bins") + 1,1);
	step = 0;
	for (auto bin:bins_.rowwise()) {
		bin = Matrix<double,1,1>(step);
		step += M_PI_2/settings.at("n_dip_bins");
	}
	for (Index az_i = 0; az_i < azimuths_.size(); ++az_i) {
		if (verbose_) std::cout << "Calculated az" << std::to_string(az_i) << "\n";
		TriangleContainer evaluation;
		std::copy(dir_triangle.begin(), dir_triangle.end(), std::back_inserter(evaluation));

		// Calculate apparent dip for each analysis direction
		std::for_each(evaluation.begin(), evaluation.end(),
			[&az_i, &cartesian_az](Triangle& triangle) {
			triangle.set_apparent_dip(cartesian_az(az_i, 0), cartesian_az(az_i, 1));
		});

		// Filter negative apparent dip
		auto remove_it = std::remove_if(evaluation.begin(), evaluation.end(),
			[&](const Triangle& triangle) {
			return triangle.apparent_dip_angle < 0;
		});
		evaluation.erase(remove_it, evaluation.end());
		std::sort(evaluation.begin(), evaluation.end(),
			[](const Triangle& a, const Triangle& b) {
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
						[](double a, const Triangle& b) {
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
					[](double a, const Triangle& b) {
					return a + b.area;
				});

				for (int bin_i = 0; bin_i <= bin_number; ++bin_i) dip_bins_data_(az_i, bin_i) += a_star;
			}
			n_facing_triangles_(az_i) = (double)evaluation.size();
			if (n_facing_triangles_(az_i) > 0) {
				// Divide all areas by the total area
				dip_bins_data_.row(az_i) /= this->total_area;

				// Get max theta !!(MUST BE SORTED ASCENDING)
				theta_max_(az_i) = evaluation.back().apparent_dip_angle * 180. / M_PI;

				A_0_parameter_(az_i) = dip_bins_data_(az_i, 0);

				// Fit power curve
				C_fitting_(az_i) = C_param_Newton_opt_backtracking_Fit(az_i);

				// Calculate chi-sq GOF
				Eigen::VectorXd observations = (dip_bins_data_.row(az_i)).transpose();
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
    this->parameters.insert({"az",azimuths_});
    this->parameters.insert({"n_tri", n_facing_triangles_});
	this->parameters.insert({"c",C_fitting_});
    this->parameters.insert({"theta_max",theta_max_});
    this->parameters.insert({"a0",A_0_parameter_});
    this->parameters.insert({"thetamax_cp1",(theta_max_ / (C_fitting_ + 1))});
    this->parameters.insert({"gof", chi_gof_});
	if (!file_path.empty()) save_file(file_path);
    points.resize(0,0);
    triangles.resize(0,0);
    normals.resize(0,0);
    triangle_mask.clear();
    areas.clear();
	
}

bool DirectionalRoughness::save_file(std::string path) 
{
	return create_file(path,this->points,this->triangles,this->normals);}

std::vector<std::string> DirectionalRoughness::result_keys()
{
    using namespace std;
    vector<string> keys(parameters.size());
    std::transform(parameters.begin(),parameters.end(),keys.begin(),
    [](const auto& param) { return param.first;});
    return keys;
}

double gamma_dr(double Z)
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
	p_value /= gamma_dr(K);
	return 1 - p_value;
}

double DirectionalRoughness::calculateGOFPValue(Eigen::VectorXd& observed, double C, double theta_max, double a_0)
{
	using namespace Eigen;
	// http://www.stat.yale.edu/Courses/1997-98/101/chigf.htm
	// https://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
	ArrayXd expected(observed.rows());
	for (Index i = 0; i < bins_.rows(); ++i)
		if (bins_(i)+1 < theta_max)
			expected(i) = a_0 * pow((1 - bins_(i) / theta_max), C);
	double chi_sq = ((observed.array() - expected)*(observed.array() - expected) / expected).sum();

	return chisqr((unsigned int)bins_.rows() - 1, chi_sq);
}


double estimate_dc(double c, Eigen::ArrayXd& x, Eigen::ArrayXd& y, double regularization=10e-7, double reduction=1) {
	using namespace Eigen;
	ArrayXd curve = pow(x,c);
	ArrayXd logx = log(x);
	double top = (2.*curve*logx*(curve-y)).sum();
	double bot = (2.*curve*square(logx)*(2.*curve-y)).sum();
	return -reduction * top/(bot + regularization);
}

inline double f(Eigen::ArrayXd& x, Eigen::ArrayXd& y, double c)
{
	using namespace Eigen;
	
	return (square(pow(x,c)-y)).sum();
}

inline double dfdc(Eigen::ArrayXd& x, Eigen::ArrayXd& y, double c)
{
	using namespace Eigen;
	return (2.*pow(x,c)*log(x)*(pow(x,c)-y)).sum();
}

double backtrack_search(Eigen::ArrayXd& x, Eigen::ArrayXd& y,double c, double dc, double reduction, double alpha, double beta)
{
	using namespace Eigen;
	double f_left = f(x,y,c+reduction*dc);
	double f_right = f(x,y,c) + alpha * reduction * dfdc(x,y,c)*dc;

	return (f_left > f_right)? beta * reduction : reduction;
}

double DirectionalRoughness::C_param_Newton_opt_backtracking_Fit(Eigen::Index az_i) {
	// https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/14-newton.pdf
	// https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
	using namespace Eigen;
	ArrayXd deg_bins = bins_.col(0) * 180 / M_PI;
	size_t n_valid_values = 0;
	for (double& deg : deg_bins) {
		if (deg > theta_max_(az_i)) {
			n_valid_values = &deg - &deg_bins[0];
			deg_bins.conservativeResize(n_valid_values);
			break;
		}
	}
	ArrayXd fit_data = dip_bins_data_.row(az_i).transpose();
	
	fit_data.conservativeResize(n_valid_values);
	ArrayXd y = fit_data / A_0_parameter_(az_i);
	ArrayXd x = 1 - deg_bins / theta_max_(az_i);

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