#ifndef _DIRECTIONALROUGHNESS_H
#define _DIRECTIONALROUGHNESS_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>

#include <Eigen/Core>

struct DirectionalRoughness_settings : public std::unordered_map<std::string,double>
{
	DirectionalRoughness_settings() {
        this->emplace("n_az",72.0);
        this->emplace("az_offset",0);
        this->emplace("n_dip_bins",90);
        this->emplace("fit_initialguess",1);
        this->emplace("fit_precision",6);
		this->emplace("fit_regularization",10e-10);
		this->emplace("fit_alpha",0.01);
		this->emplace("fit_beta",0.5);
		this->emplace("min_triangles",200);
		
	}
    void set(std::string setting, double value) 
    {if (this->find(setting) != this->end()) this->at(setting) = value;}
    double get(std::string setting) { if (this->find(setting) != this->end()) return this->at(setting); else return -999;}
};

class DirectionalRoughness
{
public:
	DirectionalRoughness(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles);
	DirectionalRoughness(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles, Eigen::ArrayXi selected_triangles);
	void evaluate(DirectionalRoughness_settings settings = DirectionalRoughness_settings(),bool verbose=false, std::string file=std::string());
    Eigen::ArrayXd operator[](std::string key) {return parameters[key];}
    Eigen::MatrixX3d get_points() {return points; }
    Eigen::MatrixX3d get_normals() { return normals; }
	
	Eigen::Vector3d get_min_bounds() {return min_bounds;}
	Eigen::Vector3d get_max_bounds() {return max_bounds;}
	Eigen::Vector3d get_centroid() {return centroid;}
	std::vector<double> get_size() {return size_;}
	double get_area() {return total_area;}

	Eigen::Vector3d get_final_orientation() { return final_orientation; }
    std::vector<std::string> result_keys();

private:
    Eigen::MatrixX3d points;
    Eigen::MatrixX3i triangles;
    Eigen::MatrixX3d normals;
	std::vector<uint8_t> triangle_mask;

    std::vector<double> areas;
    double total_area;

	bool save_file(std::string file_path);

	DirectionalRoughness_settings settings_;
    std::unordered_map<std::string,Eigen::ArrayXXd> parameters;

    void alignBestFit();

	double C_param_GGFit_3D(Eigen::Index az_i);
	double C_param_Newton_opt_backtracking_Fit(Eigen::Index az_i);
	double calculateGOFPValue(Eigen::VectorXd& observed, double C, double theta_max, double a_0);
    
    Eigen::Vector3d plane_fit(const Eigen::MatrixX3d& xyz);
    Eigen::Vector3d plane_normal(const Eigen::MatrixX3d& xyz);
	Eigen::Vector3d initial_orientation;
	Eigen::Vector3d final_orientation;

	Eigen::Vector3d min_bounds;
	Eigen::Vector3d max_bounds;
	Eigen::Vector3d centroid;
	std::vector<double> size_;

	bool aligned;

	// Collected parameters
	Eigen::ArrayXXd azimuths_;
	Eigen::ArrayXXd C_fitting_;
	Eigen::ArrayXXd A_0_parameter_;
	Eigen::ArrayXXd theta_max_;
	Eigen::ArrayXXd n_facing_triangles_;
	Eigen::ArrayXXd dip_bins_data_;
	Eigen::ArrayXXd bins_;
	Eigen::ArrayXXd chi_gof_;
};
#endif //_DIRECTIONALROUGHNESS_H