#ifndef _TINBASEDROUGHNESS_H
#define _TINBASEDROUGHNESS_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>

#include <armadillo>

struct TIN_triangle
{
	unsigned int index;
	double normal_x, normal_y, normal_z;//, horizontal_radius;
	double normal_angle;
	long double area;
	double apparent_dip_angle;
	TIN_triangle() {}
	/*  Matrix form V(vertex, dimension)
	*        x  y  z
	*   V0  [       ]
	*   V1  [       ]
	*   V2  [       ]
	*
	*/

	TIN_triangle(int index, arma::vec normal, double area) :
		index(index), area(area),apparent_dip_angle(0)
	{
		normal_x = normal(0);
		normal_y = normal(1);
		normal_z = normal(2);
		// get r = sqrt(normal_x^2+normal_y^2)
		//horizontal_radius = std::sqrt(normal_x*normal_x + normal_y*normal_y);
		normal_angle = std::atan2(normal_y, normal_x);
		if (normal_angle < 0) normal_angle += 2 * M_PI;
		// normal slope w.r.t x-y plane = normal_z / sqrt(normal_x^2+normal_y^2)
		// dip_slope is perpendicular to normal slope and needs to be positive
		// therefore get the absolute of inverse  of normal_slope
		//true_dip_slope = std::abs(horizontal_radius/(normal(2)));
	}
	inline void set_normal(double normal_x, double normal_y,double normal_z) {
		this->normal_x = normal_x;
		this->normal_y = normal_y;
		this->normal_z = normal_z;
		normal_angle = std::atan2(normal_y, normal_x);
		if (normal_angle < 0) normal_angle += 2 * M_PI;
	}
	void set_apparent_dip(double shear_dir_x, double shear_dir_y)
	{
		// apparent dip slope calculation
		// Get shear direction for shear plane
		// Project triangle normal vector onto shear plane
		// Pr = normal - proj(normal) on shear direction plane
		// Apparent dip angle is preserved and is now the true dip angle of plane represented by the projected normal vector
		double Prx = normal_x - (shear_dir_y*shear_dir_y*normal_x - shear_dir_y * shear_dir_x*normal_y);
		double Pry = normal_y - (-shear_dir_x * shear_dir_y*normal_x + shear_dir_x * shear_dir_x*normal_y);
		double Prz = normal_z;
		// Renormalize projected normal vector
		double div = std::sqrt(Prx*Prx + Pry * Pry + Prz * Prz);
		Prx /= div;
		Pry /= div;

		// Get angle between projected normal vector and shear direction
		apparent_dip_angle =  std::acos(shear_dir_x*(Prx)+shear_dir_y * (Pry)) - M_PI_2;

	}

};

struct TINBasedRoughness_settings : public std::unordered_map<std::string,double>
{
	TINBasedRoughness_settings() {
        this->emplace("n_az",72.0);
        this->emplace("az_offset",0);
		this->emplace("min_triangles",200);
	}
    void set(std::string setting, double value) 
    {if (this->find(setting) != this->end()) this->at(setting) = value;}
    double get(std::string setting) { if (this->find(setting) != this->end()) return this->at(setting); else return -999;}
};

class TINBasedRoughness
{
public:
	TINBasedRoughness(const std::vector<double>& points, const std::vector<uint64_t>& triangles);
	TINBasedRoughness(const std::vector<double>& points, const std::vector<uint64_t>& triangles, const std::vector<uint64_t>& selected_triangles);
	void evaluate(TINBasedRoughness_settings settings = TINBasedRoughness_settings(),bool verbose=false, std::string file=std::string());
    std::vector<double> operator[](std::string key) {return parameters[key];}
    std::vector<double> get_points();
    std::vector<double> get_normals();

	std::vector<double> get_min_bounds() {return min_bounds;}
	std::vector<double> get_max_bounds() {return max_bounds;}
	std::vector<double> get_centroid() {return centroid;}
	std::vector<double> get_size() {return size_;}
	double get_area() {return total_area;}

	std::vector<double> get_final_orientation() { return final_orientation; }
    std::vector<std::string> result_keys();

private:
    arma::mat points;
    arma::Mat<arma::uword> triangles;
    arma::mat normals;
	std::vector<uint8_t> triangle_mask;

    std::vector<double> areas;
    double total_area;

	bool save_file(std::string file_path);
	
	TINBasedRoughness_settings settings_;
    std::unordered_map<std::string,std::vector<double>> parameters;

	
	arma::mat pol2cart(arma::vec azimuths);
    void alignBestFit();
    void calculateNormals();
    void calculateAreas();

    arma::vec plane_fit(const arma::mat& xyz);
    arma::vec plane_normal(const arma::mat& xyz);
	std::vector<double> initial_orientation;
	std::vector<double> final_orientation;

	std::vector<double> min_bounds;
	std::vector<double> max_bounds;
	std::vector<double> centroid;
	std::vector<double> size_;

	bool aligned;

	// Collected parameters
	arma::vec azimuths_;
	arma::vec delta_t_;
	arma::vec delta_star_t_;
	arma::uvec n_facing_triangles_;
};

#endif //_TINBASEDROUGHNESS_H