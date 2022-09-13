#ifndef _DIRECTIONAL_H
#define _DIRECTIONAL_H
#define _USE_MATH_DEFINES

#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>

struct DirectionalSetting : public std::unordered_map<std::string,double>
{
    void set(std::string setting, double value) {if (this->find(setting) != this->end()) this->at(setting) = value;}
    double get(std::string setting) {if (this->find(setting) != this->end()) return this->at(setting); else return std::nan("");}
};

class Directional 
{
public:
    Directional(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles);
    Directional(Eigen::MatrixX3d points, Eigen::MatrixX3i triangles, Eigen::ArrayXi selected_triangles);
    static DirectionalSetting Setting() {
        return DirectionalSetting({
			{"n_az",72.},
			{"az_offset",0.},
			{"min_triangles",200}
		});
    }
    virtual void evaluate(DirectionalSetting setting, bool verbose, std::string file) = 0;
    Eigen::ArrayXXd operator[](std::string key) {return parameters[key];}
    Eigen::MatrixX3d get_points() { return points; }
    Eigen::MatrixX3d get_normals() { return normals; }
    Eigen::Vector3d get_min_bounds() {return min_bounds;}
	Eigen::Vector3d get_max_bounds() {return max_bounds;}
	Eigen::Vector3d get_centroid() {return centroid;}
	std::vector<double> get_size() {return size_;}
	double get_area() {return total_area;}

	Eigen::Vector3d get_final_orientation() { return final_orientation; }
    std::vector<std::string> result_keys();

protected:
    Eigen::MatrixX3d points;
    Eigen::MatrixX3i triangles;
    Eigen::MatrixX3d normals;
    std::vector<uint8_t> triangle_mask;

    std::vector<double> areas;
    double total_area;

    bool save_file(std::string file_path);

    DirectionalSetting settings_;
    std::unordered_map<std::string,Eigen::ArrayXXd> parameters;

    void alignBestFit();

    Eigen::Vector3d plane_fit(const Eigen::MatrixX3d& xyz);
    Eigen::Vector3d plane_normal(const Eigen::MatrixX3d& xyz);
	Eigen::Vector3d initial_orientation;
	Eigen::Vector3d final_orientation;

	Eigen::Vector3d min_bounds;
	Eigen::Vector3d max_bounds;
	Eigen::Vector3d centroid;
	std::vector<double> size_;

	bool aligned;

    Eigen::ArrayXXd azimuths_;
    Eigen::ArrayXXd n_facing_triangles_;
};
#endif //_DIRECTIONAL_H