#ifndef _DIRECTIONALUTIL_H
#define _DIRECTIONALUTIL_H
#define _USE_MATH_DEFINES

#include <math.h>
#include <unordered_map>
#include <tuple>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>

struct Triangle
{
	unsigned int index;
	double normal_x, normal_y, normal_z;//, horizontal_radius;
	double normal_angle;
	long double area;
	double apparent_dip_angle;
	Triangle() {}
	/*  Matrix form V(vertex, dimension)
	*        x  y  z
	*   V0  [       ]
	*   V1  [       ]
	*   V2  [       ]
	*
	*/

	Triangle(int index, Eigen::Vector3d normal, double area) :
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
	inline void set_normal(Eigen::RowVector3d normal) {
		this->normal_x = normal(0);
		this->normal_y = normal(1);
		this->normal_z = normal(2);
		normal_angle = std::atan2(normal_y,normal_x);
		if (normal_angle < 0) normal_angle += 2*M_PI;
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

Eigen::Vector3d plane_fit(const Eigen::MatrixX3d& xyz) {
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

Eigen::Vector3d plane_normal(const Eigen::MatrixX3d& xyz) {
    using namespace Eigen;
    Vector3d fit = plane_fit(xyz);
    return fit.normalized();
}

struct BestFitResult {
    Eigen::Vector3d initial_orientation;
    Eigen::Vector3d final_orientation;
    Eigen::Vector3d min_bounds;
    Eigen::Vector3d max_bounds;
    Eigen::Vector3d centroid;
};

BestFitResult align(Eigen::MatrixX3d& points, Eigen::MatrixX3i& triangles) {
    using namespace Eigen;
    Vector3d centroid = points.colwise().mean();

    Vector3d initial_orientation = plane_normal(points);
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
        points = (rotation.matrix() * points.transpose()).transpose();
        
        current_orientation = plane_normal(points);
    }
    BestFitResult result;
    result.final_orientation = current_orientation;
    result.initial_orientation = initial_orientation;
    result.centroid = points.colwise().mean();
    result.min_bounds = points.colwise().minCoeff();
    result.max_bounds = points.colwise().maxCoeff();
    return result;
}

void calculateNormals(Eigen::MatrixX3d& points, Eigen::MatrixX3i& triangles, Eigen::MatrixX3d& normals) {
    using namespace Eigen;
	normals.resize(triangles.rows(),NoChange);
	std::transform(
		triangles.rowwise().cbegin(),triangles.rowwise().cend(),
		normals.rowwise().begin(),
		[&](const auto& row){
			Vector3d V1V2 = points.row(row(1)) - points.row(row(0));
			Vector3d V1V3 = points.row(row(2)) - points.row(row(0));
			return V1V2.cross(V1V3).normalized();
		}
	);
}

double calculateAreas(std::vector<double>& areas, Eigen::MatrixX3d& points, Eigen::MatrixX3i& triangles) {
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
    return std::accumulate(areas.begin(), areas.end(), 0.0);
}

std::vector<double> get_area_vector(
	std::vector<Triangle> evaluation, 
	std::function<double(Triangle)> area_op)
{
	std::vector<double> area_vec;
	std::transform(evaluation.begin(),evaluation.end(),
	std::back_inserter(area_vec),area_op);
	return area_vec;
}

bool create_file(std::string path, Eigen::MatrixX3d& points, Eigen::MatrixX3i& triangles, Eigen::MatrixX3d& normals) {
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

Eigen::MatrixX2d pol2cart(Eigen::ArrayXd azimuth) {
    using namespace Eigen;
	// Polar to cartesian transformation by vector
	ArrayXd Tx = azimuth.cos();
	ArrayXd Ty = azimuth.sin();

	MatrixX2d T(azimuth.size(),2);
	T << Tx,Ty;
	return T;
}

#endif //_DIRECTIONALUTIL_H