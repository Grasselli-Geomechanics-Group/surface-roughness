#include "Directional.h"

#include <math.h>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <vector>

#include <Eigen/Core>

#include "DirectionalUtil.h"

Directional::Directional(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles) :
	aligned(false),
	points(points),
	triangles(triangles),
    settings_(Directional::Setting())
{
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
}


Directional::Directional(
    Eigen::MatrixX3d points, 
    Eigen::MatrixX3i triangles,
    Eigen::ArrayXi selected_triangles):
	aligned(false),
    settings_(Directional::Setting())
{
	select_triangles(this->points,points,this->triangles,triangles,selected_triangles);
    this->alignBestFit();
    calculateNormals(this->points,this->triangles,this->normals);
}

void Directional::alignBestFit()
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

bool Directional::save_file(std::string path) 
{
	return create_file(path,this->points,this->triangles,this->normals);
}

std::vector<std::string> Directional::result_keys()
{
    using namespace std;
    vector<string> keys(parameters.size());
    std::transform(parameters.begin(),parameters.end(),keys.begin(),
    [](const auto& param) { return param.first;});
    return keys;
}

