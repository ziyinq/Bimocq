#ifndef BIMOCQ_VOLUMETOOLS_H
#define BIMOCQ_VOLUMETOOLS_H

#include "../include/vec.h"
#include <openvdb/math/Math.h>
#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>
#include "openvdb/tools/ParticlesToLevelSet.h"
#include "openvdb/tools/LevelSetFilter.h"
#include "openvdb/tools/VolumeToMesh.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/Interpolation.h"
#include "openvdb/tools/GridOperators.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/LevelSetPlatonic.h"
#include <openvdb/util/Util.h>

inline void writeObj(const std::string &objname, const std::vector<openvdb::Vec3f> &verts, const std::vector <openvdb::Vec4I> & faces)
{
    ofstream outfile(objname);

    //write vertices
    for (unsigned int i = 0; i < verts.size(); ++i)
        outfile << "v" << " " << verts[i][0] << " " << verts[i][1] << " " << verts[i][2] << std::endl;
    //write triangle faces
    for (unsigned int i = 0; i < faces.size(); ++i)
        outfile << "f" << " " << faces[i][3] + 1 << " " << faces[i][2] + 1 << " " << faces[i][1] + 1 << " " << faces[i][0] + 1 << std::endl;
    outfile.close();
}

inline void writeVDB(uint frame, std::string filepath, float voxel_size, const buffer3Df &field)
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    int total = 0;
    for(uint k=0; k<field._nz; k++) for(uint j=0; j<field._ny; j++) for(uint i=0; i<field._nx; i++)
            {
                if(field(i,j,k) > 1e-4)
                {
                    openvdb::math::Coord xyz(i,j,k);
                    accessor.setValue(xyz, field(i,j,k));
                    total += 1;
                }
            }
    char file_name[256];
    std::cout << "[ Valid vdb voxel: " << total << " ] "<< std::endl << std::endl;
    sprintf(file_name,"%s/density_render_%04d.vdb", filepath.c_str(), frame);
    std::string vdbname(file_name);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size));
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    grid->setName("density");
    openvdb::io::File file(vdbname);
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

inline openvdb::FloatGrid::Ptr readMeshToLevelset(const std::string &filename, float h)
{
    std::vector<Vec3f> vertList;
    std::vector<Vec3ui> faceList;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open. Terminating.\n";
        exit(-1);
    }

    int ignored_lines = 0;
    std::string line;

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.substr(0, 1) == std::string("v")) {
            std::stringstream data(line);
            char c;
            Vec3f point;
            data >> c >> point[0] >> point[1] >> point[2];
            vertList.push_back(point);
        }
        else if (line.substr(0, 1) == std::string("f")) {
            std::stringstream data(line);
            char c;
            int v0, v1, v2;
            data >> c >> v0 >> v1 >> v2;
            faceList.push_back(Vec3ui(v0 - 1, v1 - 1, v2 - 1));
        }
        else {
            ++ignored_lines;
        }
    }
    infile.close();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    points.resize(vertList.size());
    triangles.resize(faceList.size());
    tbb::parallel_for(0, (int)vertList.size(), 1, [&](int p)
    {
        points[p] = openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
    });
    tbb::parallel_for(0, (int)faceList.size(), 1, [&](int p)
    {
        triangles[p] = openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
    });
    openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*openvdb::math::Transform::createLinearTransform(h), points, triangles,3.0);
    return grid;
}
#endif //BIMOCQ_VOLUMETOOLS_H
