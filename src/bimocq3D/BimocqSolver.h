#ifndef IVOCK_BIMOCQSOLVER_H
#define IVOCK_BIMOCQSOLVER_H
#include "../include/array.h"
#include "tbb/tbb.h"
#include "../include/fluid_buffer3D.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstdio>
#include <string>
#include "../include/vec.h"
#include "../utils/pcg_solver.h"
#include "../include/array3.h"
#include "../utils/GeometricLevelGen.h"
#include "GPU_Advection.h"
#include <chrono>
#include "../utils/color_macro.h"
#include "Mapping.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/volumeMeshTools.h"

using namespace std;

enum Scheme {BIMOCQ, SEMILAG, MACCORMACK, MAC_REFLECTION};

class Emitter{
public:
    Emitter() : emitFrame(0), emit_density(0.f), emit_temperature(0.f), e_pos(Vec3f(0.f)), e_sdf(nullptr),
                vel_func([](float frame)->Vec3f{return Vec3f(0.f);}),
                emit_velocity([](Vec3f pos)->Vec3f{return Vec3f(0.f);}) {}
    Emitter(int frame, float density, float temperature, Vec3f position, openvdb::FloatGrid::Ptr sdf,
            std::function<Vec3f(float framenum)> func,
            std::function<Vec3f(Vec3f pos)> emit_velfunc)
        : emitFrame(frame), emit_density(density), emit_temperature(temperature), e_pos(position), e_sdf(sdf), vel_func(func), emit_velocity(emit_velfunc) {}
    ~Emitter() = default;

    int emitFrame;
    float emit_density;
    float emit_temperature;
    Vec3f e_pos;
    openvdb::FloatGrid::Ptr e_sdf;
    std::function<Vec3f(float framenum)> vel_func;
    std::function<Vec3f(Vec3f pos)> emit_velocity;

    // update levelset position
    void update(float framenum, float voxel_size, float dt)
    {
        e_pos += vel_func(framenum)*dt;
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::Vec3f(voxel_size));
        transMat.setTranslation(openvdb::Vec3f(e_pos[0], e_pos[1], e_pos[2]));
        e_sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

class Boundary{
public:
    Boundary(){};
    Boundary(Vec3f position, openvdb::FloatGrid::Ptr sdf, std::function<Vec3f(float framenum)> func): b_pos(position), b_sdf(sdf), vel_func(func) {}
    ~Boundary() = default;

    Vec3f b_pos;
    openvdb::FloatGrid::Ptr b_sdf;
    std::function<Vec3f(float framenum)> vel_func;

    // update levelset position
    void update(float framenum, float voxel_size, float dt)
    {
        b_pos += vel_func(framenum)*dt;
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::Vec3f(voxel_size));
        transMat.setTranslation(openvdb::Vec3f(b_pos[0], b_pos[1], b_pos[2]));
        b_sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

class BimocqSolver {
public:
    BimocqSolver() = default;
    BimocqSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, Scheme myscheme, gpuMapper *mymapper);
    ~BimocqSolver() = default;

    void advance(int framenum, float dt);
    void advanceBimocq(int framenum, float dt);
    void advanceSemilag(int framenum, float dt);
    void advanceMacCormack(int framenum, float dt);
    void advanceReflection(int framenum, float dt);
    void blendBoundary(buffer3Df &field, const buffer3Df &blend_field);
    void velocityReinitialize();
    void scalarReinitialize();
    void addBuoyancy(float dt);
    void emitSmoke(int framenum, float dt);
    void setSmoke(float drop, float raise, const std::vector<Emitter> &emitters);
    void outputResult(uint frame, string filepath);
    void setBoundary(const std::vector<Boundary> &boundaries);
    void updateBoundary(int framenum, float dt);
    void projection();
    void semilagAdvect(float cfldt, float dt);
    void clearBoundary(buffer3Df field);
    float getCFL();
    void clampExtrema(float dt, buffer3Df & f_n, buffer3Df & f_np1);
    void diffuse_field(double dt, double nu, buffer3Df & field);

    // smoke parameter
    float _alpha;
    float _beta;

    // AMGPCG solver data
    SparseMatrixd matrix;
    std::vector<double> rhs;
    std::vector<double> pressure;
    buffer3Dc _b_desc;

    // simulation data
    uint _nx, _ny, _nz;
    float max_v;
    float _h;
    float viscosity;
    buffer3Df _un, _vn, _wn;
    buffer3Df _uinit, _vinit, _winit;
    buffer3Df _uprev, _vprev, _wprev;
    buffer3Df _utemp, _vtemp, _wtemp;
    buffer3Df _duproj, _dvproj, _dwproj;
    buffer3Df _duextern, _dvextern, _dwextern;
    buffer3Df _rho, _rhotemp, _rhoinit, _rhoprev, _drhoextern;
    buffer3Df _T, _Ttemp, _Tinit, _Tprev, _dTextern;

    buffer3Df _usolid, _vsolid, _wsolid;
    Array3c u_valid, v_valid, w_valid;
    // initialize advector
    MapperBase VelocityAdvector;
    MapperBase ScalarAdvector;
    gpuMapper *gpuSolver;
    int vel_lastReinit = 0;
    int scalar_lastReinit = 0;
    Scheme sim_scheme;

    std::vector<Emitter> sim_emitter;
    std::vector<Boundary> sim_boundary;
};


#endif //IVOCK_BIMOCQSOLVER_H
