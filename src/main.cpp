#include <cmath>
#include "array.h"
#include <iostream>
#include "GPU_Advection.h"
#include "BimocqSolver.h"
#include <boost/filesystem.hpp>

int main(int argc, char** argv) {
    uint ni;
    uint nj;
    uint nk;
    uint total_frame;
    float L;
    float h;
    float dt;
    float mapping_blend_coeff;
    float viscosity;
    float half_width;
    float smoke_rise;
    float smoke_drop;
    Scheme sim_scheme;
    string filepath = "../Out";
    boost::filesystem::create_directories(filepath);

    std::vector<Emitter> emitter_list;
    std::vector<Boundary> boundary_list;

    // 3D vortex collision example setup
    if (1)
    {
        // simulation resolution
	    ni = 100;
	    nj = 200;
	    nk = 200;
	    total_frame = 300;
	    // length in x direction
	    L = 0.2f;
	    // grid size for simulation
	    h = L / ni;
	    // time step
	    dt = 0.08f;
	    // smoke properties
	    smoke_rise = 0.f;
	    smoke_drop = 0.f;
	    viscosity = 1.0*1e-6;
	    // blend coefficient that will blend 1-level mapping result with 2-level mapping result
        // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
        mapping_blend_coeff = 1.f;
        // levelset half width, used when blending semi-lagrangian result near the boundary
        half_width = 3.f;
        // simulation scheme, semi-lagrangian, MacCormack, Reflection and BIMOCQ are implemented
        sim_scheme = BIMOCQ;
        auto vel_func_a = [](Vec3f pos)
        {
            Vec3f center(0.04f, 0.2f, 0.2f);
            Vec2f dir = Vec2f(pos[1] - center[1], pos[2] - center[2]);
            dir = normalized(dir);
            float theta = acos(dot(dir, Vec2f(1.f, 0.f)));
            float vel_x = 0.06f*(1.0f + 0.01f*cos(8.f*theta));
            float vel_y = 0.f;
            float vel_z = 0.f;
            return Vec3f(vel_x, vel_y, vel_z);
        };
        auto vel_func_b = [](Vec3f pos)
        {
            Vec3f center(0.16f, 0.201f, 0.2f);
            Vec2f dir = Vec2f(pos[1] - center[1], pos[2] - center[2]);
            dir = normalized(dir);
            float theta = acos(dot(dir, Vec2f(1.f, 0.f)));
            float vel_x = -0.06f*(1.0f + 0.01f*cos(8.f*theta));
            float vel_y = 0.f;
            float vel_z = 0.f;
            return Vec3f(vel_x, vel_y, vel_z);
        };
        openvdb::FloatGrid::Ptr sphere_sdf_a = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.015f, openvdb::Vec3f(0.f,0.f,0.f), h, half_width);
        openvdb::FloatGrid::Ptr sphere_sdf_b = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.015f, openvdb::Vec3f(0.f,0.f,0.f), h, half_width);
        Emitter e_sphere_a(10, 1.f, 50.f, Vec3f(0.04f, 0.2f, 0.2f), sphere_sdf_a, [](float frame)->Vec3f{return Vec3f(0.f, 0.f, 0.f);}, vel_func_a);
        Emitter e_sphere_b(10, 1.f, 50.f, Vec3f(0.16f, 0.201f, 0.2f), sphere_sdf_b, [](float frame)->Vec3f{return Vec3f(0.f, 0.f, 0.f);}, vel_func_b);
        emitter_list.push_back(e_sphere_a);
        emitter_list.push_back(e_sphere_b);
    }

	auto *myGPUmapper = new gpuMapper(ni, nj, nk, h);
	BimocqSolver mysolver(ni, nj, nk, L, viscosity, mapping_blend_coeff, sim_scheme, myGPUmapper);
	mysolver.setSmoke(smoke_rise, smoke_drop, emitter_list);
    mysolver.setBoundary(boundary_list);
	for (uint i = 0; i < total_frame; i++)
	{
        cout << "Frame " << i << " Starts !!!" << std::endl;
	    mysolver.updateBoundary(i, dt);
		mysolver.advance(i, dt);
        mysolver.outputResult(i, filepath);
    }
	return 0;
}