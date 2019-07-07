#include <iostream>
#include <cstdlib>
#include "BimocqSolver2D.h"
#include <string>
#include "../utils/visualize.h"

int main(int argc, char** argv)
{
    // resolution
    int nx;
    int ny;
    // time step
    float dt;
    // particle per cell
    int N;
    // simulation domain length in x-direction
    float L;
    // two level mapping blend coefficient
    float blend_coeff;
    int total_frame;
    float vorticity_distance;
    // smoke property
    float smoke_rise;
    float smoke_drop;
    // use Neumann boundary or not
    bool neumann_boundary;
    Scheme sim_scheme;
    if (argc != 2)
    {
        std::cout << "Please specify correct parameters!" << std::endl;
        exit(0);
    }

    if (0)
    {
        nx = 256;
        ny = 256;
        dt = 0.025;
        N = 4;
        L = 2.f*M_PI;
        total_frame = 300;
        vorticity_distance = 0.81;
        smoke_rise = 0.f;
        smoke_drop = 0.f;
        blend_coeff = 1.f;
        neumann_boundary = false;
        sim_scheme = static_cast<Scheme>(atoi(argv[1]));
        std::string filepath = "../Out/2D_Taylor_vortex/" + enumToString(sim_scheme) + "/";
        std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
        BimocqSolver2D smokeSimulator(nx, ny, dt, L, blend_coeff, N, neumann_boundary, sim_scheme);
        smokeSimulator.setSmoke(smoke_rise, smoke_drop);
        smokeSimulator.buildMultiGrid();
        smokeSimulator.setInitVelocity(vorticity_distance);
        smokeSimulator.initParticleVelocity();
        for (int i = 0; i < total_frame; i++)
        {
            smokeSimulator.advance(dt, i);
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(filepath, filename, i);
        }
    }
    else if (1)
    {
        nx = 256;
        ny = 256;
        dt = 0.025;
        N = 4;
        L = 2.f*M_PI;
        total_frame = 2000;
        smoke_rise = 0.f;
        smoke_drop = 0.f;
        blend_coeff = 1.f;
        neumann_boundary = false;
        sim_scheme = static_cast<Scheme>(atoi(argv[1]));
        std::string filepath = "../Out/2D_Leapfrog/" + enumToString(sim_scheme) + "/";
        std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
        BimocqSolver2D smokeSimulator(nx, ny, dt, L, blend_coeff, N, neumann_boundary, sim_scheme);
        smokeSimulator.setSmoke(smoke_rise, smoke_drop);
        smokeSimulator.buildMultiGrid();
        smokeSimulator.setInitLeapFrog(1.5, 3.0, M_PI-1.6, 0.3);
        smokeSimulator.applyVelocityBoundary();
        smokeSimulator.initParticleVelocity();
        for (int i = 0; i < total_frame; i++)
        {
            smokeSimulator.advance(dt, i);
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(filepath, filename, i);
            smokeSimulator.outputDensity(filepath, "density", i, false);
        }
    }
    else if (1)
    {
        nx = 256;
        ny = 1280;
        dt = 0.025;
        N = 4;
        L = 2.f*M_PI;
        total_frame = 1000;
        smoke_rise = 0.2f;
        smoke_drop = 0.05f;
        blend_coeff = 1.f;
        neumann_boundary = false;
        sim_scheme = static_cast<Scheme>(atoi(argv[1]));
        std::string filepath = "../Out/2D_Leapfrog/" + enumToString(sim_scheme) + "/";
        std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
        BimocqSolver2D smokeSimulator(nx, ny, dt, L, blend_coeff, N, neumann_boundary, sim_scheme);
        smokeSimulator.setSmoke(smoke_rise, smoke_drop);
        smokeSimulator.buildMultiGrid();
        smokeSimulator.setInitLeapFrog(1.5, 3.0, M_PI-1.6, 0.3);
        smokeSimulator.applyVelocityBoundary();
        smokeSimulator.initParticleVelocity();
        for (int i = 0; i < total_frame; i++)
        {
            smokeSimulator.advance(dt, i);
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(filepath, filename, i);
            smokeSimulator.outputDensity(filepath, "density", i, false);
        }
    }



    return 0;
}