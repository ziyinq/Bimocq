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
    // allowed maximum CFL
    float CFL;
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
    bool PURE_NEUMANN;
    Scheme sim_scheme;
    int sim_name = 0;
    int example = 0;
    if (argc != 3)
    {
        std::cout << "Please specify correct parameters!" << std::endl;
        exit(0);
    }
    sim_name = atoi(argv[1]);
    example = atoi(argv[2]);

    switch(example)
    {
        // 2D Taylor-vortex example
        case 0:
        {
            std::cout << GREEN << "Start running 2D Taylor Vortex example!!!" << RESET << std::endl;
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
            PURE_NEUMANN = false;
            sim_scheme = static_cast<Scheme>(sim_name);
            std::string filepath = "../Out/2D_Taylor_vortex/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
            BimocqSolver2D smokeSimulator(nx, ny, L, blend_coeff, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitVelocity(vorticity_distance);
            smokeSimulator.sampleParticlesFromGrid();
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i);
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(filepath, filename, i);
            }
        }
        break;
        // 2D Vortex-leapfrogging example
        case 1:
        {
            std::cout << GREEN << "Start running 2D Vortex Leapfrogging example!!!" << RESET << std::endl;
            nx = 256;
            ny = 256;
            dt = 0.025;
            N = 4;
            L = 2.f*M_PI;
            total_frame = 2000;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = false;
            sim_scheme = static_cast<Scheme>(sim_name);
            std::string filepath = "../Out/2D_Leapfrog/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) +"_";
            BimocqSolver2D smokeSimulator(nx, ny, L, blend_coeff, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitLeapFrog(1.5, 3.0, M_PI-1.6, 0.3);
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.sampleParticlesFromGrid();
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i);
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(filepath, filename, i);
                smokeSimulator.outputDensity(filepath, "density", i, false);
            }
        }
        break;
        // 2D Rayleigh-Taylor example
        case 2:
        {
            std::cout << GREEN << "Start running 2D Rayleigh Taylor example!!!" << RESET << std::endl;
            nx = 256;
            ny = 1280;
            dt = 0.01;
            N = 4;
            L = 0.2;
            total_frame = 1000;
            smoke_rise = 0.2f;
            smoke_drop = 0.05f;
            blend_coeff = 1.f;
            PURE_NEUMANN = true;
            sim_scheme = static_cast<Scheme>(atoi(argv[1]));
            std::string filepath = "../Out/2D_RayleighTaylor/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_dt_" + std::to_string(dt).substr(0,5) +"_";
            BimocqSolver2D smokeSimulator(nx, ny, L, blend_coeff, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitReyleighTaylor(0.5f * L * ny / nx);
            smokeSimulator.sampleParticlesFromGrid();
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i);
                smokeSimulator.outputDensity(filepath, "density", i, true);
            }
        }
        break;
        // 2D Zalesak's disk example
        case 3:
        {
            std::cout << GREEN << "Start running 2D Zalesak's Disk example!!!" << RESET << std::endl;
            nx = 200;
            ny = 200;
            CFL = 0.75;
            N = 4;
            L = 1;
            total_frame = 315;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = true;
            sim_scheme = static_cast<Scheme>(atoi(argv[1]));
            if (sim_scheme == FLIP || sim_scheme == APIC || sim_scheme == POLYPIC)
            {
                std::cout << "Simulation scheme for levelset is not supported!" << std::endl;
                exit(0);
            }
            std::string filepath = "../Out/2D_Zalesak/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_";
            BimocqSolver2D smokeSimulator(nx, ny, L, blend_coeff, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.advect_levelset = true;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitZalesak();
            smokeSimulator.outputLevelset(filepath, 0);
            for (int i = 1; i < total_frame; i++)
            {
                float frame_dt = 2;
                float T = 0.f;
                float substep = CFL*smokeSimulator.h/smokeSimulator.maxVel();
                std::cout << substep << std::endl;
                while (T < frame_dt)
                {
                    if (T + substep > frame_dt) substep = frame_dt - T;
                    std::cout << "current CFL is: " << substep*smokeSimulator.maxVel()/smokeSimulator.h << std::endl;
                    smokeSimulator.advance(substep, i);
                    T += substep;
                }
                smokeSimulator.outputLevelset(filepath, i);
            }
        }
        break;
        // 2D Vortex in a Box example
        case 4:
        {
            std::cout << GREEN << "Start running 2D Vortex in a Box example!!!" << RESET << std::endl;
            nx = 512;
            ny = 512;
            CFL = 0.5;
            N = 4;
            L = 1;
            total_frame = 500;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = true;
            sim_scheme = static_cast<Scheme>(atoi(argv[1]));
            if (sim_scheme == FLIP || sim_scheme == APIC || sim_scheme == POLYPIC)
            {
                std::cout << "Simulation scheme for levelset is not supported!" << std::endl;
                exit(0);
            }
            std::string filepath = "../Out/2D_VortexBox/" + enumToString(sim_scheme) + "/";
            std::string filename = enumToString(sim_scheme) + "_";
            BimocqSolver2D smokeSimulator(nx, ny, L, blend_coeff, N, PURE_NEUMANN, sim_scheme);
            smokeSimulator.advect_levelset = true;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitVortexBox();
            smokeSimulator.outputLevelset(filepath, 0);
            for (int i = 1; i < total_frame; i++)
            {
                float frame_dt = 0.01;
                float T = 0.f;
                float substep = CFL*smokeSimulator.h/smokeSimulator.maxVel();
                while (T < frame_dt)
                {
                    if (T + substep > frame_dt) substep = frame_dt - T;
                    std::cout << "current CFL is: " << substep*smokeSimulator.maxVel()/smokeSimulator.h << std::endl;
                    std::cout << "current dt is: " << substep << std::endl;
                    smokeSimulator.advance(substep, i);
                    T += substep;
                }
                smokeSimulator.outputLevelset(filepath, i);
            }
        }
        break;
    }

    return 0;
}