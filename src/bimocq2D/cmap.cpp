//
// Created by ziyin on 18-9-28.
//
#include <iostream>
#include "cmapFluid2D.h"
#include <string>
#include "../utils/visualize.h"

int main(int argc, char** argv)
{
    float dt = 0.06;
    float vortDistance = 0.8;
    int timesteps = 300;
    float L= 0.2f;
    int reseample_step = 30;
    int emitframe = 200;
    int type;
    int N;
    sscanf(argv[1],"%d",&type);
    sscanf(argv[2],"%f",&dt);
    sscanf(argv[3],"%f",&vortDistance);
    sscanf(argv[4],"%d",&timesteps);
    sscanf(argv[5],"%d",&N);

    unsigned char* data = 0;

    int nx = 256*N;
    int ny = nx*5;
    cmapFluid2D smokeSimulator;
    smokeSimulator.init(nx, ny, L);
    smokeSimulator.buildMultiGrid();
    smokeSimulator.setAlpha(0.2);
    smokeSimulator.setBeta(0.05);

    switch(type) {
        case 0:
            {
                cout<< "case 0 " << endl;
                std::string folder = "../Out/Semilag/";
                std::string filename = "Semilag_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
                smokeSimulator.setInitVelocity(vortDistance);
                smokeSimulator.applyVelocityBoundary();
                smokeSimulator.projection(1e-6,false);
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
                for (int i = 0; i <= timesteps; i++) {
                    std::cout << "Semi-Lagrangian Frame " << i << ": ";
                    for (int j = 0; j < 2; j++)
                    {
                        smokeSimulator.advance(dt, i, emitframe, NULL);
                    }
                    smokeSimulator.calculateCurl();
                    std::cout<<"Energy:"<<smokeSimulator.computeEnergy()<<std::endl;
                    //smokeSimulator.output("../Out/original/", "smoke_density_original", i, data);
                    smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
                }
            }
            break;


        case 1:
        {
            std::string folder = "../Out/APIC/";
            std::string filename = "APIC_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
            smokeSimulator.initCmap(nx,ny,4);
            smokeSimulator.setInitVelocity(vortDistance);
            cout<< "case 1 " << endl;
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.projection(1e-6, false);
            smokeSimulator.initParticleVelocity();
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            for (int i = 0; i <= timesteps; i++) {
                std::cout << "APIC Frame " << i << ": ";
                for (int j = 0; j < 2; j++)
                {
                    smokeSimulator.advanceAPIC(dt);
                }
                std::cout<<"Energy:"<<smokeSimulator.computeEnergy()<<std::endl;
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
            }
        }
            break;

        case 2:
        {
            std::string folder = "../Out/PolyPIC/";
            std::string filename = "PolyPIC_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
            smokeSimulator.initCmap(nx,ny,4);
            smokeSimulator.setInitVelocity(vortDistance);
            cout<< "case 2 " << endl;
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.projection(1e1-6, false);
            smokeSimulator.initParticleVelocity();
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            for (int i = 0; i <= timesteps; i++) {
                std::cout << "PolyPIC Frame " << i << ": ";
                for (int j = 0; j < 2; j++)
                {
                    smokeSimulator.advancePolyPIC(dt);
                }
                std::cout<<"Energy:"<<smokeSimulator.computeEnergy()<<std::endl;
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
            }
        }
            break;

        case 3:
        {
            std::string folder = "../Out/reflection/";
            std::string filename = "reflection_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
//            smokeSimulator.setInitVelocity(vortDistance);
//            smokeSimulator.setInitLeapFrog(1.5,3.0);
//            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            smokeSimulator.setInitDensity(0.5f*L*5.f, smokeSimulator.rho, smokeSimulator.temperature);
            smokeSimulator.initMaccormack();
//            smokeSimulator.buildMultiGrid_reflect();
            cout<< "case 4 " << endl;
//            smokeSimulator.applyVelocityBoundary();
//            smokeSimulator.projection(1e-6,false);
//            smokeSimulator.calculateCurl();
//            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            for (int i = 0; i <= timesteps; i++) {
                std::cout << "Reflection Frame " << i << ": ";
                smokeSimulator.advanceReflection(dt, i, emitframe, NULL);
                smokeSimulator.output("../Out/reflection/", "smoke_density_original", i, NULL);
            }
        }
            break;

        case 4:
        {
            std::string folder = "../Out/BFECC/";
            std::string filename = "BFECC_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
            smokeSimulator.setInitVelocity(vortDistance);
            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            smokeSimulator.initMaccormack();
            cout<< "case 5 " << endl;
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.projection(1e-6,false);
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            for (int i = 0; i <= timesteps; i++) {
                std::cout << "BFECC Frame " << i << ": ";
                for (int j = 0; j < 2; j++)
                {
                    smokeSimulator.advanceBFECC(dt, i, emitframe, NULL);
                }
                std::cout<<"Energy:"<<smokeSimulator.computeEnergy()<<std::endl;
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
            }
        }
            break;

        case 5:
        {
            std::string folder = "../Out/MacCormack/";
            std::string filename = "MacCormack_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
//            smokeSimulator.setInitVelocity(vortDistance);
            smokeSimulator.setInitDensity(0.5f*L*5.f, smokeSimulator.rho, smokeSimulator.temperature);
            smokeSimulator.initMaccormack();    
            cout<< "case 6 " << endl;
//            smokeSimulator.applyVelocityBoundary();
//            smokeSimulator.projection(1e-6,false);
//            smokeSimulator.calculateCurl();
//            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            for (int i = 0; i <= timesteps; i++) {
                std::cout << "MacCormack Frame " << i << ": ";
                smokeSimulator.advanceMaccormack(dt, i, emitframe, NULL);
//                std::cout<<"Energy:"<<smokeSimulator.computeEnergy()<<std::endl;
//                smokeSimulator.calculateCurl();
//                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
                smokeSimulator.output("../Out/MacCormack/density/", "smoke_density_original", i, NULL);
            }
        }
            break;

        case 6:
        {
            std::string folder = "../Out/FLIP/";
            std::string filename = "FLIP_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
            smokeSimulator.initCmap(nx,ny,4);
//            smokeSimulator.setInitVelocity(vortDistance);
            cout<< "case 7 " << endl;
//            smokeSimulator.applyVelocityBoundary();
//            smokeSimulator.projection(1e-6,false);
//            smokeSimulator.initParticleVelocity();
//            smokeSimulator.calculateCurl();
//            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            for (int i = 0; i <= timesteps; i++) {
                std::cout << "FLIP Frame " << i << ": ";
//                for (int j = 0; j < 2; j++)
//                {
                    smokeSimulator.advanceFLIP(dt, i, emitframe);
//                }
                std::cout<<"Energy:"<<smokeSimulator.computeEnergy()<<std::endl;
//                smokeSimulator.calculateCurl();
//                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
            }
        }
            break;

        case 7:
        {
            std::string folder = "../Out/FGCmap2/";
            std::string filename = "FGCmap2_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
            smokeSimulator.initFGCmap2(nx,ny,4);
            smokeSimulator.setInitVelocity(vortDistance);
//            smokeSimulator.setInitDensity(PI-1.6, 0.3);
//            smokeSimulator.setInitLeapFrog(1.5,3.0);
//            smokeSimulator.outputVel("U_", "V_", 0);
            cout<< endl << "********** Case 8 **********" << endl << endl;
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.projection(1e-9,false);
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            smokeSimulator.initFGCmap2Fields();
            smokeSimulator.advanceFGCmap2(0.5*dt, 0);
            for (int i = 1; i < timesteps; i++)
            {
//                if (i==50) smokeSimulator.setInitDensity(PI-1.7, 0.35);
                std::cout << "FGCmap2 Frame " << i << ": ";
                for (int j = 0; j<2; j++)
                {
                    smokeSimulator.advanceFGCmap2(dt, i*2+j);
                    float e = smokeSimulator.computeEnergy();
                    std::cout<<"Energy:"<< e <<std::endl;
                    smokeSimulator.outputEnergy(folder, "energy.txt", dt*(i*2+j), e);
                }
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), i+1);
//                smokeSimulator.output("../Out/FGCmap2/density/", "smoke_density_original", i, NULL);
            }
            cout<<"total remapping:"<<smokeSimulator.total_resampleCount<<std::endl;
        }
            break;
        case 8:
        {
            std::string folder = "../Out/FGCdecouple/";
            std::string filename = "FGCdecouple_dt" + std::to_string(dt).substr(0,6) + "_vort_dist_" + std::to_string(vortDistance).substr(0,4) +"_";
            smokeSimulator.initFGCmap2(nx,ny,4);
            smokeSimulator.setInitDensity(0.5f*L*ny/nx, smokeSimulator.rho_init, smokeSimulator.temp_init);
//            smokeSimulator.setInitVelocity(vortDistance);
//            smokeSimulator.setInitDensity(PI-1.6, 0.3);
//            smokeSimulator.setInitLeapFrog(1.5,3.0);
//            smokeSimulator.outputVel("U_", "V_", 0);
//            smokeSimulator.setEmitter(0.5f, 0.5f, 0.05);
            int emitframe = 200;
            cout<< endl << "********** Case 12 **********" << endl << endl;
//            smokeSimulator.applyVelocityBoundary();
//            smokeSimulator.projection(1e-9,false);
//            smokeSimulator.calculateCurl();
//            smokeSimulator.outputVortVisualized(folder.c_str(), filename.c_str(), 0);
            smokeSimulator.initFGCmap2Fields();
            for (int i = 0; i < timesteps; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    std::cout << GREEN << "FGCmap2 Frame " << i << ": " << RESET;
                    smokeSimulator.advanceFGCdecouple(dt/(float)N, i, emitframe);
                }
                smokeSimulator.output("../Out/FGCdecouple/", "smoke_density_original", i, NULL);
            }
            cout<< RED << "Velocity total remapping:"<<smokeSimulator.total_resampleCount<< RESET << std::endl;
            cout<< BLUE << "Density total remapping:"<<smokeSimulator.total_rho_resample<< RESET << std::endl;
        }
            break;
        }

    return 0;
}