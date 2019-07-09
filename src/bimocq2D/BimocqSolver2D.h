#ifndef BIMOCQSOLVER2D_H
#define BIMOCQSOLVER2D_H
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "tbb/tbb.h"
#include "../include/array2.h"
#include "../include/vec.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/GeometricLevelGen.h"
#include "../utils/writeBMP.h"
#include "../utils/visualize.h"
#include "../utils/color_macro.h"
#include <boost/filesystem.hpp>

enum Scheme {SEMILAG, MACCORMACK, BFECC, MAC_REFLECTION, FLIP, APIC, POLYPIC, BIMOCQ};

inline std::string enumToString(const Scheme &sim_scheme)
{
    switch(sim_scheme)
    {
        case SEMILAG:
            return std::string("Semilag");
        case MACCORMACK:
            return std::string("MacCormack");
        case BFECC:
            return std::string("BFECC");
        case MAC_REFLECTION:
            return std::string("Reflection");
        case FLIP:
            return std::string("FLIP");
        case APIC:
            return std::string("APIC");
        case POLYPIC:
            return std::string("PolyPIC");
        case BIMOCQ:
            return std::string("BiMocq");
    }
}

class CmapParticles
{
public:
    CmapParticles()
    {
        vel=Vec2f();
        pos_current = Vec2f();
        rho = 0;
        temperature = 0;
        C_x = Vec4f(0.0);
        C_y = Vec4f(0.0);
        C_rho = Vec4f(0.0);
        C_temperature = Vec4f(0.0);
    }
    ~CmapParticles(){}
    Vec2f vel;
    Vec2f pos_current;
    float rho;
    float temperature;
    Vec4f C_x;
    Vec4f C_y;
    Vec4f C_rho;
    Vec4f C_temperature;

    CmapParticles(const CmapParticles &p)
    {
        vel = p.vel;
        pos_current = p.pos_current;
        rho = p.rho;
        temperature = p.temperature;
        C_x = p.C_x;
        C_y = p.C_y;
        C_rho = p.C_rho;
        C_temperature = p.C_temperature;
    }
    inline float kernel(float r)
    {
        return fabs(r) < 1.0 ? 1 - fabs(r) : 0;
    }
    inline float frand(float a, float b)
    {
        return a + (b-a) * rand()/(float)RAND_MAX;
    }
    inline Vec4f calculateCp(Vec2f pos, const Array2f &field, float h, int ni, int nj, float offx, float offy)
    {
        Vec4f Cp = Vec4f(0.f);
        Vec2f spos = pos - h*Vec2f(offx, offy);
        int i = floor(spos.v[0] / h), j = floor(spos.v[1] / h);
        float px = spos.v[0] - i*h;
        float py = spos.v[1] - j*h;
        if (offy > 0)
        {
            if (!(i >= 0 && i <= ni - 1 && j >= 0 && j <= nj - 2))
                return Cp;
            else{
                Cp[0] = ((h-px)*(h-py)*field(i,j) + px*(h-py)*field(i+1,j)
                         + px*py*field(i+1,j+1) + (h-px)*py*field(i,j+1)) / (h*h);
                Cp[1] = (-(h-py)*field(i,j) + (h-py)*field(i+1,j) + py*field(i+1,j+1)
                         -py*field(i,j+1))/(h*h);
                Cp[2] = (-(h-px)*field(i,j) - px*field(i+1,j) + px*field(i+1,j+1)
                         + (h-px)*field(i,j+1))/(h*h);
                Cp[3] = (field(i,j) - field(i+1,j) + field(i+1,j+1) - field(i,j+1))/(h*h);
                return Cp;
            }
        }
        else
        {
            if (!(i >= 0 && i <= ni - 2 && j >= 0 && j <= nj - 1))
                return Cp;
            else{
                Cp[0] = ((h-px)*(h-py)*field(i,j) + px*(h-py)*field(i+1,j)
                         + px*py*field(i+1,j+1) + (h-px)*py*field(i,j+1)) / (h*h);
                Cp[1] = (-(h-py)*field(i,j) + (h-py)*field(i+1,j) + py*field(i+1,j+1)
                         -py*field(i,j+1))/(h*h);
                Cp[2] = (-(h-px)*field(i,j) - px*field(i+1,j) + px*field(i+1,j+1)
                         + (h-px)*field(i,j+1))/(h*h);
                Cp[3] = (field(i,j) - field(i+1,j) + field(i+1,j+1) - field(i,j+1))/(h*h);
                return Cp;
            }
        }
    }
};

class BimocqSolver2D {
public:
    void clampPos(Vec2f &pos)
    {
        pos[0] = min(max(h, pos[0]),(float)ni*h-h);
        pos[1] = min(max(h, pos[1]),(float)nj*h-h);
    }
    BimocqSolver2D(int nx, int ny, float L, float b_coeff, int N, bool bc, Scheme s_scheme);
    ~BimocqSolver2D() {};
    int ni, nj;
    inline	float lerp(float v0, float v1, float c);
    inline	float bilerp(float v00, float v01, float v10, float v11, float cx, float cy);
    void semiLagAdvect(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y);
    void solveMaccormack(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety);
    void solveBFECC(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety);
    void applyBuoyancyForce(float dt);
    void calculateCurl();
    void projection(float tol, bool PURE_NEUMANN);
    void seedParticles(int N);

    void resampleVelBuffer(float dt);
    void resampleRhoBuffer(float dt);

    void advance(float dt, int frame);
    void advanceSemilag(float dt, int currentframe);
    void advanceReflection(float dt, int currentframe);
    void advanceBFECC(float dt, int currentframe);
    void advanceMaccormack(float dt, int currentframe);
    void advanceFLIP(float dt, int currentframe);
    void advancePolyPIC(float dt, int currentframe);
    void advanceBIMOCQ(float dt, int currentframe);

    void clampExtrema2(int _ni, int _nj,Array2f &before, Array2f &after);
    void updateForward(float dt, Array2f &fwd_x, Array2f &fwd_y);
    void updateBackward(float dt, Array2f &back_X, Array2f &back_y);

    void advectVelocity(Array2f &semi_u, Array2f &semi_v);
    void advectScalars(Array2f &semi_rho, Array2f &semi_T);
    void accumulateVelocity(Array2f &u_change, Array2f &v_change, float proj_coeff, bool error_correction);
    void accumulateScalars(Array2f &rho_change, Array2f &T_change, bool error_correction);

    void buildMultiGrid(bool PURE_NEUMANN);
    void diffuseField(float nu, float dt, Array2f &field);
    void applyVelocityBoundary();

    void sampleParticlesFromGrid();

    void correctScalars(Array2f &semi_rho, Array2f &semi_T);
    void correctVelocity(Array2f &semi_u, Array2f &semi_v);

    /// new scheme for SEMILAG advection
    Vec2f calculateA(Vec2f pos, float h);
    void semiLagAdvectDMC(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y);
    inline Vec2f solveODEDMC(float dt, Vec2f &pos);
    inline Vec2f traceDMC(float dt, Vec2f &pos, Vec2f &a);

    void setInitReyleighTaylor(float layer_height);
    void setInitVelocity(float distance);
    void setInitLeapFrog(float dist1, float dist2, float rho_h, float rho_w);
    void setInitZalesak();
    void setInitVortexBox();
    void setSmoke(float smoke_rise, float smoke_drop);

    float maxVel();
    float estimateDistortion(Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y);
	inline Vec2f traceRK3(float dt, Vec2f &pos);
    inline Vec2f solveODE(float dt, Vec2f &pos);
    void emitSmoke();
    void outputDensity(std::string folder, std::string file, int i, bool color_density);
    void outputVortVisualized(std::string folder, std::string file, int i);
    void outputLevelset(std::string sdfFilename, int i);

    color_bar cBar;
    int total_resampleCount = 0;
    int total_scalar_resample = 0;
    int resampleCount = 0;
    int frameCount = 0;
    void nostickBC();
    void getCFL();
    Vec2f getVelocity(Vec2f &pos);
    float sampleField(Vec2f pos, const Array2f &field);

    float h;
    float alpha, beta;
    Array2f u, v, u_temp, v_temp;
    Array2f rho, temperature, s_temp;
    Array2f curl;
    Array2c emitterMask;
    Array2c emitterMaska;
    Array2c boundaryMask;
    std::vector<double> pressure;
    std::vector<double> rhs;

    //linear solver data
    SparseMatrixd matrix;
    FixedSparseMatrixd matrix_fix;
    std::vector<FixedSparseMatrixd *> A_L;
    std::vector<FixedSparseMatrixd *> R_L;
    std::vector<FixedSparseMatrixd *> P_L;
    std::vector<Vec2i>                S_L;
    int total_level;
    //solver
    levelGen<double> mgLevelGenerator;

    float _cfl;
    std::vector<CmapParticles> cParticles;

    // BIMOCQ mapping buffers
    Array2f forward_x;
    Array2f forward_y;
    Array2f forward_scalar_x;
    Array2f forward_scalar_y;
    Array2f backward_x;
    Array2f backward_y;
    Array2f backward_xprev;
    Array2f backward_yprev;
    Array2f backward_scalar_x;
    Array2f backward_scalar_y;
    Array2f backward_scalar_xprev;
    Array2f backward_scalar_yprev;
    Array2f map_tempx;
    Array2f map_tempy;

    // fluid buffers
    Array2f u_init;
    Array2f v_init;
    Array2f u_origin;
    Array2f v_origin;
    Array2f du;
    Array2f dv;
    Array2f du_temp;
    Array2f dv_temp;
    Array2f du_proj;
    Array2f dv_proj;
    Array2f drho;
    Array2f drho_temp;
    Array2f drho_prev;
    Array2f dT;
    Array2f dT_temp;
    Array2f dT_prev;
    Array2f rho_init;
    Array2f rho_orig;
    Array2f T_init;
    Array2f T_orig;

    // for Maccormack
    Array2f u_first;
    Array2f v_first;
    Array2f u_sec;
    Array2f v_sec;

    Array2f du_prev;
    Array2f dv_prev;

    int lastremeshing = 0;
    int rho_lastremeshing = 0;
    float blend_coeff;
    bool use_neumann_boundary;
    Scheme sim_scheme;
    bool advect_levelset = false;
};

#endif //BIMOCQSOLVER2D_H
