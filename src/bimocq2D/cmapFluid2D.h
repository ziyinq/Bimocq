//
// Created by ziyin on 18-10-3.
//

#ifndef DEEPSIM_CMAPFLUID2D_H
#define DEEPSIM_CMAPFLUID2D_H
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "tbb/tbb.h"
#include "../include/array2.h"
#include "../include/vec.h"
#include "AlgebraicMultigrid.h"
#include "GeometricLevelGen.h"
#include "../utils/writeBMP.h"
#include "../utils/visualize.h"

#define PI 3.141592653


#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


struct mat2x2{
    double v[4];
    mat2x2()
    {}

    mat2x2(double a, double b, double c, double d)
    {
        v[0] = a, v[1] = b, v[2] = c, v[3] = d;
    }

    mat2x2(const mat2x2 &m)
    {
        for(int i=0; i<4; i++) v[i] = m[i];
    }

    const double& operator()(int i, int j) const
    {
        return v[j*2 + i];
    }
    double& operator()(int i, int j)
    {
        return v[j*2 + i];
    }

    const double& operator[](int i) const
    {
        return v[i];
    }
    double& operator[](int i)
    {
        return v[i];
    }

    double det()
    {
        return v[0]*v[3] - v[1]*v[2];
    }
    mat2x2 inverse() {
        double d = det();
        return mat2x2(v[3] / d, -v[1] / d, -v[2] / d, v[0] / d);
    }
    mat2x2 operator*(const mat2x2 &m)
    {
        return mat2x2(v[0]*m[0]+v[1]*m[2], v[0]*m[1]+v[1]*m[3], v[2]*m[0]+v[3]*m[2], v[2]*m[1]+v[3]*m[3]);
    }
    Vec2f operator*(const Vec2f &vec)
    {
        Vec2f v0 = Vec2f(v[0], v[1]);
        Vec2f v1 = Vec2f(v[2], v[3]);
        Vec2f res = Vec2f(dot(v0,vec), dot(v1,vec));
        return res;
    }
    mat2x2 operator*=(const double &c)
    {
        v[0]*=c; v[1]*=c; v[2]*=c; v[3]*=c;
        return *this;
    }
};

inline float kernel(float r)
{
    return r <= 1.0 ? 1 - r : 0;
}

inline float frand(float a, float b)
{
    return a + (b-a) * rand()/(float)RAND_MAX;
}


class CmapParticles
{
public:
    CmapParticles()
    {
        vel=Vec2f();
        pos_start = Vec2f();
        pos_current = Vec2f();
        rho = 0;
        drho = 0;
        dtemp = 0;
        du = 0;
        dv = 0;
        C_x = Vec4f(0.0);
        C_y = Vec4f(0.0);
    }
    ~CmapParticles(){}
    Vec2f vel;
    Vec2f pos_start;
    Vec2f pos_current;
    float rho;
    float drho;
    float dtemp;
    float du;
    float dv;
    Vec4f C_x;
    Vec4f C_y;

    CmapParticles(const CmapParticles &p)
    {
        vel = p.vel;
        pos_start = p.pos_start;
        pos_current = p.pos_current;
        drho = p.drho;
        dtemp = p.dtemp;
        du = p.du;
        dv = p.dv;
        C_x = p.C_x;
        C_y = p.C_y;
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

class cmapFluid2D {
public:
    void clampPos(Vec2f &pos)
    {
        pos[0] = min(max(h, pos[0]),(float)ni*h-h);
        pos[1] = min(max(h, pos[1]),(float)nj*h-h);
    }
    cmapFluid2D() {};
    ~cmapFluid2D() {};
    int ni, nj;
    void diffuseField(float nu, float dt, Array2f &field);
    inline	float lerp(float v0, float v1, float c);
    inline	float bilerp(float v00, float v01, float v10, float v11, float cx, float cy);
    void semiLagAdvect(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y);
    void semiLagAdvect_fmap(const Array2f &srcx, const Array2f &srcy, Array2f & dstx, Array2f & dsty, float dt, int ni, int nj, float off_x, float off_y);
    void semiLagAdvect_correct(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y, const Array2f &diff_x, const Array2f &diff_y);
    void solveAdvection(float dt);
    void solveAdvectionCmap();
    void solveMaccormack(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety);
    void solveBFECC(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety);
    void applyBouyancyForce(float dt);
    void applyForce();
    void calculateCurl();
    void projection(float tol, bool bc);
    void advance(float dt, int currentframe, int emitframe, unsigned char* boundary);
    void setBoundary(unsigned char* boundary);
    void seedParticles(int N);

    void advanceCmap(float dt, int currentframe, int emitframe, unsigned char* boundary);
    void advanceGridCmap(float dt, int currentframe, int emitframe, unsigned char* boundary);

    void advanceFGCmap2(float dt, int currentframe);
    void resampleFGCmap2(float dt, Array2f &du_last, Array2f dv_last);
    void resampleVelBuffer(float dt);
    void resampleRhoBuffer(float dt);
    void initFGCmap2(int nx, int ny, int N);

    void advanceReflection(float dt, int currentframe, int emitframe, unsigned char* boundary);
    void advanceBFECC(float dt, int currentframe, int emitframe, unsigned char* boundary);
    void advanceMaccormack(float dt, int currentframe, int emitframe, unsigned char* boundary);
    void advanceFLIP(float dt, int currentframe, int emitframe);
    void advanceAPIC(float dt);
    void advancePolyPIC(float dt);

    void clampExtrema2(int _ni, int _nj,Array2f &before, Array2f &after);
    void advanceFGCdecouple(float dt, int currentframe, int emitframe);
    void updateForward(float dt, Array2f &fwd_x, Array2f &fwd_y);
    void updateBackward(float dt, Array2f &back_X, Array2f &back_y);


    void backwardmap(const Array2f &xinit, const Array2f &yinit, const Array2f &xfwd, const Array2f &yfwd, Array2f & dstx, Array2f & dsty, float dt, int ni, int nj, float off_x, float off_y, int sample_num);
    void advectParticles(float dt);
    void characteristicMap(float dt);

    void advectVelocity(bool db, Array2f &semi_u, Array2f &semi_v);
    void advectRho(bool db, Array2f &semi_rho, Array2f &semi_T, Array2f &back_x, Array2f &back_y);
    void cumulateVelocity(float c, bool correct);
    void cumulateScalar(Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y, bool correct);

    void buildMultiGrid();
    void buildMultiGrid_reflect();
    void buildMultiGridPsi();

    void applyVelocityBoundary();

    void init(int nx, int ny, float L);
    void initCmap(int nx, int ny, int N);
    void initParticleVelocity();
    void initGridCmap(int nx, int ny, int N);
    void initFGCmap(int nx, int ny, int N);
    void initMaccormack();

    void correctD(float c);
    void correctRhoRemapping(Array2f &semi_rho, Array2f &semi_T, Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y);
    void correctVelRemapping(Array2f &semi_u, Array2f &semi_v);
    void resampleParticle(int N);
    void resampleGrid();
    void resampleFGCmap(float dt);
    void resampleFGCmap_cummulative();

    /// new scheme for SEMILAG advection
    Vec2f calculateA(Vec2f pos, float h);
    Vec2f calculateAF(Vec2f pos, float h);
    void semiLagAdvectDMC(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y);
    inline Vec2f solveODEDMC(float dt, Vec2f &pos);
    inline Vec2f solveODEDMCF(float dt, Vec2f &pos);
    inline Vec2f traceDMC(float dt, Vec2f &pos, Vec2f &a);
    inline Vec2f traceDMCF(float dt, Vec2f &pos, Vec2f &a);


    double computeEnergy();

    void init_interpolate(int nx, int ny, float L);
    void setInitDensity(float h, Array2f &buffer, Array2f &buffer_sec);
    void setInitVelocity(float distance);
    void setInitLeapFrog(float dist1, float dist2);
    void setAlpha(float a) { alpha = a; }
    void setBeta(float b) { beta = b; }
    void setEmitter(float x, float y, float r)
    {
        emitterMask.assign(ni, nj, (char)0);
        for (int j = 0;j < nj;j++)for (int i = 0;i < ni;i++)
            {
                Vec2f pos = h*(Vec2f(i, j) + Vec2f(0.5f));
                if (sqrt((pos.v[0] - x)*(pos.v[0] - x) + (pos.v[1] - y)*(pos.v[1] - y)) <= r)
                {
                    emitterMask(i, j) = 1;
                }
            }

    }
    void clampExtrema(Array2f & src, Array2f & dst,float dt, float offsetx, float offsety);
    void setEmitter_a(float x, float y, float r, float u_vel, float v_vel)
    {
        emitterMaska.assign(ni, nj, (char)0);
        for (int j = 0;j < nj;j++)for (int i = 0;i < ni;i++)
            {
                Vec2f pos = h*(Vec2f(i, j) + Vec2f(0.5f));
                Vec2f pos_u = h*(Vec2f(i, j) + Vec2f(0.0, 0.5f));
                Vec2f pos_v = h*(Vec2f(i, j) + Vec2f(0.5, 0.0f));
                if (sqrt((pos.v[0] - x)*(pos.v[0] - x) + (pos.v[1] - y)*(pos.v[1] - y)) <= r)
                {
                    emitterMaska(i, j) = 1;
                }
                if (sqrt((pos_u.v[0] - x)*(pos_u.v[0] - x) + (pos_u.v[1] - y)*(pos_u.v[1] - y)) <= r)
                {
                    u(i, j) = u_vel;
                }
                if (sqrt((pos_v.v[0] - x)*(pos_v.v[0] - x) + (pos_v.v[1] - y)*(pos_v.v[1] - y)) <= r)
                {
                    v(i, j) = v_vel;
                }
            }

    }
    float maxVel();
    float estimateDistortion(Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y);
    float estimateDistortionFace();
	inline Vec2f traceRK3(float dt, Vec2f &pos);
    inline Vec2f solveODE(float dt, Vec2f &pos);
    void emitSmoke();
    void output(std::string folder, std::string file, int i, unsigned char* boundary)
    {
        // these three lines of code doesn't work on Linux
//		std::string command_line_call;
//		command_line_call = std::string("if not exist ") + folder + std::string(" mkdir ") + folder;
//		system(command_line_call.c_str());
        std::string filestr;
        filestr = folder + file + std::string("_\%05d.bmp");
        char filename[1024];
        sprintf(filename, filestr.c_str(), i);
//        writeBMP(filename, ni, nj, rho.a.data);
        writeBMPColor(filename, ni, nj, rho.a.data, temperature.a.data);
//        writeBMPboundary(filename, ni, nj, rho.a.data, boundary);
    }
    void outputEnergy(std::string folder, std::string file, float t, float energy)
    {
        std::ofstream fout;
        std::string filestr = folder + file;
        fout.open(filestr, std::ios_base::app);
        fout << t << " " << energy << std::endl;
        fout.close();
    }
    void initFGCmap2Fields()
    {
        u_init = u;
        v_init = v;
        u_origin = u_init;
        v_origin = v_init;
        du.assign(0);
        dv.assign(0);
        du_prev = du;
        dv_prev = dv;
        x_origin_buffer = grid_xinit;
        y_origin_buffer = grid_yinit;
        difu_prev.assign(ni+1,nj,0.0f);
        difv_prev.assign(ni,nj+1,0.0f);

    }
    void outputVortVisualized(std::string folder, std::string file, int i)
    {
        std::string filestr;
        filestr = folder + file + std::string("_\%05d.bmp");
        char filename[1024];
        sprintf(filename, filestr.c_str(), i);
        std::vector<Vec3uc> color;
        color.resize(ni*nj);
        tbb::parallel_for((int)0, (int)(ni*nj), 1, [&](int tIdx)
        {
            int i = tIdx%ni;
            int j = tIdx/ni;

            float vort = 0.25*(curl(i,j)+curl(i+1,j)+curl(i,j+1)+curl(i+1,j+1));
            color[j*ni + i] = cBar.toRGB(fabs(vort));
        });
        wrtieBMPuc3(filename, ni, nj, (unsigned char*)(&(color[0])));
    }

    void outputVel(std::string uFilename, std::string vFilename, int i)
    {
        std::ofstream foutU;
        std::ofstream foutV;
        std::string filenameU = uFilename + std::to_string(i) + std::string(".txt");
        std::string filenameV = vFilename + std::to_string(i) + std::string(".txt");
        foutU.open(filenameU);
        foutV.open(filenameV);
        for (int i = 0; i<=ni/2; i++)
        {
            for (int j = 0; j<nj+1; j++)
            {
                if (j != nj){
                    foutU << u(i,j) << " ";
                }
                if (i != ni/2){
                    foutV << v(i,j) << " ";
                }
            }
            foutU << std::endl;
            foutV << std::endl;
        }
        foutU.close();
        foutV.close();
        std::cout << "Output velocity finished!" << std::endl;
    }

    color_bar cBar;
    int total_resampleCount = 0;
    int total_rho_resample = 0;
    int resampleCount = 0;
    int frameCount = 0;
    void nostickBC();
    void getCFL();
    Vec2f getVelocity(Vec2f &pos);
    float sampleField(Vec2f pos, const Array2f &field);
    float map_h;
    int map_ni, map_nj;
    float h;
    float alpha, beta;
    Array2f p_temp;
    Array2f u, v, u_temp, v_temp, u_mean, v_mean;
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

    // for characteristic map
    float _cfl;
    float _cfl_dt;
    std::vector<CmapParticles> cParticles;
    Array2f gridpos_x;
    Array2f gridpos_y;
    Array2f rho_init;
    Array2f rho_orig;
    Array2f rho_diff;
    Array2f temp_init;
    Array2f temp_diff;
    Array2f u_init;
    Array2f u_diff;
    Array2f v_init;
    Array2f v_diff;
    Array2f weight;

    // for grid cmap
    Array2f grid_x;
    Array2f grid_xinit;
    Array2f grid_x_new;
    Array2f grid_tempx;
    Array2f grid_y;
    Array2f grid_yinit;
    Array2f grid_y_new;
    Array2f grid_tempy;
    Array2f du;
    Array2f du_temp;
    Array2f dv;
    Array2f dv_temp;
    Array2f drho;
    Array2f drho_temp;
    Array2f drho_prev;
    Array2f dT;
    Array2f dT_temp;

    // for Maccormack
    Array2f u_first;
    Array2f v_first;
    Array2f u_sec;
    Array2f v_sec;

    Array2f grid_pre_x;
    Array2f grid_pre_y;
    Array2f grid_back_x;
    Array2f grid_back_y;
    Array2f diff_x;
    Array2f diff_y;

    Array2f u_prev;
    Array2f v_prev;
    Array2f  u_origin;
    Array2f  v_origin;
    Array2f  du_prev;
    Array2f  dv_prev;
    Array2f  x_origin_buffer;
    Array2f  y_origin_buffer;

    Array2f  difu_prev;
    Array2f  difv_prev;

    Array2f grid_xscalar;
    Array2f grid_yscalar;

    Array2i index_x;
    Array2i index_y;
    Array2f max_dist;


    int lastremeshing = 0;
    int rho_lastremeshing = 0;
};

#endif //DEEPSIM_CMAPFLUID2D_H
