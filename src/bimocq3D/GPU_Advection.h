

#ifndef IVOCK_GPU_ADVECTION_H
#define IVOCK_GPU_ADVECTION_H


#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <iostream>
#include "../include/fluid_buffer3D.h"
#include <tbb/tbb.h>

extern "C" void gpu_solve_forward(float *u, float *v, float *w,
                                  float *x_fwd, float *y_fwd, float *z_fwd,
                                  float h, int ni, int nj, int nk, float cfldt, float dt);

extern "C" void gpu_solve_backwardDMC(float *u, float *v, float *w,
                                      float *x_in, float *y_in, float *z_in,
                                      float *x_out, float *y_out, float *z_out,
                                      float h, int ni, int nj, int nk, float substep);

extern "C" void gpu_advect_velocity(float *u, float *v, float *w,
                                    float *u_init, float *v_init, float *w_init,
                                    float *backward_x, float *backward_y, float *backward_z,
                                    float h, int ni, int nj, int nk, bool is_point);

extern "C" void gpu_advect_vel_double(float *u, float *v, float *w,
                                      float *utemp, float *vtemp, float *wtemp,
                                      float *backward_x, float *backward_y, float *backward_z,
                                      float *backward_xprev,  float *backward_yprev, float *backward_zprev,
                                      float h, int ni, int nj, int nk, bool is_point, float blend_coeff);

extern "C" void gpu_advect_field(float *field, float *field_init,
                                 float *backward_x, float *backward_y, float *backward_z,
                                 float h, int ni, int nj, int nk, bool is_point);

extern "C" void gpu_advect_field_double(float *field, float *field_init,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float *backward_xprev, float *backward_yprev, float *backward_zprev,
                                        float h, int ni, int nj, int nk, bool is_point, float blend_coeff);

extern "C" void gpu_accumulate_velocity(float *u_change, float *v_change, float *w_change,
                                        float *du_init, float *dv_init, float *dw_init,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float h, int ni, int nj, int nk, bool is_point, float coeff);

extern "C" void gpu_accumulate_field(float *field_change, float *dfield_init,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float h, int ni, int nj, int nk, bool is_point, float coeff);

extern "C" void gpu_estimate_distortion(float *du,
                                        float *x_init, float *y_init, float *z_init,
                                        float *x_fwd, float *y_fwd, float *z_fwd,
                                        float h, int ni, int nj, int nk);

extern "C" void gpu_add(float *field1, float *field2, float coeff, int number);

extern "C" void gpu_compensate_velocity(float *u, float *v, float *w,
                                        float *du, float *dv, float *dw,
                                        float *u_src, float *v_src, float *w_src,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float h, int ni, int nj, int nk, bool is_point);

extern "C" void gpu_compensate_field(float *u, float *du, float *u_src,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float *backward_x, float *backward_y, float *backward_z,
                                     float h, int ni, int nj, int nk, bool is_point);

extern "C" void gpu_semilag(float *field, float *field_src,
                            float *u, float *v, float *w,
                            int dim_x, int dim_y, int dim_z,
                            float h, int ni, int nj, int nk, float cfldt, float dt);

class gpuMapper{
public:
    gpuMapper(){}
    gpuMapper(int nx, int ny, int nz, float h)
    {
        cudaInit();
        ni = nx; nj = ny; nk = nz;
        _hx = h;
        allocGPUBuffer((void**)&u, (ni+1)*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&v, ni*(nj+1)*nk*sizeof(float));
        allocGPUBuffer((void**)&w, ni*nj*(nk+1)*sizeof(float));

        allocGPUBuffer((void**)&u_src, (ni+1)*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&v_src, ni*(nj+1)*nk*sizeof(float));
        allocGPUBuffer((void**)&w_src, ni*nj*(nk+1)*sizeof(float));

        allocGPUBuffer((void**)&du, (ni+1)*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&dv, ni*(nj+1)*nk*sizeof(float));
        allocGPUBuffer((void**)&dw, ni*nj*(nk+1)*sizeof(float));

        allocGPUBuffer((void**)&x_in, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&y_in, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&z_in, ni*nj*nk*sizeof(float));

        allocGPUBuffer((void**)&x_out, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&y_out, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&z_out, ni*nj*nk*sizeof(float));

        u_host = new float[(ni+1)*nj*nk];
        v_host = new float[ni*(nj+1)*nk];
        w_host = new float[ni*nj*(nk+1)];

        x_host = new float[ni*nj*nk];
        y_host = new float[ni*nj*nk];
        z_host = new float[ni*nj*nk];
    }
    ~gpuMapper(){}
    int ni, nj, nk;
    float _hx;
    float *x_in;
    float *y_in;
    float *z_in;

    float *x_out;
    float *y_out;
    float *z_out;

    float *x_host;
    float *y_host;
    float *z_host;

    float *u;
    float *v;
    float *w;

    float *u_src;
    float *v_src;
    float *w_src;

    float *du;
    float *dv;
    float *dw;



    float *u_host;
    float *v_host;
    float *w_host;


    void init(int nx, int ny, int nz, float h)
    {
        cudaInit();
        ni = nx; nj = ny; nk = nz;
        _hx = h;
        allocGPUBuffer((void**)&u, (ni+1)*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&v, ni*(nj+1)*nk*sizeof(float));
        allocGPUBuffer((void**)&w, ni*nj*(nk+1)*sizeof(float));

        allocGPUBuffer((void**)&u_src, (ni+1)*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&v_src, ni*(nj+1)*nk*sizeof(float));
        allocGPUBuffer((void**)&w_src, ni*nj*(nk+1)*sizeof(float));

        allocGPUBuffer((void**)&du, (ni+1)*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&dv, ni*(nj+1)*nk*sizeof(float));
        allocGPUBuffer((void**)&dw, ni*nj*(nk+1)*sizeof(float));

        allocGPUBuffer((void**)&x_in, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&y_in, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&z_in, ni*nj*nk*sizeof(float));

        allocGPUBuffer((void**)&x_out, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&y_out, ni*nj*nk*sizeof(float));
        allocGPUBuffer((void**)&z_out, ni*nj*nk*sizeof(float));

        u_host = new float[(ni+1)*nj*nk];
        v_host = new float[ni*(nj+1)*nk];
        w_host = new float[ni*nj*(nk+1)];

        x_host = new float[ni*nj*nk];
        y_host = new float[ni*nj*nk];
        z_host = new float[ni*nj*nk];
    }


    void cudaInit()
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(0, NULL);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void copyHostToDevice(const buffer3Df &field, float *host_field, float *device_field, size_t device_size)
    {
        int compute_elements = field._blockx*field._blocky*field._blockz;
        int slice = field._blockx*field._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](uint thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/field._blockx;
            uint bi = thread_idx%(field._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<field._nx && j<field._ny && k<field._nz)
                {
                    int index = i + field._nx*j + field._nx * field._ny * k;
                    host_field[index] = field(i,j,k);
                }
            }
        });
        size_t host_size = field._nx*field._ny*field._nz*sizeof(float);
        // clear gpu buffer before we use it
        cudaMemset(device_field, 0, device_size);
        cudaMemcpy(device_field, host_field, host_size, cudaMemcpyHostToDevice);
    }

    void copyDeviceToHost(buffer3Df &field, float *host_field, float *device_field)
    {
        size_t size = sizeof(float)*field._nx*field._ny*field._nz;
        cudaMemcpy(host_field, device_field, size, cudaMemcpyDeviceToHost);

        int compute_elements = field._blockx*field._blocky*field._blockz;
        int slice = field._blockx*field._blocky;
        tbb::parallel_for(0, compute_elements, 1, [&](uint thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/field._blockx;
            uint bi = thread_idx%(field._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<field._nx && j<field._ny && k<field._nz)
                {
                    int index = i + field._nx*j + field._nx * field._ny * k;
                    field(i,j,k) = host_field[index];
                }
            }
        });
    }

    void allocGPUBuffer(void ** buffer, size_t size)
    {
        cudaMalloc(buffer, size);
    }

    void solveForward(float cfl_dt, float dt)
    {
        gpu_solve_forward(u, v, w, x_out, y_out, z_out, _hx, ni, nj, nk, cfl_dt, dt);
    }

    void solveBackwardDMC(float substep)
    {
        cudaMemcpy(x_out, x_in, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(y_out, y_in, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(z_out, z_in, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
        gpu_solve_backwardDMC(u, v, w, x_in, y_in, z_in, x_out, y_out, z_out, _hx, ni, nj, nk, substep);
        cudaMemcpy(x_in, x_out, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(y_in, y_out, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(z_in, z_out, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
    }

    void advectVelocity(bool is_point)
    {
        cudaMemset(u, 0, sizeof(float)*(ni+1)*nj*nk);
        cudaMemset(v, 0, sizeof(float)*ni*(nj+1)*nk);
        cudaMemset(w, 0, sizeof(float)*ni*nj*(nk+1));
        gpu_advect_velocity(u, v, w, du, dv, dw, x_in, y_in, z_in, _hx, ni, nj, nk, is_point);
    }

    void advectVelocityDouble(bool is_point, float blend_coeff)
    {
        gpu_advect_vel_double(u, v, w, du, dv, dw, x_in, y_in, z_in, x_out, y_out, z_out, _hx, ni, nj, nk, is_point, blend_coeff);
    }

    void advectField(bool is_point)
    {
        cudaMemset(u, 0, sizeof(float)*(ni+1)*nj*nk);
        gpu_advect_field(u, du, x_in, y_in, z_in, _hx, ni, nj, nk, is_point);
    }

    void advectFieldDouble(bool is_point, float blend_coeff)
    {
        gpu_advect_field_double(u, du, x_in, y_in, z_in, x_out, y_out, z_out, _hx, ni, nj, nk, is_point, blend_coeff);
    }

    void accumulateVelocity(bool is_point, float coeff)
    {
        gpu_accumulate_velocity(u, v, w, du, dv, dw, x_out, y_out, z_out, _hx, ni, nj, nk, is_point, coeff);
    }

    void accumulateField(bool is_point, float coeff)
    {
        gpu_accumulate_field(u, du, x_out, y_out, z_out, _hx, ni, nj, nk, is_point, coeff);
    }

    void compensateVelocity(bool is_point)
    {
        // clear and reuse u_src, v_src, w_src buffers to do compensation
        cudaMemset(u_src, 0, sizeof(float)*(ni+1)*nj*nk);
        cudaMemset(v_src, 0, sizeof(float)*ni*(nj+1)*nk);
        cudaMemset(w_src, 0, sizeof(float)*ni*nj*(nk+1));
        gpu_compensate_velocity(u, v, w, du, dv, dw, u_src, v_src, w_src, x_out, y_out, z_out, x_in, y_in, z_in, _hx, ni, nj, nk, is_point);
    }

    void compensateField(bool is_point)
    {
        // clear and reuse u_src, buffer to do compensation
        cudaMemset(u_src, 0, sizeof(float)*(ni+1)*nj*nk);
        gpu_compensate_field(u, du, u_src, x_out, y_out, z_out, x_in, y_in, z_in, _hx, ni, nj, nk, is_point);
    }

    void semilagAdvectVelocity(float cfldt, float dt)
    {
        cudaMemset(du, 0, sizeof(float)*(ni+1)*nj*nk);
        cudaMemset(dv, 0, sizeof(float)*ni*(nj+1)*nk);
        cudaMemset(dw, 0, sizeof(float)*ni*nj*(nk+1));
        gpu_semilag(du, u_src, u, v, w, 1, 0, 0, _hx, ni, nj, nk, cfldt, dt);
        gpu_semilag(dv, v_src, u, v, w, 0, 1, 0, _hx, ni, nj, nk, cfldt, dt);
        gpu_semilag(dw, w_src, u, v, w, 0, 0, 1, _hx, ni, nj, nk, cfldt, dt);
    }

    void semilagAdvectField(float cfldt, float dt)
    {
        cudaMemset(du, 0, sizeof(float)*(ni+1)*nj*nk);
        gpu_semilag(du, dv, u, v, w, 0, 0, 0, _hx, ni, nj, nk, cfldt, dt);
    }

    void add(float *field1, float *field2, float coeff, int number)
    {
        gpu_add(field1, field2, coeff, number);
    }

    void estimateDistortionCUDA()
    {
        cudaMemset(du, 0, sizeof(float)*(ni+1)*nj*nk);
        gpu_estimate_distortion(du, x_in, y_in, z_in, x_out, y_out, z_out, _hx, ni, nj, nk);
    }
};

#endif //IVOCK_GPU_ADVECTION_H
