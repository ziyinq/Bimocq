#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include "GPU_Advection.h"
#include <iostream>


__device__ float clamp(float a, float minv, float maxv)
{
    return fminf(fmaxf(minv, a),maxv);
}

__device__ float3 clampv3(float3 in, float3 minv, float3 maxv)
{
    float xout = clamp(in.x,minv.x,maxv.x);
    float yout = clamp(in.y,minv.y,maxv.y);
    float zout = clamp(in.z,minv.z,maxv.z);
    return make_float3(xout, yout, zout);
}

__device__ float lerp(float a, float b, float c)
{
    return (1.0-c)*a + c*b;
}

__device__ float triLerp(float v000, float v001, float v010, float v011, float v100, float v101,
        float v110, float v111, float a, float b, float c)
{
    return lerp(
            lerp(
                    lerp(v000, v001, a),
                    lerp(v010, v011, a),
                    b),
            lerp(
                    lerp(v100, v101, a),
                    lerp(v110, v111, a),
                    b),
            c);

}

__device__ float sample_buffer(float * b, int nx, int ny, int nz, float h, float3 off_set, float3 pos)
{
    float3 samplepos = make_float3(pos.x-off_set.x, pos.y-off_set.y, pos.z-off_set.z);
    int i = int(floorf(samplepos.x/h));
    int j = int(floorf(samplepos.y/h));
    int k = int(floorf(samplepos.z/h));
    float fx = samplepos.x/h - float(i);
    float fy = samplepos.y/h - float(j);
    float fz = samplepos.z/h - float(k);

    int idx000 = i + nx*j + nx*ny*k;
    int idx001 = i + nx*j + nx*ny*k + 1;
    int idx010 = i + nx*j + nx*ny*k + nx;
    int idx011 = i + nx*j + nx*ny*k + nx + 1;
    int idx100 = i + nx*j + nx*ny*k + nx*ny;
    int idx101 = i + nx*j + nx*ny*k + nx*ny + 1;
    int idx110 = i + nx*j + nx*ny*k + nx*ny + nx;
    int idx111 = i + nx*j + nx*ny*k + nx*ny + nx + 1;
    return triLerp(b[idx000], b[idx001],b[idx010],b[idx011],b[idx100],b[idx101],b[idx110],b[idx111], fx, fy, fz);
}

__device__ float3 getVelocity(float *u, float *v, float *w, float h, float nx, float ny, float nz, float3 pos)
{

    float _u = sample_buffer(u, nx+1, ny, nz, h, make_float3(-0.5*h,0,0), pos);
    float _v = sample_buffer(v, nx, ny+1, nz, h, make_float3(0,-0.5*h,0), pos);
    float _w = sample_buffer(w, nx, ny, nz+1, h, make_float3(0,0,-0.5*h), pos);

    return make_float3(_u,_v,_w);
}

__device__ float3 traceRK3(float *u, float *v, float *w, float h, int ni, int nj, int nk, float dt, float3 pos)
{
    float c1 = 2.0/9.0*dt, c2 = 3.0/9.0 * dt, c3 = 4.0/9.0 * dt;
    float3 input = pos;
    float3 v1 = getVelocity(u,v,w,h,ni,nj,nk, input);
    float3 midp1 = make_float3(input.x + 0.5*dt*v1.x, input.y + 0.5*dt*v1.y, input.z + 0.5*dt*v1.z);
    float3 v2 = getVelocity(u,v,w,h,ni,nj,nk, midp1);
    float3 midp2 = make_float3(input.x + 0.75*dt*v2.x, input.y + 0.75*dt*v2.y, input.z + 0.75*dt*v2.z);
    float3 v3 = getVelocity(u,v,w,h,ni,nj,nk, midp2);

    float3 output = make_float3(input.x + c1*v1.x + c2*v2.x + c3*v3.x,
                                input.y + c1*v1.y + c2*v2.y + c3*v3.y,
                                input.z + c1*v1.z + c2*v2.z + c3*v3.z);
    output = clampv3(output, make_float3(h,h,h),
            make_float3(float(ni) * h - h, float(nj) * h - h, float(nk) * h - h ));
    return output;
}

__device__ float3 trace(float *u, float *v, float *w, float h, int ni, int nj, int nk, float cfldt, float dt, float3 pos)
{
    if(dt>0)
    {
        float T = dt;
        float3 opos = pos;
        float t = 0;
        float substep = cfldt;
        while(t<T)
        {
            if(t+substep > T)
                substep = T - t;
            opos = traceRK3(u,v,w,h,ni,nj,nk,substep,opos);

            t+=substep;
        }
        return opos;
    }
    else
    {
        float T = -dt;
        float3 opos = pos;
        float t = 0;
        float substep = cfldt;
        while(t<T)
        {
            if(t+substep > T)
                substep = T - t;
            opos = traceRK3(u,v,w,h,ni,nj,nk,-substep,opos);
            t+=substep;
        }
        return opos;
    }
}

__global__ void forward_kernel(float *u, float *v, float *w,
                            float *x_fwd, float *y_fwd, float *z_fwd,
                            float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 point = make_float3(x_fwd[index], y_fwd[index], z_fwd[index]);
        float3 pointout = trace(u,v,w,h,ni,nj,nk,cfldt,dt,point);
        x_fwd[index] = pointout.x;
        y_fwd[index] = pointout.y;
        z_fwd[index] = pointout.z;
    }
    __syncthreads();
}

__global__ void clampExtrema_kernel(float *before, float *after, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    float max_value = before[index];
    float min_value = before[index];
    if (i>0 && i<ni-1 && j>0 && j<nj-1 && k>0 && k<nk-1)
    {
        for(int kk=k-1;kk<=k+1;kk++)for(int jj=j-1;jj<=j+1;jj++)for(int ii=i-1;ii<=i+1;ii++)
        {
            int idx = ii + jj*ni + kk*ni*nj;
            if(before[idx]>max_value)
                max_value = before[idx];
            if(before[idx]<min_value)
                min_value = before[idx];
        }
        after[index] = min(max(min_value, after[index]), max_value);
    }
    __syncthreads();
}

__global__ void DMC_backward_kernel(float *u, float *v, float *w,
                                    float *x_in, float *y_in, float *z_in,
                                    float *x_out, float *y_out, float *z_out,
                                    float h, int ni, int nj, int nk, float substep)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 point = make_float3(h*float(i),h*float(j),h*float(k));

        float3 vel = getVelocity(u, v, w, h, ni, nj, nk, point);

        float temp_x = (vel.x > 0)? point.x - h: point.x + h;
        float temp_y = (vel.y > 0)? point.y - h: point.y + h;
        float temp_z = (vel.z > 0)? point.z - h: point.z + h;
        float3 temp_point = make_float3(temp_x, temp_y, temp_z);
        float3 temp_vel = getVelocity(u, v, w, h, ni, nj, nk, temp_point);

        float a_x = (vel.x - temp_vel.x) / (point.x - temp_point.x);
        float a_y = (vel.y - temp_vel.y) / (point.y - temp_point.y);
        float a_z = (vel.z - temp_vel.z) / (point.z - temp_point.z);

        float new_x = (fabs(a_x) > 1e-4)? point.x - (1 - exp(-a_x*substep))*vel.x/a_x : point.x - vel.x*substep;
        float new_y = (fabs(a_y) > 1e-4)? point.y - (1 - exp(-a_y*substep))*vel.y/a_y : point.y - vel.y*substep;
        float new_z = (fabs(a_z) > 1e-4)? point.z - (1 - exp(-a_z*substep))*vel.z/a_z : point.z - vel.z*substep;
        float3 pointnew = make_float3(new_x, new_y, new_z);

        x_out[index] = sample_buffer(x_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
        y_out[index] = sample_buffer(y_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
        z_out[index] = sample_buffer(z_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
    }
    __syncthreads();
}

__global__ void semilag_kernel(float *field, float *field_src,
                               float *u, float *v, float *w,
                               int dim_x, int dim_y, int dim_z,
                               float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float3 buffer_origin = make_float3(-float(dim_x)*0.5f*h, -float(dim_y)*0.5f*h, -float(dim_z)*0.5f*h);

    int field_buffer_i = ni + dim_x;
    int field_buffer_j = nj + dim_y;
    int field_buffer_k = nk + dim_z;

    int i = index % field_buffer_i;
    int j = (index % (field_buffer_i * field_buffer_j)) / field_buffer_i;
    int k = index/(field_buffer_i*field_buffer_j);

    if (i > 1 && i < field_buffer_i-2-dim_x && j > 1 && j < field_buffer_j-2-dim_y && k > 1 && k < field_buffer_k-2-dim_z)
    {
        float3 point = make_float3(h*float(i) + buffer_origin.x,
                                   h*float(j) + buffer_origin.y,
                                   h*float(k) + buffer_origin.z);

        float3 pointnew = trace(u, v, w, h, ni, nj, nk, cfldt, dt, point);

        field[index] = sample_buffer(field_src, field_buffer_i, field_buffer_j, field_buffer_k, h, buffer_origin, pointnew);
    }
    __syncthreads();
}


__global__ void doubleAdvect_kernel(float *field, float *temp_field,
                                    float *backward_x, float *backward_y, float * backward_z,
                                    float *backward_xprev, float *backward_yprev, float *backward_zprev,
                                    float h, int ni, int nj, int nk,
                                    int dimx, int dimy, int dimz, bool is_point, float blend_coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);


    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }


    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (2+dimx<i && i<vel_buffer_i-3 && 2+dimy< j && j<vel_buffer_j-3 && 2+dimz<k && k<vel_buffer_k-3)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                     float(j)*h + buffer_origin.y + volume[ii].y,
                                     float(k)*h + buffer_origin.z + volume[ii].z);
            float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
            float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
            float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

            float3 midpos = make_float3(x_init, y_init, z_init);
            midpos = clampv3(midpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) -h ));
            float x_orig = sample_buffer(backward_xprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float y_orig = sample_buffer(backward_yprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float z_orig = sample_buffer(backward_zprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float3 finalpos = make_float3(x_orig, y_orig, z_orig);

            finalpos = clampv3(finalpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
            sum += weight*sample_buffer(temp_field, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, finalpos);
        }
        float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                 float(j)*h + buffer_origin.y,
                                 float(k)*h + buffer_origin.z);
        float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
        float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
        float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

        float3 midpos = make_float3(x_init, y_init, z_init);
        midpos = clampv3(midpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) -h ));
        float x_orig = sample_buffer(backward_xprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float y_orig = sample_buffer(backward_yprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float z_orig = sample_buffer(backward_zprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float3 finalpos = make_float3(x_orig, y_orig, z_orig);

        finalpos = clampv3(finalpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
        float value = sample_buffer(temp_field, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, finalpos);
        float prev_value = 0.5f*(sum + value);
        field[index] = field[index]*blend_coeff + (1-blend_coeff)*prev_value;
    }
    __syncthreads();
}

__global__ void advect_kernel(float *field, float *field_init,
                              float *backward_x, float *backward_y, float *backward_z,
                              float h, int ni, int nj, int nk,
                              int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (2+dimx<i && i<vel_buffer_i-3 && 2+dimy< j && j<vel_buffer_j-3 && 2+dimz<k && k<vel_buffer_k-3)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                     float(j)*h + buffer_origin.y + volume[ii].y,
                                     float(k)*h + buffer_origin.z + volume[ii].z);

            float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
            float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
            float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

            float3 pos_init = make_float3(x_init, y_init, z_init);

            pos_init = clampv3(pos_init, make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
            sum += weight*sample_buffer(field_init, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, pos_init);
        }
        float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                 float(j)*h + buffer_origin.y,
                                 float(k)*h + buffer_origin.z);

        float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
        float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
        float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

        float3 pos_init = make_float3(x_init, y_init, z_init);

        pos_init = clampv3(pos_init, make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
        float value = sample_buffer(field_init, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, pos_init);
        field[index] = 0.5f*sum + 0.5f*value;
    }
    __syncthreads();
}

__global__ void cumulate_kernel(float *dfield, float *dfield_init,
                                float *x_map, float *y_map, float *z_map,
                                float h, int ni, int nj, int nk,
                                int dimx, int dimy, int dimz, bool is_point, float coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (1+dimx<i && i<vel_buffer_i-2 && 1+dimy< j && j<vel_buffer_j-2 && 1+dimz<k && k<vel_buffer_k-2)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                       float(j)*h + buffer_origin.y + volume[ii].y,
                                       float(k)*h + buffer_origin.z + volume[ii].z);
            // forward mapping position
            // also used in compensation
            float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float3 map_pos = make_float3(x_pos, y_pos, z_pos);
            map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
            sum += weight * coeff * sample_buffer(dfield, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        }
        float3 point = make_float3(float(i)*h + buffer_origin.x,
                                   float(j)*h + buffer_origin.y,
                                   float(k)*h + buffer_origin.z);
        // forward mapping position
        float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 map_pos = make_float3(x_pos, y_pos, z_pos);
        map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
        float value = coeff * sample_buffer(dfield, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        sum = 0.5*sum + 0.5 * value;
        dfield_init[index] += sum;
    }
    __syncthreads();
}

__global__ void compensate_kernel(float *src_buffer, float *temp_buffer, float *test_buffer,
                                  float *x_map, float *y_map, float *z_map,
                                  float h, int ni, int nj, int nk,
                                  int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (1+dimx<i && i<vel_buffer_i-2 && 1+dimy< j && j<vel_buffer_j-2 && 1+dimz<k && k<vel_buffer_k-2)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                       float(j)*h + buffer_origin.y + volume[ii].y,
                                       float(k)*h + buffer_origin.z + volume[ii].z);
            float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float3 map_pos = make_float3(x_pos, y_pos, z_pos);
            map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
            sum += weight * sample_buffer(src_buffer, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        }
        float3 point = make_float3(float(i)*h + buffer_origin.x,
                                   float(j)*h + buffer_origin.y,
                                   float(k)*h + buffer_origin.z);
        // forward mapping position
        float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 map_pos = make_float3(x_pos, y_pos, z_pos);
        map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
        float value = sample_buffer(src_buffer, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        sum = 0.5*sum + 0.5*value;
        test_buffer[index] = sum - temp_buffer[index];
//        sum -= temp_buffer[index];
//        sum *= 0.5f;
//        temp_buffer[index] = sum;
    }
    __syncthreads();
}

__global__ void estimate_kernel(float *dist_buffer, float *x_first, float *y_first, float *z_first,
                                float *x_second, float *y_second, float *z_second,
                                float h, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>1 && i<ni-2 && j>1 && j<nj-2 && k>1 && k<nk-2)
    {
        float3 point = make_float3(h*float(i),h*float(j),h*float(k));
        // backward then forward
        float back_x = sample_buffer(x_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float back_y = sample_buffer(y_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float back_z = sample_buffer(z_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 back_pos = make_float3(back_x, back_y, back_z);
        float fwd_x = sample_buffer(x_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),back_pos);
        float fwd_y = sample_buffer(y_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),back_pos);
        float fwd_z = sample_buffer(z_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),back_pos);
        float dist_bf = (point.x-fwd_x)*(point.x-fwd_x) +
                        (point.y-fwd_y)*(point.y-fwd_y) +
                        (point.z-fwd_z)*(point.z-fwd_z);
        // forward then backward
        fwd_x = sample_buffer(x_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        fwd_y = sample_buffer(y_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        fwd_z = sample_buffer(z_second,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 fwd_pos = make_float3(fwd_x, fwd_y, fwd_z);
        back_x = sample_buffer(x_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),fwd_pos);
        back_y = sample_buffer(y_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),fwd_pos);
        back_z = sample_buffer(z_first,ni,nj,nk,h,make_float3(0.0,0.0,0.0),fwd_pos);
        float dist_fb = (point.x-back_x)*(point.x-back_x) +
                        (point.y-back_y)*(point.y-back_y) +
                        (point.z-back_z)*(point.z-back_z);
        dist_buffer[index] = max(dist_bf, dist_fb);
    }
    __syncthreads();
}

__global__ void reduce0(float *g_idata, float *g_odata, int N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;

    sdata[tid] = (i<N)?g_idata[i]:0;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s && i < N)
        {
            sdata[tid] = max(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void add_kernel(float *field1, float *field2, float coeff)
{
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
    field1[i] += coeff*field2[i];
    __syncthreads();
}

extern "C" void gpu_solve_forward(float *u, float *v, float *w,
                                  float *x_fwd, float *y_fwd, float *z_fwd,
                                  float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    forward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_fwd, y_fwd, z_fwd, h, ni, nj, nk, cfldt, dt);
}

extern "C" void gpu_solve_backwardDMC(float *u, float *v, float *w,
                                      float *x_in, float *y_in, float *z_in,
                                      float *x_out, float *y_out, float *z_out,
                                      float h, int ni, int nj, int nk, float substep)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    DMC_backward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_in, y_in, z_in, x_out, y_out, z_out, h, ni, nj, nk, substep);
}

extern "C" void gpu_advect_velocity(float *u, float *v, float *w,
                                    float *u_init, float *v_init, float *w_init,
                                    float *backward_x, float *backward_y, float *backward_z,
                                    float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    advect_kernel<<< numBlocks_u, blocksize >>>(u, u_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 1, 0, 0, is_point);
    advect_kernel<<< numBlocks_v, blocksize >>>(v, v_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 1, 0, is_point);
    advect_kernel<<< numBlocks_w, blocksize >>>(w, w_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 1, is_point);
}

extern "C" void gpu_advect_vel_double(float *u, float *v, float *w,
                                      float *utemp, float *vtemp, float *wtemp,
                                      float *backward_x, float *backward_y, float *backward_z,
                                      float *backward_xprev,  float *backward_yprev,  float *backward_zprev,
                                      float h, int ni, int nj, int nk, bool is_point, float blend_coeff)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    doubleAdvect_kernel<<< numBlocks_u, blocksize >>> (u, utemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 1, 0, 0, is_point, blend_coeff);

    doubleAdvect_kernel<<< numBlocks_v, blocksize >>> (v, vtemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 1, 0, is_point, blend_coeff);

    doubleAdvect_kernel<<< numBlocks_w, blocksize >>> (w, wtemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 0, 1, is_point, blend_coeff);
}

extern "C" void gpu_advect_field(float *field, float *field_init,
                                 float *backward_x, float *backward_y, float *backward_z,
                                 float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    advect_kernel<<< numBlocks, blocksize >>>(field, field_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point);
}

extern "C" void gpu_advect_field_double(float *field, float *field_prev,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float *backward_xprev, float *backward_yprev,   float *backward_zprev,
                                        float h, int ni, int nj, int nk, bool is_point, float blend_coeff)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    doubleAdvect_kernel<<< numBlocks, blocksize >>> (field, field_prev, backward_x, backward_y, backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 0, 0, is_point, blend_coeff);
}

extern "C" void gpu_compensate_velocity(float *u, float *v, float *w,
                                        float *du, float *dv, float *dw,
                                        float *u_src, float *v_src, float *w_src,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    // error at time 0 will be in du, dv, dw
    compensate_kernel<<< numBlocks_u, blocksize>>>(u, du, u_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 1, 0, 0, is_point);
    compensate_kernel<<< numBlocks_v, blocksize>>>(v, dv, v_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 1, 0, is_point);
    compensate_kernel<<< numBlocks_w, blocksize>>>(w, dw, w_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 1, is_point);
    // now subtract error at time t, compensated velocity will be stored in gpu.u, gpu.v, gpu.w
    cudaMemcpy(du, u, sizeof(float)*(ni+1)*nj*nk, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dv, v, sizeof(float)*ni*(nj+1)*nk, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dw, w, sizeof(float)*ni*nj*(nk+1), cudaMemcpyDeviceToDevice);
    cumulate_kernel<<< numBlocks_u, blocksize >>>(u_src, u, backward_x, backward_y, backward_z, h, ni, nj, nk, 1, 0, 0, is_point, -0.5f);
    cumulate_kernel<<< numBlocks_v, blocksize >>>(v_src, v, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 1, 0, is_point, -0.5f);
    cumulate_kernel<<< numBlocks_w, blocksize >>>(w_src, w, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 1, is_point, -0.5f);
    // clamp extrema, clamped result will be in gpu.u, gpu.v, gpu.w
    clampExtrema_kernel<<< numBlocks_u, blocksize >>>(du, u, ni+1, nj, nk);
    clampExtrema_kernel<<< numBlocks_v, blocksize >>>(dv, v, ni, nj+1, nk);
    clampExtrema_kernel<<< numBlocks_w, blocksize >>>(dw, w, ni, nj, nk+1);
}

extern "C" void gpu_compensate_field(float *u, float *du, float *u_src,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float *backward_x, float *backward_y, float *backward_z,
                                     float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    // error at time 0 will be in du
    compensate_kernel<<< numBlocks_u, blocksize>>>(u, du, u_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point);
    // now subtract error at time t, compensated velocity will be stored in gpu.u
    cudaMemcpy(du, u, sizeof(float)*(ni+1)*nj*nk, cudaMemcpyDeviceToDevice);
    cumulate_kernel<<< numBlocks_u, blocksize >>>(u_src, u, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point, -0.5f);
    // clamp extrema, clamped result will be in gpu.u
    clampExtrema_kernel<<< numBlocks_u, blocksize >>>(du, u, ni, nj, nk);
}

extern "C" void gpu_accumulate_velocity(float *u_change, float *v_change, float *w_change,
                                        float *du_init, float *dv_init, float *dw_init,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float h, int ni, int nj, int nk, bool is_point, float coeff)
{
    int blocksize = 256;
    int numBlocks_u = ((ni+1)*nj*nk + 255)/256;
    int numBlocks_v = (ni*(nj+1)*nk + 255)/256;
    int numBlocks_w = (ni*nj*(nk+1) + 255)/256;
    cumulate_kernel<<< numBlocks_u, blocksize >>> (u_change, du_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 1, 0, 0, is_point, coeff);
    cumulate_kernel<<< numBlocks_v, blocksize >>> (v_change, dv_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 1, 0, is_point, coeff);
    cumulate_kernel<<< numBlocks_w, blocksize >>> (w_change, dw_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 1, is_point, coeff);
}

extern "C" void gpu_accumulate_field(float *field_change, float *dfield_init,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float h, int ni, int nj, int nk, bool is_point, float coeff)
{
    int blocksize = 256;
    int numBlocks = ((ni*nj*nk) + 255)/256;
    cumulate_kernel<<< numBlocks, blocksize >>> (field_change, dfield_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point, coeff);
}

extern "C" void gpu_estimate_distortion(float *du,
                                        float *x_back, float *y_back, float *z_back,
                                        float *x_fwd, float *y_fwd, float *z_fwd,
                                        float h, int ni, int nj, int nk)
{
    int blocksize = 256;
    int est_numBlocks = ((ni*nj*nk) + 255)/256;
    // distortion will be stored in gpu.du
    estimate_kernel<<< est_numBlocks, blocksize>>> (du, x_back, y_back, z_back, x_fwd, y_fwd, z_fwd, h, ni, nj, nk);
}

extern "C" void gpu_semilag(float *field, float *field_src,
                            float *u, float *v, float *w,
                            int dim_x, int dim_y, int dim_z,
                            float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int blocksize = 256;
    int total_num = (ni+dim_x)*(nj+dim_y)*(nk+dim_z);
    int numBlocks = (total_num + 255)/256;
    semilag_kernel<<<numBlocks, blocksize>>>(field, field_src, u, v, w, dim_x, dim_y, dim_z, h, ni, nj, nk, cfldt, dt);
}

extern "C" void gpu_add(float *field1, float *field2, float coeff, int number)
{
    int blocksize = 256;
    int numBlocks = (number + 255)/256;
    add_kernel<<<numBlocks, blocksize>>>(field1, field2, coeff);
}