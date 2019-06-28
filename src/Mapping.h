#ifndef BIMOCQ_MAPPING_H
#define BIMOCQ_MAPPING_H

#include <iostream>
#include <cstdint>
#include "tbb/tbb.h"
#include "color_macro.h"
#include "fluid_buffer3D.h"
#include "GPU_Advection.h"

// two level BIMOCQ advector
class MapperBase
{
public:
    MapperBase() = default;
    virtual ~MapperBase() = default;

    virtual void init(uint ni, uint nj, uint nk, float h, float coeff, gpuMapper *mymapper);
    virtual void updateForward(float cfldt, float dt);
    virtual void updateBackward(float cfldt, float dt);
    virtual void updateMapping(const buffer3Df &un, const buffer3Df &vn, const buffer3Df &wn, float cfldt, float dt);

    virtual void accumulateVelocity(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                    const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                    float coeff);
    virtual void accumulateField(buffer3Df &field_init, const buffer3Df &field_change);
    virtual float estimateDistortion(const buffer3Dc &boundary);

    virtual void reinitializeMapping();
    virtual void advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev);
    virtual void advectField(buffer3Df &field, const buffer3Df &field_init, const buffer3Df &field_prev);

    float _h;
    // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
    float blend_coeff;
    uint total_reinit_count;
    uint _ni, _nj, _nk;
    buffer3Df forward_x, forward_y, forward_z;
    buffer3Df backward_x, backward_y, backward_z;
    buffer3Df backward_xprev, backward_yprev, backward_zprev;
    /// gpu solver
    gpuMapper *gpuSolver;
};

#endif //BIMOCQ_MAPPING_H
