#include "BimocqSolver2D.h"
#define ziyin false

inline Vec2f BimocqSolver2D::traceRK3(float dt, Vec2f &pos)
{

	float c1 = 2.0 / 9.0*dt, c2 = 3.0 / 9.0 * dt, c3 = 4.0 / 9.0 * dt;
	Vec2f input = pos;
	Vec2f velocity1 = getVelocity(input);
	Vec2f midp1 = input + ((float)(0.5*dt))*velocity1;
	Vec2f velocity2 = getVelocity(midp1);
	Vec2f midp2 = input + ((float)(0.75*dt))*velocity2;
	Vec2f velocity3 = getVelocity(midp2);
	//velocity = get_velocity(input + 0.5f*dt*velocity);
	input = input + c1*velocity1 + c2*velocity2 + c3*velocity3;
	input[0] = min(max(0.001f*h, input[0]),(float)ni*h-0.001f*h);
	input[1] = min(max(0.001f*h, input[1]),(float)nj*h-0.001f*h);
	return input;
}

inline Vec2f BimocqSolver2D::solveODE(float dt, Vec2f &pos)
{
    float ddt = dt;
    Vec2f pos1 = traceRK3(ddt, pos);
    ddt/=2.0;
    int substeps = 2;
    Vec2f pos2 = traceRK3(ddt, pos);pos2 = traceRK3(ddt, pos2);
    int iter = 0;

    while(dist(pos2,pos1)>0.0001*h && iter<6)
    {
        pos1 = pos2;
        ddt/=2.0;
        substeps *= 2;
        pos2 = pos;
        for(int j=0;j<substeps;j++)
        {
            pos2 = traceRK3(ddt, pos2);
        }
        iter++;
    }
    return pos2;
}

inline Vec2f BimocqSolver2D::solveODEDMC(float dt, Vec2f &pos)
{
    Vec2f a = calculateA(pos, h);
    Vec2f opos=pos;
    float T=dt;
    float t = 0;
    float substep = _cfl;
//    while(t < T)
//    {
//        if(t + substep > T)
//            substep = T - t;
//        opos = traceDMC(-substep, opos, a);
//        a = calculateA(opos, h);
//        t+=substep;
//    }
    opos = traceDMC(dt, opos, a);
    return opos;
}

void BimocqSolver2D::getCFL()
{
    _cfl = h / fabs(maxVel());
}

inline Vec2f BimocqSolver2D::traceDMC(float dt, Vec2f &pos, Vec2f &a)
{
    Vec2f vel = getVelocity(pos);
    float new_x = pos[0] - dt*vel[0];
    float new_y = pos[1] - dt*vel[1];
    if (fabs(a[0]) >1e-4) new_x = pos[0] - (1-exp(-a[0]*dt))*vel[0]/(a[0]);
    else new_x = solveODE(dt, pos)[0];

    if (fabs(a[1]) >1e-4) new_y = pos[1] - (1-exp(-a[1]*dt))*vel[1]/(a[1]);
    else new_y = solveODE(dt, pos)[1];
    return Vec2f(new_x, new_y);
}

inline float BimocqSolver2D::lerp(float v0, float v1, float c)
{
    return (1-c)*v0+c*v1;
}

inline float BimocqSolver2D::bilerp(float v00, float v01, float v10, float v11, float cx, float cy)
{
    return lerp(lerp(v00,v01,cx), lerp(v10,v11,cx),cy);
}

Vec2f BimocqSolver2D::calculateA(Vec2f pos, float h)
{
    Vec2f vel = getVelocity(pos);
    float new_x = (vel[0] > 0)? pos[0]-h : pos[0]+h;
    float new_y = (vel[1] > 0)? pos[1]-h : pos[1]+h;
    Vec2f new_pos = Vec2f(new_x, new_y);
    Vec2f new_vel = getVelocity(new_pos);
    float a_x = (vel[0] - new_vel[0]) / (pos[0] - new_pos[0]);
    float a_y = (vel[1] - new_vel[1]) / (pos[1] - new_pos[1]);
    return Vec2f(a_x, a_y);
}

void BimocqSolver2D::semiLagAdvectDMC(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y)
{
    tbb::parallel_for((int)0,
                      (int)(ni*nj),
                      (int)1,
                      [&](int tId)
                      {
                          int j = tId / ni;
                          int i = tId % ni;
                          Vec2f pos = h*Vec2f(i, j) + h*Vec2f(off_x, off_y);
                          Vec2f back_pos = solveODEDMC(dt, pos);
                          clampPos(back_pos);
                          dst(i, j) = sampleField(back_pos - h*Vec2f(off_x, off_y), src);
                      });
}


void BimocqSolver2D::semiLagAdvect(const Array2f & src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y)
{
    tbb::parallel_for((int)0,
                      (int)(ni*nj),
                      (int)1,
                      [&](int tId)
                      {
                          int j = tId / ni;
                          int i = tId % ni;
                          Vec2f pos = h*Vec2f(i, j) + h*Vec2f(off_x, off_y);
						  Vec2f back_pos = solveODE(-dt, pos);
                          dst(i, j) = sampleField(back_pos - h*Vec2f(off_x, off_y), src);
                      });
}

void BimocqSolver2D::advance(float dt, int frame)
{
    switch(sim_scheme)
    {
        case SEMILAG:
            advanceSemilag(dt, frame);
            break;
        case MACCORMACK:
            advanceMaccormack(dt, frame);
            break;
        case BFECC:
            advanceBFECC(dt, frame);
            break;
        case MAC_REFLECTION:
            advanceReflection(dt, frame);
            break;
        case FLIP:
            advanceFLIP(dt, frame);
            break;
        case APIC:
            advancePolyPIC(dt, frame);
            break;
        case POLYPIC:
            advancePolyPIC(dt, frame);
            break;
        case BIMOCQ:
            advanceBIMOCQ(dt, frame);
            break;
    }
}

BimocqSolver2D::BimocqSolver2D(int nx, int ny, float L, float b_coeff, int N, bool bc, Scheme s_scheme)
{
    blend_coeff = b_coeff;
    use_neumann_boundary = bc;
    h = L / (float)nx;
    ni = nx;
    nj = ny;
    u.resize(nx + 1, ny);
    v.resize(nx, ny + 1);
    u_temp.resize(nx + 1, ny);
    v_temp.resize(nx, ny + 1);
    rho.resize(nx, ny);
    temperature.resize(nx, ny);
    s_temp.resize(nx, ny);
    pressure.resize(nx*ny);
    rhs.resize(nx*ny);
    emitterMask.resize(nx, ny);
    boundaryMask.resize(nx,ny);
    boundaryMask.assign(0.0);
    curl.resize(nx+1, ny+1);

    // for maccormack
    u_first.resize(ni+1, nj);
    v_first.resize(ni, nj+1);
    u_sec.resize(ni+1, nj);
    v_sec.resize(ni, nj+1);

    // init BIMOCQ
    du.resize(nx+1, ny);
    du_temp.resize(nx+1, ny);
    du_proj.resize(nx+1, ny);
    dv.resize(nx, ny+1);
    dv_temp.resize(nx, ny+1);
    dv_proj.resize(nx, ny+1);
    drho.resize(nx, ny);
    drho.assign(nx, ny, 0.0);
    drho_temp.resize(nx, ny);
    drho_prev.resize(nx, ny);
    drho_prev.assign(nx, ny, 0.0);

    dT.resize(nx, ny);
    dT_temp.resize(nx, ny);
    dT_prev.resize(nx, ny);
    dT_prev.assign(nx, ny, 0.0);

    du.assign(ni+1, nj, 0.0);
    u.assign(ni+1, nj, 0.0);
    dv.assign(ni, nj+1, 0.0);
    v.assign(ni, nj+1, 0.0);
    drho.assign(ni, nj, 0.0);
    dT.assign(ni, nj, 0.0);

    u_init.resize(nx+1, ny);
    u_init.assign(nx+1, ny, 0.0);
    v_init.resize(nx, ny+1);
    v_init.assign(nx, ny+1, 0.0);
    rho_init.resize(nx, ny);
    rho_init.assign(nx, ny, 0.0);
    rho_orig.resize(nx, ny);
    rho_orig.assign(nx, ny, 0.0);
    T_init.resize(nx, ny);
    T_init.assign(nx, ny, 0.0);
    T_orig.resize(nx, ny);
    T_orig.assign(nx, ny, 0.0);
    du_temp.assign(0);
    dv_temp.assign(0);

    u_origin.resize(ni+1,nj);
    u_origin.assign(0);

    v_origin.resize(ni,nj+1);
    v_origin.assign(0);

    du_prev.resize(ni+1,nj);
    du_prev.assign(0);

    dv_prev.resize(ni,nj+1);
    dv_prev.assign(0);

    // mapping buffers
    forward_x.resize(nx, ny);
    forward_y.resize(nx, ny);
    backward_x.resize(nx, ny);
    backward_y.resize(nx, ny);
    forward_scalar_y.resize(nx, ny);
    forward_scalar_x.resize(nx, ny);
    backward_scalar_x.resize(nx, ny);
    backward_scalar_y.resize(nx, ny);
    backward_xprev.resize(nx, ny);
    backward_yprev.resize(nx, ny);
    backward_scalar_xprev.resize(nx, ny);
    backward_scalar_yprev.resize(nx, ny);

    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        forward_x(i, j) = h*((float)i + 0.5);
        backward_x(i, j) = h*((float)i + 0.5);
        forward_y(i, j) = h*((float)j + 0.5);
        backward_y(i, j) = h*((float)j + 0.5);
    });
    backward_scalar_x = backward_x;
    backward_scalar_y = backward_y;
    forward_scalar_x = forward_x;
    forward_scalar_y = forward_y;
    backward_xprev = backward_x;
    backward_yprev = backward_y;
    backward_scalar_xprev = backward_x;
    backward_scalar_yprev = backward_y;

    // for particle methods
    seedParticles(N);

    sim_scheme = s_scheme;
}

void BimocqSolver2D::solveMaccormack(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety)
{
    semiLagAdvect(src, dst, dt, ni, nj, offsetx, offsety);
    semiLagAdvect(dst, aux, -dt, ni, nj, offsetx, offsety);
    // clamp extrema
    tbb::parallel_for((int)0, (dst.ni)*dst.nj, 1, [&](int tIdx) {
        int i = tIdx%(dst.ni);
        int j = tIdx / (dst.ni);
        Vec2f pos = h*(Vec2f(i,j) + Vec2f(offsetx, offsety));
        dst(i, j) = dst(i,j) + 0.5*(src(i,j) - aux(i,j));
        Vec2f newpos = solveODE(-dt, pos);
        float v00,v01,v10,v11;
        int ii,jj;
        ii = floor((newpos[0]-offsetx*h)/h);
        jj = floor((newpos[1]-offsety*h)/h);
        v00 = src.boundedAt(ii,jj);
        v01 = src.boundedAt(ii+1, jj);
        v10 = src.boundedAt(ii, jj+1);
        v11 = src.boundedAt(ii+1, jj+1);
        float minVal = std::min(std::min(std::min(v00,v01),v10),v11);
        float maxVal = std::max(std::max(std::max(v00,v01),v10),v11);
        if(dst(i,j)<minVal||dst(i,j)>maxVal)
            dst(i,j) = sampleField(newpos-h*Vec2f(offsetx,offsety),src);
    });
}

void BimocqSolver2D::solveBFECC(const Array2f &src, Array2f &dst, Array2f &aux, float dt, int ni, int nj, float offsetx, float offsety)
{
    semiLagAdvect(src, dst, dt, ni, nj, offsetx, offsety);
    semiLagAdvect(dst, aux, -dt, ni, nj, offsetx, offsety);
    tbb::parallel_for((int)0, (dst.ni)*dst.nj, 1, [&](int tIdx) {
        int i = tIdx%(dst.ni);
        int j = tIdx / (dst.ni);
        dst(i, j) = 0.5*(3*src(i,j) - aux(i,j));
    });
    semiLagAdvect(dst, aux, dt, ni, nj, offsetx, offsety);
    dst = aux;
    // clamp extrema
    tbb::parallel_for((int)0, (dst.ni)*dst.nj, 1, [&](int tIdx) {
        int i = tIdx%(dst.ni);
        int j = tIdx / (dst.ni);
        Vec2f pos = h*(Vec2f(i,j) + Vec2f(offsetx, offsety));
        Vec2f newpos = solveODE(-dt, pos);
        float v00,v01,v10,v11;
        int ii,jj;
        ii = floor((newpos[0]-offsetx*h)/h);
        jj = floor((newpos[1]-offsety*h)/h);
        v00 = src.boundedAt(ii,jj);
        v01 = src.boundedAt(ii+1, jj);
        v10 = src.boundedAt(ii, jj+1);
        v11 = src.boundedAt(ii+1, jj+1);
        float minVal = std::min(std::min(std::min(v00,v01),v10),v11);
        float maxVal = std::max(std::max(std::max(v00,v01),v10),v11);
        if(dst(i,j)<minVal||dst(i,j)>maxVal)
            dst(i,j) = sampleField(newpos-h*Vec2f(offsetx,offsety),src);
    });
}

void BimocqSolver2D::applyBuoyancyForce(float dt)
{
    /// NOTE: this function is used for Rayleigh-Taylor example, where rho and temperature represent two kinds of fluid
    /// with different density, so both rho and temperature act like drop force
    /// for smoke, you may want temperature acts like rising force, which change the - beta*temperature(i,j) to be + beta*temperature(i,j)
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        v(i, j) += 0.5*dt*(-alpha*rho(i,j) - beta*temperature(i, j));
    });
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        v(i, j + 1) += 0.5*dt*(-alpha*rho(i, j) - beta*temperature(i, j));
    });
}

void BimocqSolver2D::projection(float tol, bool PURE_NEUMANN)
{
    applyVelocityBoundary();

    rhs.assign(ni*nj, 0.0);
    pressure.assign(ni*nj, 0.0);
    //build rhs;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        //rhs[tIdx] = 0;
        //if(boundaryMask(i,j)==0)
        rhs[tIdx] = -(u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)) / h;
    });
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(matrix_fix,rhs,pressure,A_L,R_L,P_L,S_L,total_level,(double)tol,500,res_out,iter_out,ni,nj, PURE_NEUMANN);

    if (converged)
        std::cout << "pressure solver converged in " << iter_out << " iterations, with residual " << res_out << std::endl;
    else
        std::cout << "warning! solver didn't reach convergence with maximum iterations!" << std::endl;
    //subtract gradient
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        u(i, j) -= pressure[tIdx] / h;
        v(i, j) -= pressure[tIdx] / h;
    });
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        u(i + 1, j) += pressure[tIdx] / h;
        v(i, j + 1) += pressure[tIdx] / h;
    });
    applyVelocityBoundary();
//    nostickBC();
}

void BimocqSolver2D::advanceBIMOCQ(float dt, int currentframe)
{
    std::cout << BLUE <<  "BIMOCQ scheme frame " << currentframe << " starts !" << RESET << std::endl;

    float proj_coeff = 2.0;
    getCFL();
    if (currentframe != 0)
    {
        u = u_temp;
        v = v_temp;
    }
    frameCount++;
    resampleCount++;

    /// update Backward & Forward mapping
    updateForward(dt, forward_x, forward_y);
    updateForward(dt, forward_scalar_x, forward_scalar_y);
    updateBackward(dt, backward_x, backward_y);
    updateBackward(dt, backward_scalar_x, backward_scalar_y);

    Array2f semi_u, semi_v, semi_rho, semi_T;
    semi_u.resize(ni+1, nj, 0.0);
    semi_v.resize(ni, nj+1, 0.0);
    semi_rho.resize(ni, nj, 0.0);
    semi_T.resize(ni, nj, 0.0);
    semiLagAdvect(rho, semi_rho, dt, ni, nj, 0.5, 0.5);
    semiLagAdvect(temperature, semi_T, dt, ni, nj, 0.5, 0.5);
    semiLagAdvect(u, semi_u, dt, ni+1, nj, 0.0, 0.5);
    semiLagAdvect(v, semi_v, dt, ni, nj+1, 0.5, 0.0);

    Array2f u_presave;
    Array2f v_presave;
    u_presave = u;
    v_presave = v;
    /// advect U,V
    advectVelocity(semi_u, semi_v);
    correctVelocity(semi_u, semi_v);
    /// advect Rho, T
    advectScalars(semi_rho, semi_T);
    correctScalars(semi_rho, semi_T);

    Array2f u_save;
    Array2f v_save;
    Array2f rho_save;
    Array2f T_save;

    u_save = u;
    v_save = v;
    rho_save = rho;
    T_save = temperature;

    applyBuoyancyForce(dt);

    du_temp = u; du_temp -= u_save;
    dv_temp = v; dv_temp -= v_save;
    u_save = u;
    v_save = v;

    projection(1e-6, use_neumann_boundary);
    float d_vel = estimateDistortion(backward_x, backward_y, forward_x, forward_y);
    float d_scalar = estimateDistortion(backward_scalar_x, backward_scalar_y, forward_scalar_x, forward_scalar_y);
    float vel = maxVel();
    std::cout << "Velocity remapping condition:" << d_vel / (vel * dt) << std::endl;
    std::cout << "Scalars remapping condition:" << d_scalar / (vel * dt) << std::endl;

    bool vel_remapping = ((d_vel / (vel * dt))>1.0 ||currentframe-lastremeshing>=8);
    bool rho_remapping = ((d_scalar / (vel * dt))>10.0 ||currentframe-rho_lastremeshing>=50);

    if (vel_remapping)
        proj_coeff = 1.0;

    // calculate the field difference
    du_proj = u; du_proj -= u_save;
    dv_proj = v; dv_proj -= v_save;
    drho_temp = rho; drho_temp -= rho_save;
    dT_temp = temperature; dT_temp -= T_save;

    /// cumulate du, dv
    accumulateVelocity(du_temp, dv_temp, 1.0, false);
    accumulateVelocity(du_proj, dv_proj, proj_coeff, false);
    accumulateScalars(drho_temp, dT_temp, false);

    if (vel_remapping)
    {
        lastremeshing = currentframe;
        resampleVelBuffer(dt);
        accumulateVelocity(du_proj, dv_proj, proj_coeff, false);
    }
    if (rho_remapping)
    {
        rho_lastremeshing = currentframe;
        resampleRhoBuffer(dt);
    }

    u_temp = u;
    v_temp = v;
    if (currentframe != 0)
    {
        tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
            int i = tIdx % (ni + 1);
            int j = tIdx / (ni + 1);
            u(i,j) = 0.5*(u_presave(i,j) + u(i,j));
        });
        tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
            int i = tIdx % ni;
            int j = tIdx / ni;
            v(i,j) = 0.5*(v_presave(i,j) + v(i,j));
        });
    }
}

void BimocqSolver2D::advanceSemilag(float dt, int currentframe)
{
    std::cout << BLUE <<  "Semi-Lagrangian scheme frame " << currentframe << " starts !" << RESET << std::endl;
    // Semi-Lagrangian advect density
    s_temp.assign(ni, nj, 0.0f);
    semiLagAdvect(rho, s_temp, dt, ni, nj, 0.5, 0.5);
    rho.assign(ni, nj, s_temp.a.data);

    // Semi-Lagrangian advect temperature
    s_temp.assign(ni, nj, 0.0f);
    semiLagAdvect(temperature, s_temp, dt, ni, nj, 0.5, 0.5);
    temperature.assign(ni, nj, s_temp.a.data);

    // Semi-Lagrangian advect velocity
    u_temp.assign(ni + 1, nj, 0.0f);
    v_temp.assign(ni, nj + 1, 0.0f);
    semiLagAdvect(u, u_temp, dt, ni + 1, nj, 0.0, 0.5);
    semiLagAdvect(v, v_temp, dt, ni, nj + 1, 0.5, 0.0);
    u.assign(ni + 1, nj, u_temp.a.data);
    v.assign(ni, nj + 1, v_temp.a.data);

    applyBuoyancyForce(dt);
    projection(1e-6,use_neumann_boundary);
}

void BimocqSolver2D::advanceReflection(float dt, int currentframe)
{
    std::cout << BLUE <<  "Reflection scheme frame " << currentframe << " starts !" << RESET << std::endl;

    // advect rho
    Array2f rho_first;
    Array2f rho_sec;
    rho_first.assign(ni, nj, 0.0);
    rho_sec.assign(ni, nj, 0.0);
    solveMaccormack(rho, rho_first, rho_sec, dt, ni, nj, 0.5, 0.5);
    rho = rho_first;

    // advect rho
    Array2f T_first;
    Array2f T_sec;
    T_first.assign(ni, nj, 0.0);
    T_sec.assign(ni, nj, 0.0);
    solveMaccormack(temperature, T_first, T_sec, dt, ni, nj, 0.5, 0.5);
    temperature = T_first;

    Array2f u_save;
    Array2f v_save;
    // step 1
    u_first.assign(ni+1, nj, 0.0);
    v_first.assign(ni, nj+1, 0.0);
    u_sec.assign(ni+1, nj, 0.0);
    v_sec.assign(ni, nj+1, 0.0);
    solveMaccormack(u, u_first, u_sec, 0.5*dt, ni+1, nj, 0.0, 0.5);
    solveMaccormack(v, v_first, v_sec, 0.5*dt, ni, nj+1, 0.5, 0.0);

    u = u_first;
    v = v_first;

    applyBuoyancyForce(0.5f*dt);
    u_save = u;
    v_save = v;
    // step 2
    projection(1e-6, use_neumann_boundary);

    // step 3
    tbb::parallel_for((int)0, (ni+1)*nj, 1, [&](int tIdx) {
        int i = tIdx%(ni+1);
        int j = tIdx / (ni+1);
        u_temp(i, j) = 2.0*u(i,j) - u_save(i,j);
    });
    tbb::parallel_for((int)0, ni*(nj+1), 1, [&](int tIdx) {
        int i = tIdx%ni;
        int j = tIdx / ni;
        v_temp(i, j) = 2.0*v(i,j) - v_save(i,j);
    });

    // step 4
    u_first.assign(ni+1, nj, 0.0);
    v_first.assign(ni, nj+1, 0.0);
    u_sec.assign(ni+1, nj, 0.0);
    v_sec.assign(ni, nj+1, 0.0);
    solveMaccormack(u_temp, u_first, u_sec, 0.5*dt, ni+1, nj, 0.0, 0.5);
    solveMaccormack(v_temp, v_first, v_sec, 0.5*dt, ni, nj+1, 0.5, 0.0);
    u = u_first;
    v = v_first;

    applyBuoyancyForce(0.5f*dt);
    // step 5
    projection(1e-6, use_neumann_boundary);

}

void BimocqSolver2D::nostickBC()
{

    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx/ni;
        if(boundaryMask(i,j) == 1&&i>=1&&i<ni-1&&j>=1&&j<nj-1)
        {
            char bleft = boundaryMask(i-1,j);
            char bright= boundaryMask(i+1,j);
            char bup   = boundaryMask(i,j+1);
            char bdown = boundaryMask(i,j-1);

            char sumb = bleft+bright+bup+bdown;
            if(sumb==1)
            {
                if(bleft==0)
                {
                    v(i,j) = v(i-1,j);
                }
                else if(bright==0)
                {
                    v(i,j) = v(i+1,j);
                }
                else if(bup==0)
                {
                    u(i,j) = u(i,j+1);
                }
                else if(bdown==0)
                {
                    u(i,j) = u(i,j-1);
                }
            }
            else if(sumb==2)
            {
                if(bleft==0&&bup==0)
                {

                }
                else if(bleft==0&&bdown==0)
                {

                }
                else if(bright==0&&bup==0)
                {

                }
                else if(bright==0&&bdown==0)
                {

                }
            }

        }
    }
    );
}

float BimocqSolver2D::estimateDistortion(Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y)
{
    float d = 0;
    for(int tIdx=0;tIdx<ni*nj;tIdx++)
    {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if(i>2&&i<ni-3 && j>2&&j<nj-3) {
            Vec2f init_pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
            Vec2f fpos0 = Vec2f(fwd_x(i, j), fwd_y(i, j));
            Vec2f bpos = Vec2f(sampleField(fpos0 - Vec2f(0.5) * h, back_x),
                               sampleField(fpos0 - Vec2f(0.5) * h, back_y));
            float new_dist = dist(bpos, init_pos);
            if ( new_dist > d) d = new_dist;
        }
    }

    for(int tIdx=0;tIdx<ni*nj;tIdx++)
    {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if(i>2&&i<ni-3 && j>2&&j<nj-3) {
            Vec2f init_pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
            Vec2f bpos0 = Vec2f(back_x(i, j), back_y(i, j));
            Vec2f fpos = Vec2f(sampleField(bpos0 - Vec2f(0.5) * h, fwd_x),
                               sampleField(bpos0 - Vec2f(0.5) * h, fwd_y));
            float new_dist = dist(fpos, init_pos);
            if (new_dist > d) d = new_dist;
        }
    }
    return d;
}

float BimocqSolver2D::maxVel()
{
    float vel=0;
    int idx_i = 0;
    int idx_j = 0;
    for(int i=0;i<u.a.n;i++)
    {
        if (u.a[i] > vel)
        {
            vel = u.a[i];
            idx_i = i / (ni+1);
            idx_j = i % (ni+1);
        }
    }
    for(int j=0;j<v.a.n;j++)
    {
        if (v.a[j] > vel)
        {
            vel = v.a[j];
            idx_i = j / ni;
            idx_j = j % nj;
        }
    }
//    std::cout << "max velocity index: (" << idx_i << ", " << idx_j << ")" << std::endl;
    return vel + 1e-5;

}

void BimocqSolver2D::correctScalars(Array2f &semi_rho, Array2f &semi_T)
{
    Array2f rho_curr = rho;
    Array2f T_curr = temperature;
    Array2f temp_rho;
    Array2f temp_T;
    temp_rho.resize(ni,nj,0.0);
    temp_T.resize(ni,nj,0.0);
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
//        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                temp_rho(i, j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), rho) - drho(i,j));
            }
    });
    temp_rho -= rho_init;
    temp_rho *= 0.5f;
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
//        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                rho(i, j) -= w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_rho);
            }
    });
    clampExtrema2(ni, nj,rho_curr,rho);
    /// T
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
//        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                temp_T(i, j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), temperature) - dT(i,j));
            }
    });
    temp_T -= T_init;
    temp_T *= 0.5f;
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
//        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                temperature(i, j) -= w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_T);
            }
    });
    clampExtrema2(ni,nj,T_curr,temperature);
}

void BimocqSolver2D::correctVelocity(Array2f &semi_u, Array2f &semi_v)
{
    Array2f u_curr = u;
    Array2f v_curr = v;
    Array2f temp_u, temp_v;
    temp_u.resize(ni+1, nj, 0.0);
    temp_v.resize(ni, nj+1, 0.0);
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        //        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(i>1&&i<ni-1 && j>0 && j<nj-1)
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                clampPos(pos1);
                temp_u(i, j) +=  w[k] * (sampleField(pos1 - h * Vec2f(0.0, 0.5), u) - du(i,j));
            }
    });
    temp_u -= u_init;
    temp_u *= 0.5f;
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        //        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(i>1&&i<ni-1 && j>0 && j<nj-1)
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                clampPos(pos1);
                u(i, j) -=  w[k] * sampleField(pos1 - h * Vec2f(0.0, 0.5), temp_u);
//                if (i<3 || i> ni-3 || j<2 || j>nj-3) u(i,j) = semi_u(i,j);
            }
    });
    clampExtrema2(ni+1, nj, u_curr, u);
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        //        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(j>1&&j<nj-1 && i>0&&i<ni-1)
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                clampPos(pos1);
                temp_v(i, j) += w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.0), v) - dv(i,j));
            }
    });
    temp_v -= v_init;
    temp_v *= 0.5f;
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        //        float w[5] = {0.f,0.f,0.f,0.f,1.0f};
        float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
        if(j>1&&j<nj-1 && i>0&&i<ni-1)
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                clampPos(pos1);
                v(i, j) -=  w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.0), temp_v));
//                if (i<2 || i> ni-3 || j<3 || j>nj-3) v(i,j) = semi_v(i,j);
            }
    });
    clampExtrema2(ni, nj+1, v_curr, v);
}

void BimocqSolver2D::advectVelocity(Array2f &semi_u, Array2f &semi_v)
{
    float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
    /// u
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        u(i,j) = 0;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>1 && j<nj-2)
        {
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init, y_init);
                clampPos(pos1);
                float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_xprev);
                float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_yprev);
                Vec2f pos2 = Vec2f(x_origin, y_origin);
                clampPos(pos2);
                u(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.0, 0.5), u_origin) +
                                             sampleField(pos1 - h * Vec2f(0.0, 0.5), du) +
                                             sampleField(pos2 - h* Vec2f(0,0.5), du_prev)
                );
                u(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.0, 0.5), u_init) +
                                             sampleField(pos1 - h * Vec2f(0.0, 0.5), du));
            }
        }
        else
        {
            u(i,j) = semi_u(i,j);
        }
    });
    /// v
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        v(i, j) = 0.0;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(j>1 && j<nj-1 && i>1 && i<ni-2)
        {
            for (int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init, y_init);
                clampPos(pos1);
                float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_xprev);
                float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_yprev);
                Vec2f pos2 = Vec2f(x_origin, y_origin);
                clampPos(pos2);
                v(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.5, 0.0), v_origin) +
                                             sampleField(pos1 - h * Vec2f(0.5, 0.0), dv) +
                                             sampleField(pos2 - h * Vec2f(0.5,0.0), dv_prev));
                v(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.0), v_init) +
                                             sampleField(pos1 - h * Vec2f(0.5, 0.0), dv));
            }
        }
        else{
            v(i,j) = semi_v(i,j);
        }
    });
}

void BimocqSolver2D::advectScalars(Array2f &semi_rho, Array2f &semi_T)
{
    float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        rho(i,j) = 0.f;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(j>1&&j<nj-1 && i>0&&i<ni-1)
        {
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                clampPos(pos1);
                float x_origin = sampleField(pos1 - h*Vec2f(0.5), backward_scalar_xprev);
                float y_origin = sampleField(pos1 - h*Vec2f(0.5), backward_scalar_yprev);
                Vec2f pos2 = Vec2f(x_origin,y_origin);
                clampPos(pos2);
                rho(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.5, 0.5), rho_orig) +
                                               sampleField(pos1 - h * Vec2f(0.5, 0.5), drho) +
                                               sampleField(pos2 - h * Vec2f(0.5, 0.5), drho_prev));
                rho(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.5), rho_init) +
                                               sampleField(pos1 - h * Vec2f(0.5, 0.5), drho));
            }
        }
        else{
            rho(i,j) = semi_rho(i,j);
        }
    });
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        temperature(i,j) = 0.f;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(j>1&&j<nj-1 && i>0&&i<ni-1) {
            for (int k = 0; k < 5; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f pos1 = Vec2f(x_init, y_init);
                clampPos(pos1);
                float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_xprev);
                float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_yprev);
                Vec2f pos2 = Vec2f(x_origin, y_origin);
                clampPos(pos2);
                temperature(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.5, 0.5), T_orig) +
                                                        sampleField(pos1 - h * Vec2f(0.5, 0.5), dT) +
                                                        sampleField(pos2 - h * Vec2f(0.5, 0.5), dT_prev));
                temperature(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.5), T_init) +
                                             sampleField(pos1 - h * Vec2f(0.5, 0.5), dT));
            }
        }
        else{
            temperature(i,j) = semi_T(i,j);
        }
    });
}

void BimocqSolver2D::accumulateVelocity(Array2f &u_change, Array2f &v_change, float proj_coeff, bool error_correction)
{
    float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
    Array2f test_du, test_du_star;
    Array2f test_dv, test_dv_star;
    test_du.resize(ni+1, nj, 0.0);
    test_du_star.resize(ni+1, nj, 0.0);
    test_dv.resize(ni, nj+1, 0.0);
    test_dv_star.resize(ni, nj+1, 0.0);
    // step 7
    // sample du, dv, drho, dT
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                test_du(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.0, 0.5), u_change);
            }
    });
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                test_du_star(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.0, 0.5), test_du);
            }
    });
    test_du_star -= u_change;
    test_du_star *= 0.5;
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                if (error_correction){
                    du(i,j) += w[k]* proj_coeff *(sampleField(samplePos - h * Vec2f(0.0, 0.5), u_change) -
                                       sampleField(samplePos - h * Vec2f(0.0, 0.5), test_du_star));
                }
                else{
                    du(i,j) += w[k]* proj_coeff *sampleField(samplePos - h * Vec2f(0.0, 0.5), u_change);
                }
            }
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>0 && i<ni-1 && j>1 && j< nj-1)
            for (int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                test_dv(i, j) += w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.0), v_change);
            }
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>0 && i<ni-1 && j>1 && j< nj-1)
            for (int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                test_dv_star(i, j) += w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.0), test_dv);
            }
    });
    test_dv_star -= v_change;
    test_dv_star *= 0.5;
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>0 && i<ni-1 && j>1 && j< nj-1)
            for (int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                if (error_correction){
                    dv(i,j) += w[k] * proj_coeff * (sampleField(samplePos - h * Vec2f(0.5, 0.0), v_change) -
                                           sampleField(samplePos - h * Vec2f(0.5, 0.0), test_dv_star));
                }
                else{
                    dv(i,j) += w[k] * proj_coeff * (sampleField(samplePos - h * Vec2f(0.5, 0.0), v_change));
                }
            }
    });
}

void BimocqSolver2D::updateForward(float dt, Array2f &fwd_x, Array2f &fwd_y)
{
    // forward mapping
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = Vec2f(fwd_x(i, j), fwd_y(i, j));
        Vec2f posNew = solveODE(dt, pos);
        clampPos(posNew);
        fwd_x(i, j) = posNew[0];
        fwd_y(i, j) = posNew[1];
    });
}

void BimocqSolver2D::updateBackward(float dt, Array2f &back_x, Array2f &back_y)
{
    // backward mapping
    float substep = _cfl;
    float T = dt;
    float t = 0;
    while(t < T)
    {
        if (t + substep > T) substep = T - t;
        map_tempx.assign(ni, nj, 0.0f);
        map_tempy.assign(ni, nj, 0.0f);
        semiLagAdvectDMC(back_x, map_tempx, substep, ni, nj, 0.5, 0.5);
        semiLagAdvectDMC(back_y, map_tempy, substep, ni, nj, 0.5, 0.5);
        back_x = map_tempx;
        back_y = map_tempy;
        t += substep;
    }
}

void BimocqSolver2D::clampExtrema2(int _ni, int _nj, Array2f &before, Array2f &after)
{
    tbb::parallel_for((int) 0, (_ni) * _nj, 1, [&](int tIdx) {
        int i = tIdx % (_ni);
        int j = tIdx / (_ni);
        float min_v=1e+6, max_v=0;
        for(int jj=j-1;jj<=j+1;jj++)for(int ii=i-1;ii<=i+1;ii++)
            {
                max_v = std::max(max_v, before.at(ii,jj));
                min_v = std::min(min_v, before.at(ii,jj));
            }
        after(i,j) = std::min(std::max(after(i,j),min_v),max_v);
    });
}

void BimocqSolver2D::accumulateScalars(Array2f &rho_change, Array2f &T_change, bool error_correction)
{
    float w[5] = {0.125f,0.125f,0.125f,0.125f,0.5f};
    Array2f temp_scalar;
    Array2f T_scalar;
    Array2f temp_scalar_star;
    Array2f T_scalar_star;
    temp_scalar.resize(ni, nj, 0.0);
    T_scalar.resize(ni, nj, 0.0);
    temp_scalar_star.resize(ni, nj, 0.0);
    T_scalar_star.resize(ni, nj, 0.0);
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                temp_scalar(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_change);
            }
    });
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                temp_scalar_star(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_scalar);
            }
    });
    temp_scalar_star -= rho_change;
    temp_scalar_star *= 0.5f;
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                if (error_correction){
                    drho(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_change) -
                                       sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_scalar_star));
                }
                else{
                    drho(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_change));
                }
            }
    });
    /// T
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                T_scalar(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), T_change);
            }
    });
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                T_scalar_star(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), T_scalar);
            }
    });
    T_scalar_star -= T_change;
    T_scalar_star *= 0.5f;
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        std::vector<Vec2f> dir(5);
        dir[0] = Vec2f(-0.25,-0.25);
        dir[1] = Vec2f(0.25, -0.25);
        dir[2] = Vec2f(-0.25, 0.25);
        dir[3] = Vec2f( 0.25, 0.25);
        dir[4] = Vec2f(0.0, 0.0);
        if(i>1&&i<ni-1 && j>0 && j< nj-1)
            for(int k = 0; k < 5; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                clampPos(samplePos);
                if (error_correction){
                    dT(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), T_change) -
                                       sampleField(samplePos - h * Vec2f(0.5, 0.5), T_scalar_star));
                }
                else{
                    dT(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), T_change));
                }
            }
    });
}

void BimocqSolver2D::resampleVelBuffer(float dt)
{
    std::cout<< RED << "velocity remeshing!\n" << RESET;
    total_resampleCount ++;
    u_origin = u_init;
    v_origin = v_init;
    u_init = u;
    v_init = v;
    du_prev = du;
    dv_prev = dv;
    du.assign(0);
    dv.assign(0);

    backward_xprev = backward_x;
    backward_yprev = backward_y;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        forward_x(i, j) = h*((float)i + 0.5);
        backward_x(i, j) = h*((float)i + 0.5);
        forward_y(i, j) = h*((float)j + 0.5);
        backward_y(i, j) = h*((float)j + 0.5);
    });
}

void BimocqSolver2D::resampleRhoBuffer(float dt)
{
    std::cout<< BLUE << "rho remeshing!\n" << RESET;
    total_scalar_resample ++;
    rho_orig = rho_init;
    rho_init = rho;
    T_orig = T_init;
    T_init = temperature;
    drho_prev = drho;
    dT_prev = dT;
    drho.assign(0);
    dT.assign(0);

    backward_scalar_xprev = backward_scalar_x;
    backward_scalar_yprev = backward_scalar_y;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        forward_scalar_x(i, j) = h*((float)i + 0.5);
        backward_scalar_x(i, j) = h*((float)i + 0.5);
        forward_scalar_y(i, j) = h*((float)j + 0.5);
        backward_scalar_y(i, j) = h*((float)j + 0.5);
    });
}

void BimocqSolver2D::advanceFLIP(float dt, int currentframe)
{
    std::cout << BLUE <<  "FLIP scheme frame " << currentframe << " starts !" << RESET << std::endl;

    tbb::parallel_for((int)0, (int)cParticles.size(), 1, [&](int p)
    {
        Vec2f pos = solveODE(dt,cParticles[p].pos_current);
        //push particle back to domain;
        pos[0] = std::min(std::max(h, pos[0]), ((float)ni-1)*h);
        pos[1] = std::min(std::max(h, pos[1]), ((float)nj-1)*h);
        cParticles[p].pos_current = pos;
    });
    Array2f u_weight, v_weight, rho_weight, T_weight;
    u_weight.resize(u.ni,u.nj);
    v_weight.resize(v.ni,v.nj);
    rho_weight.resize(rho.ni, rho.nj);
    T_weight.resize(temperature.ni, temperature.nj);
    u_weight.assign(1e-4);
    v_weight.assign(1e-4);
    rho_weight.assign(1e-4);
    T_weight.assign(1e-4);
    u.assign(0);
    v.assign(0);
    rho.assign(0);
    temperature.assign(0);
    // splat particle properties to grid
    for(int i=0;i<cParticles.size();i++)
    {
        CmapParticles p = cParticles[i];
        int ii, jj;
        ii = floor(p.pos_current[0]/h);
        jj = floor(p.pos_current[1]/h-0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0,0.5)*h;
            float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);
            u(iii,jjj) += w*p.vel[0];
            u_weight(iii,jjj) += w;
        }

        ii = floor(p.pos_current[0]/h - 0.5);
        jj = floor(p.pos_current[1]/h);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0.5,0)*h;
            float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);
            v(iii,jjj) += w*p.vel[1];
            v_weight(iii,jjj) += w;
        }
        // density
        ii = floor(p.pos_current[0]/h - 0.5);
        jj = floor(p.pos_current[1]/h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0.5,0.5)*h;
            float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);
            rho(iii,jjj) += w*p.rho;
            rho_weight(iii,jjj) += w;
        }
        // temperature
        ii = floor(p.pos_current[0]/h - 0.5);
        jj = floor(p.pos_current[1]/h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
        {
            Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0.5,0.5)*h;
            float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);
            temperature(iii,jjj) += w*p.temperature;
            T_weight(iii,jjj) += w;
        }
    }
    u /= u_weight;
    v /= v_weight;

    Array2f u_save, v_save, rho_save, temperature_save;
    u_save = u;
    v_save=v;
    rho_save = rho;
    temperature_save = temperature;

    applyBuoyancyForce(dt);
    float tol = 1e-6;
    projection(tol,use_neumann_boundary);
    Array2f u_diff, v_diff, rho_diff, temperature_diff;
    u_diff = u;
    v_diff = v;
    rho_diff = rho;
    temperature_diff = temperature;
    u_diff -= u_save;
    v_diff -= v_save;
    rho_diff -= rho_save;
    temperature_diff -= temperature_save;
    float flip = 0.99;

    tbb::parallel_for((int)0, (int)cParticles.size(), 1, [&](int i)
    {
        Vec2f p_vel = cParticles[i].vel;
        float p_rho = cParticles[i].rho;
        float p_temperature = cParticles[i].temperature;
        p_vel = flip*(p_vel + Vec2f(sampleField(cParticles[i].pos_current - h*Vec2f(0,0.5), u_diff), sampleField(cParticles[i].pos_current - h*Vec2f(0.5,0), v_diff)))
                + (1-flip)*getVelocity(cParticles[i].pos_current);
        p_rho = flip*(p_rho + sampleField(cParticles[i].pos_current - h*Vec2f(0.5,0.5), rho_diff))
                + (1-flip)*sampleField(cParticles[i].pos_current - h*Vec2f(0.5,0.5), rho);
        p_temperature = flip*(p_temperature + sampleField(cParticles[i].pos_current - h*Vec2f(0.5,0.5), temperature_diff))
                + (1-flip)*sampleField(cParticles[i].pos_current - h*Vec2f(0.5,0.5), temperature);
        cParticles[i].vel = p_vel;
        cParticles[i].rho = p_rho;
        cParticles[i].temperature = p_temperature;
    });
}

void BimocqSolver2D::setSmoke(float smoke_rise, float smoke_drop)
{
    alpha = smoke_rise;
    beta = smoke_drop;
}

void BimocqSolver2D::advancePolyPIC(float dt, int currentframe)
{
    if (sim_scheme == POLYPIC) std::cout << BLUE <<  "PolyPIC scheme frame " << currentframe << " starts !" << RESET << std::endl;
    if (sim_scheme == APIC) std::cout << BLUE <<  "APIC scheme frame " << currentframe << " starts !" << RESET << std::endl;

    tbb::parallel_for((int)0, (int)cParticles.size(), 1, [&](int p)
    {
        Vec2f pos = solveODE(dt,cParticles[p].pos_current);
        //push particle back to domain;
        pos[0] = std::min(std::max(h, pos[0]), ((float)ni-1)*h);
        pos[1] = std::min(std::max(h, pos[1]), ((float)nj-1)*h);
        cParticles[p].pos_current = pos;
    });
    Array2f u_weight, v_weight, rho_weight, T_weight;
    u_weight.resize(u.ni,u.nj);
    v_weight.resize(v.ni,v.nj);
    rho_weight.resize(rho.ni, rho.nj);
    T_weight.resize(temperature.ni, temperature.nj);
    u_weight.assign(1e-4);
    v_weight.assign(1e-4);
    rho_weight.assign(1e-4);
    T_weight.assign(1e-4);
    u.assign(0);
    v.assign(0);
    rho.assign(0);
    temperature.assign(0);

    // splat particle properties to grid
    for(int i=0;i<cParticles.size();i++)
    {
        CmapParticles p = cParticles[i];
        int ii, jj;
        ii = floor(p.pos_current[0]/h);
        jj = floor(p.pos_current[1]/h-0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0,0.5)*h;
                float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);

                float c0 = p.C_x[0];
                float c1 = p.C_x[1]*(gpos[0] - p.pos_current.v[0]);
                float c2 = p.C_x[2]*(gpos[1] - p.pos_current.v[1]);
                float c3 = p.C_x[3]*(gpos[0] - p.pos_current.v[0])*(gpos[1] - p.pos_current.v[1]);

                if (sim_scheme == POLYPIC) u(iii,jjj) += w * (c0 + c1 + c2 + c3);
                if (sim_scheme == APIC) u(iii,jjj) += w * (c0 + c1 + c2);
                u_weight(iii,jjj) += w;
            }

        ii = floor(p.pos_current[0]/h - 0.5);
        jj = floor(p.pos_current[1]/h);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0.5,0)*h;
                float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);

                float c0 = p.C_y[0];
                float c1 = p.C_y[1]*(gpos[0] - p.pos_current.v[0]);
                float c2 = p.C_y[2]*(gpos[1] - p.pos_current.v[1]);
                float c3 = p.C_y[3]*(gpos[0] - p.pos_current.v[0])*(gpos[1] - p.pos_current.v[1]);

                if (sim_scheme == POLYPIC) v(iii,jjj) += w * (c0 + c1 + c2 + c3);
                if (sim_scheme == APIC) v(iii,jjj) += w * (c0 + c1 + c2);
                v_weight(iii,jjj) += w;
            }

        ii = floor(p.pos_current[0]/h - 0.5);
        jj = floor(p.pos_current[1]/h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0.5,0.5)*h;
                float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);

                float c0 = p.C_rho[0];
                float c1 = p.C_rho[1]*(gpos[0] - p.pos_current.v[0]);
                float c2 = p.C_rho[2]*(gpos[1] - p.pos_current.v[1]);
                float c3 = p.C_rho[3]*(gpos[0] - p.pos_current.v[0])*(gpos[1] - p.pos_current.v[1]);

                if (sim_scheme == POLYPIC) rho(iii,jjj) += w*(c0 + c1 + c2 + c3);
                if (sim_scheme == APIC) rho(iii,jjj) += w*(c0 + c1 + c2);
                rho_weight(iii,jjj) += w;
            }

        ii = floor(p.pos_current[0]/h - 0.5);
        jj = floor(p.pos_current[1]/h - 0.5);
        for(int jjj=jj;jjj<=jj+1;jjj++)for(int iii=ii;iii<=ii+1;iii++)
            {
                Vec2f gpos = Vec2f(iii,jjj)*h + Vec2f(0.5,0.5)*h;
                float w = p.kernel((p.pos_current[0] - gpos[0])/h)*p.kernel((p.pos_current[1] - gpos[1])/h);

                float c0 = p.C_temperature[0];
                float c1 = p.C_temperature[1]*(gpos[0] - p.pos_current.v[0]);
                float c2 = p.C_temperature[2]*(gpos[1] - p.pos_current.v[1]);
                float c3 = p.C_temperature[3]*(gpos[0] - p.pos_current.v[0])*(gpos[1] - p.pos_current.v[1]);

                if (sim_scheme == POLYPIC) temperature(iii,jjj) += w*(c0 + c1 + c2 + c3);
                if (sim_scheme == APIC) temperature(iii,jjj) += w*(c0 + c1 + c2);
                T_weight(iii,jjj) += w;
            }
    }

    u /= u_weight;
    v /= v_weight;
    rho /= rho_weight;
    temperature /= T_weight;

    applyBuoyancyForce(dt);
    float tol = 1e-6;
    projection(tol, use_neumann_boundary);

    // gather from grid
    tbb::parallel_for((int)0, (int)cParticles.size(), 1, [&](int i)
    {
        Vec2f pos = cParticles[i].pos_current;
        cParticles[i].vel = getVelocity(cParticles[i].pos_current);
        cParticles[i].rho = sampleField(cParticles[i].pos_current - Vec2f(0.5, 0.5), rho);
        cParticles[i].temperature = sampleField(cParticles[i].pos_current - Vec2f(0.5, 0.5), temperature);
        // update Cp
        cParticles[i].C_x = cParticles[i].calculateCp(pos, u, h, ni+1, nj, 0.0, 0.5);
        cParticles[i].C_y = cParticles[i].calculateCp(pos, v, h, ni, nj+1, 0.5, 0.0);
        cParticles[i].C_rho = cParticles[i].calculateCp(pos, rho, h, ni, nj, 0.5, 0.5);
        cParticles[i].C_temperature = cParticles[i].calculateCp(pos, temperature, h, ni, nj, 0.5, 0.5);
    });
}

void BimocqSolver2D::diffuseField(float nu, float dt, Array2f &field)
{
    Array2f field_temp;
    field_temp = field;
    double coef = nu*(dt/(h*h));
    int n = field.ni*field.nj;
    for(int iter=0;iter<20;iter++)
    {
        tbb::parallel_for(0,n,1,[&](int tid)
        {
            int i = tid%field.ni;
            int j = tid/field.ni;
            if((i+j)%2==0)
            {
                float b_ij = field(i,j);
                float x_l  = ((i-1)>=0)?field_temp(i-1,j):0;
                float x_r  = ((i+1)<field.ni)?field_temp(i+1,j):0;
                float x_u  = ((j+1)<field.nj)?field_temp(i,j+1):0;
                float x_d  = ((j-1)>=0)?field_temp(i,j-1):0;

                field_temp(i,j) = (b_ij + coef*(x_l + x_r + x_u + x_d))/(1.0+4.0*coef);
            }
        });
        tbb::parallel_for(0,n,1,[&](int tid)
        {
            int i = tid%field.ni;
            int j = tid/field.ni;
            if((i+j)%2==1)
            {
                float b_ij = field(i,j);
                float x_l  = ((i-1)>=0)?field_temp(i-1,j):0;
                float x_r  = ((i+1)<field.ni)?field_temp(i+1,j):0;
                float x_u  = ((j+1)<field.nj)?field_temp(i,j+1):0;
                float x_d  = ((j-1)>=0)?field_temp(i,j-1):0;

                field_temp(i,j) = (b_ij + coef*(x_l + x_r + x_u + x_d))/(1.0+4.0*coef);
            }
        });
    }
    field = field_temp;
}

void BimocqSolver2D::advanceBFECC(float dt, int currentframe)
{
    std::cout << BLUE <<  "BFECC scheme frame " << currentframe << " starts !" << RESET << std::endl;
    // advect rho
    Array2f rho_first;
    Array2f rho_sec;
    rho_first.assign(ni, nj, 0.0);
    rho_sec.assign(ni, nj, 0.0);
    solveBFECC(rho, rho_first, rho_sec, dt, ni, nj, 0.5, 0.5);
    rho = rho_first;

    // advect temperature
    Array2f T_first;
    Array2f T_sec;
    T_first.assign(ni, nj, 0.0);
    T_sec.assign(ni, nj, 0.0);
    solveBFECC(temperature, T_first, T_sec, dt, ni, nj, 0.5, 0.5);
    temperature = T_first;

    // advect velocity
    u_first.assign(ni+1, nj, 0.0);
    v_first.assign(ni, nj+1, 0.0);
    u_sec.assign(ni+1, nj, 0.0);
    v_sec.assign(ni, nj+1, 0.0);
    solveBFECC(u, u_first, u_sec, dt, ni+1, nj, 0.0, 0.5);
    solveBFECC(v, v_first, v_sec, dt, ni, nj+1, 0.5, 0.0);
    u = u_first;
    v = v_first;

    applyBuoyancyForce(dt);
    projection(1e-6,use_neumann_boundary);
}

void BimocqSolver2D::advanceMaccormack(float dt, int currentframe)
{
    std::cout << BLUE <<  "MacCormack scheme frame " << currentframe << " starts !" << RESET << std::endl;
    // advect rho
    Array2f rho_first;
    Array2f rho_sec;
    rho_first.assign(ni, nj, 0.0);
    rho_sec.assign(ni, nj, 0.0);
    solveMaccormack(rho, rho_first, rho_sec, dt, ni, nj, 0.5, 0.5);
    rho = rho_first;

    // advect temperature
    Array2f T_first;
    Array2f T_sec;
    T_first.assign(ni, nj, 0.0);
    T_sec.assign(ni, nj, 0.0);
    solveMaccormack(temperature, T_first, T_sec, dt, ni, nj, 0.5, 0.5);
    temperature = T_first;

    // advect velocity
    u_first.assign(ni+1, nj, 0.0);
    v_first.assign(ni, nj+1, 0.0);
    u_sec.assign(ni+1, nj, 0.0);
    v_sec.assign(ni, nj+1, 0.0);
    solveMaccormack(u, u_first, u_sec, dt, ni+1, nj, 0.0, 0.5);
    solveMaccormack(v, v_first, v_sec, dt, ni, nj+1, 0.5, 0.0);
    u = u_first;
    v = v_first;

    applyBuoyancyForce(dt);
    projection(1e-6,use_neumann_boundary);
}

void BimocqSolver2D::seedParticles(int N)
{
    cParticles.resize(N*N*ni*nj);
    tbb::parallel_for((int)0,
                      (int)cParticles.size() / (N*N),
                      (int)1,
                      [&](int tId)
                      {
                          //start frame particles
                          int i = tId%ni;
                          int j = tId/ni;
                          float x = ((float)i + 1./(2*N))*h;
                          float y = ((float)j + 1./(2*N))*h;
                          for(int ii=0;ii<N;ii++)
                          {
                              for(int jj=0;jj<N;jj++)
                              {
                                  int idx = ii*N+jj;
                                  CmapParticles *p = &(cParticles[tId + idx*ni*nj]);
                                  p->pos_current = Vec2f(x+(1./N)*ii*h,y+(1./N)*jj*h);
                              }
                          }
                      });
}

void BimocqSolver2D::setInitVelocity(float distance)
{
    SparseMatrixd M;
    int n = ni*nj;
    M.resize(n);
    tbb::parallel_for((int)0, ni*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / ni;
        int i = thread_idx % ni;
        //if in fluid domain
        if (i >= 0 && j >= 0 && i < ni&&j < nj)
        {
            //if(boundaryMask(i,j) == 0)
            {
                if (i-1>=0 ){//&& boundaryMask(i - 1, j) == 0) {
                    M.add_to_element(thread_idx, thread_idx - 1, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }

                if (i+1<ni ){//&& boundaryMask(i + 1, j) == 0) {
                    M.add_to_element(thread_idx, thread_idx + 1, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }

                if (j-1>=0 ){//&& boundaryMask(i, j - 1) == 0) {
                    M.add_to_element(thread_idx, thread_idx - ni, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }

                if (j+1<nj ){//&& boundaryMask(i, j + 1) == 0) {
                    M.add_to_element(thread_idx, thread_idx + ni, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
            }
        }
    });
    FixedSparseMatrixd M_fix;
    M_fix.construct_from_matrix(M);
    std::vector<FixedSparseMatrixd *> AA_L;
    std::vector<FixedSparseMatrixd *> RR_L;
    std::vector<FixedSparseMatrixd *> PP_L;
    std::vector<Vec2i>                SS_L;
    int ttotal_level;
    mgLevelGenerator.generateLevelsGalerkinCoarsening2D(AA_L, RR_L, PP_L, SS_L, ttotal_level, M_fix, ni, nj);

    //initialize curl;
    float max_curl=0;
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1 , [&](int thread_Idx)
    {
        int j = thread_Idx/(ni+1);
        int i = thread_Idx%(ni+1);
        Vec2f pos = h*Vec2f(i, j) - Vec2f(M_PI);
        Vec2f vort_pos0 = Vec2f(-0.5*distance,0);
        Vec2f vort_pos1 = Vec2f(+0.5*distance,0);
        double r_sqr0 = dist2(pos, vort_pos0);
        double r_sqr1 = dist2(pos, vort_pos1);
        curl(i,j) = +1.0/0.3*(2.0 - r_sqr0/0.09)*exp(0.5*(1.0 - r_sqr0/0.09));
        curl(i,j) += 1.0/0.3*(2.0 - r_sqr1/0.09)*exp(0.5*(1.0 - r_sqr1/0.09));
        max_curl = std::max(fabs(curl(i,j)), max_curl);
    }
    );
    rhs.assign(ni*nj,0);
    pressure.resize(ni*nj);
    //compute stream function
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        rhs[j*ni + i] = curl(i,j);
        pressure[j*ni+i] = 0;
    }
    );
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(M_fix,rhs,pressure,AA_L,RR_L,PP_L,SS_L,ttotal_level,1e-6,500,res_out,iter_out,ni,nj, false);
    if (converged)
        std::cout << "pressure solver converged in " << iter_out << " iterations, with residual " << res_out << std::endl;
    else
        std::cout << "warning! solver didn't reach convergence with maximum iterations!" << std::endl;

    curl.assign(0);
    //compute u = curl psi
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        curl(i,j) = pressure[j*ni+i];
    }
    );

    tbb::parallel_for((int)0, (ni+1)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni+1);
        int i = thread_Idx%(ni+1);
        u(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_init(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_origin(i,j) = (curl(i, j+1) - curl(i,j))/h;
    });
    tbb::parallel_for((int)0, (ni)*(nj+1), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        v(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_init(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_origin(i,j) = -(curl(i+1, j) - curl(i,j))/h;
    });
    cBar = color_bar(max_curl);
}

void BimocqSolver2D::setInitReyleighTaylor(float layer_height)
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        Vec2f pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
        float preturb = layer_height + 0.05f*cos(10*M_PI*pos.v[0]);
        if (pos.v[1] >= preturb)
        {
            rho(i,j) = 1.f;
            rho_init(i,j) = 1.f;
            rho_orig(i,j) = 1.f;
        }
        else{
            temperature(i,j) = 1.f;
            T_init(i,j) = 1.f;
            T_orig(i,j) = 1.f;
        }
    });
}

void BimocqSolver2D::setInitLeapFrog(float dist_a, float dist_b, float rho_h, float rho_w)
{
    //initialize curl;
    float max_curl=0;
    float a = 0.02f;
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1 , [&](int thread_Idx)
      {
          int j = thread_Idx/(ni+1);
          int i = thread_Idx%(ni+1);
          Vec2f pos = h*Vec2f(i, j) - Vec2f(M_PI);
          Vec2f vort_pos0 = Vec2f(-0.5*dist_a,-2.0f);
          Vec2f vort_pos1 = Vec2f(+0.5*dist_a,-2.0f);
          Vec2f vort_pos2 = Vec2f(-0.5*dist_b,-2.0f);
          Vec2f vort_pos3 = Vec2f(+0.5*dist_b,-2.0f);
          double r_sqr0 = dist2(pos, vort_pos0);
          double r_sqr1 = dist2(pos, vort_pos1);
          double r_sqr2 = dist2(pos, vort_pos2);
          double r_sqr3 = dist2(pos, vort_pos3);
          float c_a = 1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr0)/a/a);
          float c_b = -1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr1)/a/a);
          float c_c = 1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr2)/a/a);
          float c_d = -1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr3)/a/a);
          curl(i,j) += c_a;
          curl(i,j) += c_b;
          curl(i,j) += c_c;
          curl(i,j) += c_d;
          max_curl = std::max(fabs(curl(i,j)), max_curl);
      }
    );
    rhs.assign(ni*nj,0);
    pressure.resize(ni*nj);
    //compute stream function
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
                          int j = thread_Idx/(ni);
                          int i = thread_Idx%(ni);
                          rhs[j*ni + i] = curl(i,j);
                          pressure[j*ni+i] = 0;
                      }
    );
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(matrix_fix,rhs,pressure,A_L,R_L,P_L,S_L,total_level,1e-6,500,res_out,iter_out,ni,nj, false);
    if (converged)
        std::cout << "pressure solver converged in " << iter_out << " iterations, with residual " << res_out << std::endl;
    else
        std::cout << "warning! solver didn't reach convergence with maximum iterations!" << std::endl;

    curl.assign(0);
    //compute u = curl psi
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
                          int j = thread_Idx/(ni);
                          int i = thread_Idx%(ni);
                          curl(i,j) = pressure[j*ni+i];
                      }
    );

    tbb::parallel_for((int)0, (ni+1)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni+1);
        int i = thread_Idx%(ni+1);
        u(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_init(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_origin(i,j) = (curl(i, j+1) - curl(i,j))/h;
    });
    tbb::parallel_for((int)0, (ni)*(nj+1), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        v(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_init(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_origin(i,j) = -(curl(i+1, j) - curl(i,j))/h;
    });
    cBar = color_bar(max_curl);

    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        Vec2f pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
        if (rho_h - rho_w < pos[1] && pos[1] < rho_h + rho_w && pos[0] > rho_w && pos[0] < 2*M_PI - rho_w)
        {
            rho(i,j) = 1.f;
            rho_init(i,j) = 1.f;
            rho_orig(i,j) = 1.f;
        }
    });

}

void BimocqSolver2D::setInitZalesak()
{
    // for circle
    float r = 0.1*ni*h;
    float center_x = 0.5*ni*h;
    float center_y = 0.65*ni*h;
    // for rectangle
    float width = 0.04*ni*h;
    float height = 0.20*ni*h;
    float rec_x = 0.5*ni*h;
    float rec_y = 0.6*ni*h;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        float pos_x = h*(i + 0.5);
        float pos_y = h*(j + 0.5);
        float circle = sqrt((pos_x-center_x)*(pos_x-center_x) + (pos_y-center_y)*(pos_y-center_y)) - r;
        Vec2f p = Vec2f(pos_x, pos_y) - Vec2f(rec_x, rec_y);
        Vec2f d = Vec2f(abs(p[0]), abs(p[1])) - .5f*Vec2f(width, height);
        Vec2f maxv = max_union(d,Vec2f(0));
        float rec = dist(maxv, Vec2f(0.0)) + min(max(d[0], d[1]), 0.f);
        rho(i,j) = max(circle,-rec);
        rho_init(i,j) = max(circle,-rec);
        rho_orig(i,j) = max(circle,-rec);
    });
    // init velocity field
    Vec2f center = Vec2f(0.5*ni*h, 0.5*ni*h);
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
//        if(dist(pos, center) < 0.5*ni*h) u(i, j) = PI*(0.5*ni*h - pos.v[1]) / 314.f;
        u(i, j) = M_PI*(0.5*ni*h - pos.v[1]) / 314.f;
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
//        if(dist(pos, center) < 0.5*ni*h) v(i, j) = PI*(pos.v[0] - 0.5*ni*h) / 314.f;
        v(i, j) = M_PI*(pos.v[0] - 0.5*ni*h) / 314.f;
    });
}

void BimocqSolver2D::setInitVortexBox()
{
    // for circle
    float r = 0.15*ni*h;
    float center_x = 0.5*ni*h;
    float center_y = 0.75*ni*h;
    float normalize = 0.f;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        float pos_x = h*(i + 0.5);
        float pos_y = h*(j + 0.5);
        float circle = sqrt((pos_x-center_x)*(pos_x-center_x) + (pos_y-center_y)*(pos_y-center_y)) - r;
        rho(i,j) = circle;
        rho_init(i,j) = circle;
        rho_orig(i,j) = circle;
        pos_x /= ni*h;
        pos_y /= nj*h;
        float tmp_x = -2.f*sin(M_PI*pos_x)*sin(M_PI*pos_x)*sin(M_PI*pos_y)*cos(M_PI*pos_y);
        float tmp_y = 2.f*sin(M_PI*pos_x)*cos(M_PI*pos_x)*sin(M_PI*pos_y)*sin(M_PI*pos_y);
        float mag = sqrt(tmp_x*tmp_x+tmp_y*tmp_y);
        if (mag > normalize) normalize = mag;
    });
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
        float pos_x = pos[0] / (ni*h);
        float pos_y = pos[1] / (nj*h);
        u(i, j) = -2.f*sin(M_PI*pos_x)*sin(M_PI*pos_x)*sin(M_PI*pos_y)*cos(M_PI*pos_y) / normalize;
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
        float pos_x = pos[0] / (ni*h);
        float pos_y = pos[1] / (nj*h);
        v(i, j) = 2.f*sin(M_PI*pos_x)*cos(M_PI*pos_x)*sin(M_PI*pos_y)*sin(M_PI*pos_y) / normalize;
    });
}

void BimocqSolver2D::buildMultiGrid(bool PURE_NEUMANN)
{
    //build the matrix
    //we are assuming a a whole fluid domain
    int n = ni*nj;
    matrix.resize(n);
    tbb::parallel_for((int)0, ni*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / ni;
        int i = thread_idx % ni;
        //if in fluid domain
        if (i >= 0 && j >= 0 && i < ni&&j < nj)
        {
            if (i-1>=0 ){
                matrix.add_to_element(thread_idx, thread_idx - 1, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }

            if (i+1<ni ){
                matrix.add_to_element(thread_idx, thread_idx + 1, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }

            if (j-1>=0 ){
                matrix.add_to_element(thread_idx, thread_idx - ni, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }

            if (j+1<nj ){
                matrix.add_to_element(thread_idx, thread_idx + ni, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
        }
    });
    matrix_fix.construct_from_matrix(matrix);
    mgLevelGenerator.generateLevelsGalerkinCoarsening2D(A_L, R_L, P_L, S_L, total_level, matrix_fix, ni, nj);
}

void BimocqSolver2D::applyVelocityBoundary()
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx) {
        int i = tIdx%ni;
        int j = tIdx / ni;
        if(i==0)
        {
            u(i,j) = 0;
            u(i+1,j) = 0;
        }
        if(j==0)
        {
            v(i, j) = 0;
            v(i, j+1) = 0;
        }
        if(i==ni-1)
        {
            u(i,j) = 0;
            u(i + 1, j) = 0;
        }
        if(j==nj-1)
        {
            v(i,j) = 0;
            v(i, j + 1) = 0;
        }
    });
}

void BimocqSolver2D::calculateCurl() {
    curl.assign(0);
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1, [&](int tIdx)
    {
        int i = tIdx%(ni+1);
        int j = tIdx/(ni+1);
        if(i>0&&i<ni+1&&j>0&&j<nj+1)
        {
            curl(i,j) = (u(i,j) - u(i,j-1) + v(i-1,j) - v(i,j))/h;
        }
    });
}

void BimocqSolver2D::sampleParticlesFromGrid()
{

    tbb::parallel_for((int)0, (int)cParticles.size(), 1, [&](int i)
    {
        Vec2f pos = cParticles[i].pos_current;
        cParticles[i].vel = getVelocity(pos);
        cParticles[i].rho = sampleField(pos - Vec2f(0.5, 0.5), rho);
        cParticles[i].temperature = sampleField(pos - Vec2f(0.5, 0.5), temperature);
        // update Cp
        cParticles[i].C_x = cParticles[i].calculateCp(pos, u, h, ni+1, nj, 0.0, 0.5);
        cParticles[i].C_y = cParticles[i].calculateCp(pos, v, h, ni, nj+1, 0.5, 0.0);
        cParticles[i].C_rho = cParticles[i].calculateCp(pos, rho, h, ni, nj, 0.5, 0.5);
        cParticles[i].C_temperature = cParticles[i].calculateCp(pos, temperature, h, ni, nj, 0.5, 0.5);
    });
}

void BimocqSolver2D::emitSmoke()
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        if (emitterMask(i, j) == 1)
        {
            rho(i, j) = 1.0;
            temperature(i, j) = 1.0;
            if(rho_init.a.size()>0)
            {
                rho_init(i, j) = 1.0;
            }
            if(T_init.a.size()>0)
            {
                T_init(i,j) = 1.0;
            }
        }
    });
}

Vec2f BimocqSolver2D::getVelocity(Vec2f & pos)
{
    float u_sample, v_sample;
    //offset of u, we are in a staggered grid
    Vec2f upos = pos - Vec2f(0.0f, 0.5*h);
    int i = floor(upos.v[0]/h),j = floor(upos.v[1]/h);
    if (!(i >= 0 && i <= ni - 1 && j >= 0 && j <= nj - 2))
        u_sample = 0;
    else
        u_sample = bilerp(u(i, j), u(i + 1, j), u(i, j + 1), u(i + 1, j + 1), upos.v[0] / h - (float)i, upos.v[1] / h - (float)j);
    //offset of v, we are in a staggered grid
    Vec2f vpos = pos - Vec2f(0.5*h, 0.0f);
    i = floor(vpos.v[0] / h), j = floor(vpos.v[1] / h);
    if (!(i >= 0 && i <= ni - 2 && j >= 0 && j <= nj - 1))
        v_sample = 0;
    else
        v_sample = bilerp(v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1), vpos.v[0] / h - (float)i, vpos.v[1] / h - (float)j);
    return Vec2f(u_sample, v_sample);
}


float BimocqSolver2D::sampleField(Vec2f pos, const Array2f &field)
{
    Vec2f spos = pos;
    int i = floor(spos.v[0] / h), j = floor(spos.v[1] / h);
    return bilerp(field.boundedAt(i, j), field.boundedAt(i + 1, j),
                  field.boundedAt(i, j + 1), field.boundedAt(i + 1, j + 1), spos.v[0] / h - (float)i, spos.v[1] / h - (float)j);
}

void BimocqSolver2D::outputDensity(std::string folder, std::string file, int i, bool color_density)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("_\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    if (color_density)
        writeBMPColor(filename, ni, nj, rho.a.data, temperature.a.data);
    else
        writeBMP(filename, ni, nj, rho.a.data);
}

void BimocqSolver2D::outputVortVisualized(std::string folder, std::string file, int i)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("\%04d.bmp");
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

void BimocqSolver2D::outputLevelset(std::string sdfFilename, int i)
{
    std::ofstream foutU;
    std::string old_string = std::to_string(i);
    std::string new_string = std::string(4 - old_string.length(), '0') + old_string;
    std::string filenameU = sdfFilename + std::string("levelset_") + new_string + std::string(".txt");
    foutU.open(filenameU);
    for (int i = 0; i<ni; i++)
    {
        for (int j = 0; j<nj; j++)
        {
            foutU << rho(i,j) << " ";
        }
        if (i != ni-1) foutU << std::endl;
    }
    foutU.close();
}