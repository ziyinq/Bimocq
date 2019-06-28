#include "BimocqSolver.h"

BimocqSolver::BimocqSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, Scheme myscheme, gpuMapper *mymapper)
{
    _nx = nx;
    _ny = ny;
    _nz = nz;
    _h = L/nx;
    max_v = 0.f;
    viscosity = vis_coeff;
    sim_scheme = myscheme;

    _un.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vn.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wn.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _utemp.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vtemp.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wtemp.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _uinit.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vinit.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _winit.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _uprev.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vprev.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wprev.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _duproj.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _dvproj.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _dwproj.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _duextern.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _dvextern.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _dwextern.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);

    _rho.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _rhotemp.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _rhoinit.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _rhoprev.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _drhoextern.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);

    _T.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _Ttemp.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _Tinit.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _Tprev.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _dTextern.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);

    _usolid.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vsolid.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wsolid.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    u_valid.resize(_nx+1,_ny,_nz);
    v_valid.resize(_nx,_ny+1,_nz);
    w_valid.resize(_nx,_ny,_nz+1);



    _b_desc.init(_nx,_ny,_nz);
    // initialize BIMOCQ advector
    VelocityAdvector.init(_nx, _ny, _nz, _h, blend_coeff, mymapper);
    ScalarAdvector.init(_nx, _ny, _nz, _h, blend_coeff, mymapper);
    gpuSolver = mymapper;
}

void BimocqSolver::advance(int framenum, float dt)
{
    switch (sim_scheme)
    {
        case BIMOCQ:
            advanceBimocq(framenum, dt);
            break;
        case SEMILAG:
            advanceSemilag(framenum, dt);
            break;
        case MACCORMACK:
            advanceMacCormack(framenum, dt);
            break;
        case MAC_REFLECTION:
            advanceReflection(framenum, dt);
            break;
        default:
            break;
    }
}

void BimocqSolver::advanceBimocq(int framenum, float dt)
{
    float proj_coeff = 2.f;
    bool velReinit = false;
    bool scalarReinit = false;
    float cfldt = getCFL();
    if (framenum == 0) max_v = _h;
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;

    VelocityAdvector.updateMapping(_un, _vn, _wn, cfldt, dt);
    ScalarAdvector.updateMapping(_un, _vn, _wn, cfldt, dt);
    cout << "[ Update Mapping Done! ]" << endl;

    semilagAdvect(cfldt, dt);
    cout << "[ Semilag Advect Fields Done! ]" << endl;

    VelocityAdvector.advectVelocity(_un, _vn, _wn, _uinit, _vinit, _winit, _uprev, _vprev, _wprev);
    ScalarAdvector.advectField(_rho, _rhoinit, _rhoprev);
    ScalarAdvector.advectField(_T, _Tinit, _Tprev);
    cout << "[ Bimocq Advect Fields Done! ]" << endl;

    blendBoundary(_un, _utemp);
    blendBoundary(_vn, _vtemp);
    blendBoundary(_wn, _wtemp);
    blendBoundary(_rho, _rhotemp);
    blendBoundary(_T, _Ttemp);
    cout << "[ Blend Boundary Fields Done! ]" << endl;

    // save current fields to calculate change
    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    _rhotemp.copy(_rho);
    _Ttemp.copy(_T);

    clearBoundary(_rho);
    emitSmoke(framenum, dt);
    addBuoyancy(dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    // calculate velocity change due to external forces(e.g. buoyancy)
    _duextern.copy(_un); _duextern -= _utemp;
    _dvextern.copy(_vn); _dvextern -= _vtemp;
    _dwextern.copy(_wn); _dwextern -= _wtemp;

    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    projection();
    // calculate velocity change due to pressure projection
    _duproj.copy(_un); _duproj -= _utemp;
    _dvproj.copy(_vn); _dvproj -= _vtemp;
    _dwproj.copy(_wn); _dwproj -= _wtemp;
    _drhoextern.copy(_rho); _drhoextern -= _rhotemp;
    _dTextern.copy(_T); _dTextern -= _Ttemp;

    float VelocityDistortion = VelocityAdvector.estimateDistortion(_b_desc) / (max_v * dt);
    float ScalarDistortion = ScalarAdvector.estimateDistortion(_b_desc) / (max_v * dt);
    cout << "[ Velocity Distortion is " << VelocityDistortion << " ]" << endl;
    cout << "[ Scalar Distortion is " << ScalarDistortion << " ]" << endl;
    if (VelocityDistortion > 1.f || framenum - vel_lastReinit > 10)
    {
        velReinit = true;
        vel_lastReinit = framenum;
        proj_coeff = 1.f;
    }
    if (ScalarDistortion > 5.f || framenum - scalar_lastReinit > 30)
    {
        scalarReinit = true;
        scalar_lastReinit = framenum;
    }
    // accumuate buffer changes
    VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duextern, _dvextern, _dwextern, 1.f);
    VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duproj, _dvproj, _dwproj, proj_coeff);
    ScalarAdvector.accumulateField(_rhoinit, _drhoextern);
    ScalarAdvector.accumulateField(_Tinit, _dTextern);

    cout << "[ Accumulate Fields Done! ]" << endl;
    if (velReinit)
    {
        VelocityAdvector.reinitializeMapping();
        velocityReinitialize();
        VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duproj, _dvproj, _dwproj, 1.f);
        cout << RED << "[ Bimocq Velocity Re-initialize, total reinitialize count: " << VelocityAdvector.total_reinit_count << " ]" << RESET << endl;
    }
    if (scalarReinit)
    {
        ScalarAdvector.reinitializeMapping();
        scalarReinitialize();
        cout << RED << "[ Bimocq Scalar Re-initialize, total reinitialize count: " << ScalarAdvector.total_reinit_count << " ]" << RESET << endl;
    }
}

void BimocqSolver::advanceSemilag(int framenum, float dt)
{
    // first copy velocity buffers to GPU.u, GPU.v, GPU.w
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    // semi-lagrangian advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for semi-lagrangian advection
    // advect density
    float cfldt = getCFL();
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_rho, gpuSolver->x_host, gpuSolver->du);
    cout << "[ Semilag Advect Density Done! ]" << endl;

    // advect Temperature
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_T, gpuSolver->x_host, gpuSolver->du);
    cout << "[ Semilag Advect Temperature Done! ]" << endl;

    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back

    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->dw);
    cout << "[ Semilag Advect Velocity Done! ]" << endl;

    clearBoundary(_rho);
    emitSmoke(framenum, dt);
    addBuoyancy(dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    projection();
}

void BimocqSolver::advanceMacCormack(int framenum, float dt)
{
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    // MacCormack advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for advection
    // advect density
    float cfldt = getCFL();
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    cudaMemcpy(gpuSolver->dv, gpuSolver->du, sizeof(float)*_nx*_ny*_nz, cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectField(cfldt, dt);
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, -0.5f, _nx*_ny*_nz);
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, 0.5f, _nx*_ny*_nz);
    // update rho
    gpuSolver->copyDeviceToHost(_rhotemp, gpuSolver->x_host, gpuSolver->dv);
    // clamp extrema, clamped new density will be in GPU.dv
    clampExtrema(dt, _rho, _rhotemp);
    _rho.copy(_rhotemp);
    cout << "[ MacCormack Advect Density Done! ]" << endl;

    // advect Temperature
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    cudaMemcpy(gpuSolver->dv, gpuSolver->du, sizeof(float)*_nx*_ny*_nz, cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectField(cfldt, dt);
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, -0.5f, _nx*_ny*_nz);
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, 0.5f, _nx*_ny*_nz);
    // update temperature
    gpuSolver->copyDeviceToHost(_Ttemp, gpuSolver->x_host, gpuSolver->dv);
    // clamp extrema, clamped new temperature will be in GPU.dv
    clampExtrema(dt, _T, _Ttemp);
    _T.copy(_Ttemp);
    cout << "[ MacCormack Advect Temperature Done! ]" << endl;

    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, -dt);
    cudaMemcpy(gpuSolver->u_src, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->dw, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, dt);
    gpuSolver->add(gpuSolver->u_src, gpuSolver->du, -0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v_src, gpuSolver->dv, -0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w_src, gpuSolver->dw, -0.5f, _nx*_ny*(_nz+1));
    gpuSolver->add(gpuSolver->u_src, gpuSolver->u, 0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v_src, gpuSolver->v, 0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w_src, gpuSolver->w, 0.5f, _nx*_ny*(_nz+1));
    // copy velocity back to CPU buffers
    gpuSolver->copyDeviceToHost(_utemp, gpuSolver->u_host, gpuSolver->u_src);
    gpuSolver->copyDeviceToHost(_vtemp, gpuSolver->v_host, gpuSolver->v_src);
    gpuSolver->copyDeviceToHost(_wtemp, gpuSolver->w_host, gpuSolver->w_src);
    // clamp extrema, clamped new velocity will be in GPU.u_src, GPU.v_src, GPU.w_src
    clampExtrema(dt, _un, _utemp);
    clampExtrema(dt, _vn, _vtemp);
    clampExtrema(dt, _wn, _wtemp);
    _un.copy(_utemp);
    _vn.copy(_vtemp);
    _wn.copy(_wtemp);
    cout << "[ MacCormack Advect Velocity Done! ]" << endl;

    clearBoundary(_rho);
    emitSmoke(framenum, dt);
    addBuoyancy(dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    projection();
}

void BimocqSolver::advanceReflection(int framenum, float dt)
{
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    // Reflection advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for advection
    // advect density
    float cfldt = getCFL();
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    cudaMemcpy(gpuSolver->dv, gpuSolver->du, sizeof(float)*_nx*_ny*_nz, cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectField(cfldt, dt);
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, -0.5f, _nx*_ny*_nz);
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, 0.5f, _nx*_ny*_nz);
    // update rho
    gpuSolver->copyDeviceToHost(_rhotemp, gpuSolver->x_host, gpuSolver->dv);
    // clamp extrema, clamped new density will be in GPU.dv
    clampExtrema(dt, _rho, _rhotemp);
    _rho.copy(_rhotemp);
    cout << "[ Reflection Advect Density Done! ]" << endl;

    // advect Temperature
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    cudaMemcpy(gpuSolver->dv, gpuSolver->du, sizeof(float)*_nx*_ny*_nz, cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectField(cfldt, dt);
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, -0.5f, _nx*_ny*_nz);
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->add(gpuSolver->dv, gpuSolver->du, 0.5f, _nx*_ny*_nz);
    // update temperature
    gpuSolver->copyDeviceToHost(_Ttemp, gpuSolver->x_host, gpuSolver->dv);
    // clamp extrema, clamped new temperature will be in GPU.dv
    clampExtrema(dt, _T, _Ttemp);
    _T.copy(_Ttemp);
    cout << "[ Reflection Advect Temperature Done! ]" << endl;

    clearBoundary(_rho);

    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, -0.5f*dt);
    cudaMemcpy(gpuSolver->u_src, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->dw, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, 0.5f*dt);
    gpuSolver->add(gpuSolver->u_src, gpuSolver->du, -0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v_src, gpuSolver->dv, -0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w_src, gpuSolver->dw, -0.5f, _nx*_ny*(_nz+1));
    gpuSolver->add(gpuSolver->u_src, gpuSolver->u, 0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v_src, gpuSolver->v, 0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w_src, gpuSolver->w, 0.5f, _nx*_ny*(_nz+1));
    // copy velocity back to CPU buffers
    gpuSolver->copyDeviceToHost(_utemp, gpuSolver->u_host, gpuSolver->u_src);
    gpuSolver->copyDeviceToHost(_vtemp, gpuSolver->v_host, gpuSolver->v_src);
    gpuSolver->copyDeviceToHost(_wtemp, gpuSolver->w_host, gpuSolver->w_src);
    // clamp extrema, clamped new velocity will be in GPU.u_src, GPU.v_src, GPU.w_src
    clampExtrema(0.5f*dt, _un, _utemp);
    clampExtrema(0.5f*dt, _vn, _vtemp);
    clampExtrema(0.5f*dt, _wn, _wtemp);
    _un.copy(_utemp);
    _vn.copy(_vtemp);
    _wn.copy(_wtemp);
    cout << "[ Reflection Advect Velocity First Half Done! ]" << endl;

    emitSmoke(framenum, dt);
    addBuoyancy(0.5f*dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(0.5f*dt, viscosity, _un);
        diffuse_field(0.5f*dt, viscosity, _vn);
        diffuse_field(0.5f*dt, viscosity, _wn);
    }

    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    projection();
    _duproj.copy(_un);
    _dvproj.copy(_vn);
    _dwproj.copy(_wn);
    _duproj *= 2.f;
    _dvproj *= 2.f;
    _dwproj *= 2.f;
    _duproj -= _utemp;
    _dvproj -= _vtemp;
    _dwproj -= _wtemp;

    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    gpuSolver->copyHostToDevice(_duproj, gpuSolver->u_host, gpuSolver->u_src, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_dvproj, gpuSolver->v_host, gpuSolver->v_src, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_dwproj, gpuSolver->w_host, gpuSolver->w_src, _nx*_ny*(_nz+1)*sizeof(float));
    gpuSolver->semilagAdvectVelocity(cfldt, -0.5f*dt);
    cudaMemcpy(gpuSolver->u_src, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->dw, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, 0.5f*dt);
    gpuSolver->add(gpuSolver->u_src, gpuSolver->du, -0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v_src, gpuSolver->dv, -0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w_src, gpuSolver->dw, -0.5f, _nx*_ny*(_nz+1));
    gpuSolver->copyHostToDevice(_duproj, gpuSolver->u_host, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_dvproj, gpuSolver->v_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_dwproj, gpuSolver->w_host, gpuSolver->dw, _nx*_ny*(_nz+1)*sizeof(float));
    gpuSolver->add(gpuSolver->u_src, gpuSolver->du, 0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v_src, gpuSolver->dv, 0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w_src, gpuSolver->dw, 0.5f, _nx*_ny*(_nz+1));
    // copy velocity back to CPU buffers
    gpuSolver->copyDeviceToHost(_utemp, gpuSolver->u_host, gpuSolver->u_src);
    gpuSolver->copyDeviceToHost(_vtemp, gpuSolver->v_host, gpuSolver->v_src);
    gpuSolver->copyDeviceToHost(_wtemp, gpuSolver->w_host, gpuSolver->w_src);
    // clamp extrema, clamped new velocity will be in GPU.u_src, GPU.v_src, GPU.w_src
    clampExtrema(0.5f*dt, _un, _utemp);
    clampExtrema(0.5f*dt, _vn, _vtemp);
    clampExtrema(0.5f*dt, _wn, _wtemp);
    _un.copy(_utemp);
    _vn.copy(_vtemp);
    _wn.copy(_wtemp);
    cout << "[ Reflection Advect Velocity Second Half Done! ]" << endl;

    addBuoyancy(0.5f*dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(0.5f*dt, viscosity, _un);
        diffuse_field(0.5f*dt, viscosity, _vn);
        diffuse_field(0.5f*dt, viscosity, _wn);
    }

    projection();
}

void BimocqSolver::diffuse_field(double dt, double nu, buffer3Df &field)
{
    buffer3Df field_temp;
    field_temp.init(field._nx, field._ny, field._nz, field._hx, 0,0,0);
    field_temp.setZero();
    field_temp.copy(field);
    int compute_elements = field_temp._blockx*field_temp._blocky*field_temp._blockz;

    int slice = field_temp._blockx*field_temp._blocky;
    double coef = nu*(dt/(_h*_h));

    for(int iter = 0; iter<20; iter++) {
        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            int bk = thread_idx / slice;
            int bj = (thread_idx % slice) / field_temp._blockx;
            int bi = thread_idx % (field_temp._blockx);

            for (int kk = 0; kk < 8; kk++)
                for (int jj = 0; jj < 8; jj++)
                    for (int ii = 0; ii < 8; ii++) {
                        int i = bi * 8 + ii, j = bj * 8 + jj, k = bk * 8 + kk;
                        if((i+j+k)%2==0)
                            field_temp(i, j, k) = (field(i, j, k) + coef * (
                                     field_temp.at(i - 1, j, k) +
                                     field_temp.at(i + 1, j, k) +
                                     field_temp.at(i, j - 1, k) +
                                     field_temp.at(i, j + 1, k) +
                                     field_temp.at(i, j, k - 1) +
                                     field_temp.at(i, j, k + 1)
                            )) / (1.0f + 6.0f * coef);
                    }
        });
        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            int bk = thread_idx / slice;
            int bj = (thread_idx % slice) / field_temp._blockx;
            int bi = thread_idx % (field_temp._blockx);

            for (int kk = 0; kk < 8; kk++)
                for (int jj = 0; jj < 8; jj++)
                    for (int ii = 0; ii < 8; ii++) {
                        int i = bi * 8 + ii, j = bj * 8 + jj, k = bk * 8 + kk;
                        if((i+j+k)%2==1)
                            field_temp(i, j, k) = (field(i, j, k) + coef * (
                                 field_temp.at(i - 1, j, k) +
                                 field_temp.at(i + 1, j, k) +
                                 field_temp.at(i, j - 1, k) +
                                 field_temp.at(i, j + 1, k) +
                                 field_temp.at(i, j, k - 1) +
                                 field_temp.at(i, j, k + 1)
                            )) / (1.0f + 6.0f * coef);
                    }
        });
    }
    field.copy(field_temp);
    field_temp.free();
}

void BimocqSolver::clampExtrema(float dt, buffer3Df &f_n, buffer3Df &f_np1)
{
    int compute_elements = f_np1._blockx*f_np1._blocky*f_np1._blockz;
    int slice = f_np1._blockx*f_np1._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/f_np1._blockx;
        uint bi = thread_idx%(f_np1._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<f_np1._nx && j<f_np1._ny && k<f_np1._nz)
                    {

                        float world_x = ((float)i-f_np1._ox)*_h;
                        float world_y = ((float)j-f_np1._oy)*_h;
                        float world_z = ((float)k-f_np1._oz)*_h;
                        //Vec3f pos(world_x,world_y,world_z);
                        //Vec3f trace_pos = trace(dt, pos);
                        float u = _un.sample_linear(world_x,world_y,world_z);
                        float v = _vn.sample_linear(world_x,world_y,world_z);
                        float w = _wn.sample_linear(world_x,world_y,world_z);

                        float px = world_x - 0.5*dt * u, py = world_y - 0.5*dt *v, pz = world_z - 0.5*dt*w;
                        u = _un.sample_linear(px,py,pz);
                        v = _vn.sample_linear(px,py,pz);
                        w = _wn.sample_linear(px,py,pz);

                        px = world_x - dt * u, py = world_y - dt *v, pz = world_z - dt*w;

                        float v0,v1,v2,v3,v4,v5,v6,v7;
                        //f_n.sample_cube(px,py,pz,v0,v1,v2,v3,v4,v5,v6,v7);
                        float SLv = f_n.sample_cube_lerp(px,py,pz,
                                                         v0,v1,v2,v3,v4,v5,v6,v7);

                        float min_value = min(v0,min(v1,min(v2,min(v3,min(v4,min(v5,min(v6,v7)))))));
                        float max_value = max(v0,max(v1,max(v2,max(v3,max(v4,max(v5,max(v6,v7)))))));

                        if(f_np1(i,j,k)<min_value || f_np1(i,j,k)>max_value)
                        {
                            f_np1(i,j,k) = SLv;
                        }
                        //f_np1(i,j,k) = max(min(max_value, f_np1(i,j,k)),min_value);
                    }
                }
    });
}

void BimocqSolver::semilagAdvect(float cfldt, float dt)
{
    // NOTE: TO SAVE TRANSFER TIME, NEED U,V,W BE STORED IN GPU.U, GPU.V, GPU.W ALREADY
    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_utemp, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(_vtemp, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(_wtemp, gpuSolver->w_host, gpuSolver->dw);
    // semi-lagrangian advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for semi-lagrangian advection
    // advect density
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_rhotemp, gpuSolver->x_host, gpuSolver->du);
    // advect Temperature
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_Ttemp, gpuSolver->x_host, gpuSolver->du);
}

void BimocqSolver::emitSmoke(int framenum, float dt)
{
    for(auto &emitter : sim_emitter)
    {
        emitter.update(framenum, _h, dt);
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
        if(framenum < emitter.emitFrame)
        {
//            float in_value = -emitter.e_sdf->background();

            int compute_elements = _rho._blockx*_rho._blocky*_rho._blockz;
            int slice = _rho._blockx*_rho._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_rho._blockx;
                uint bi = thread_idx%(_rho._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<_rho._nx && j<_rho._ny && k<_rho._nz)
                    {
                        float w_x = ((float)i-_rho._ox)*_h;
                        float w_y = ((float)j-_rho._oy)*_h;
                        float w_z = ((float)k-_rho._oz)*_h;
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _rho(i,j,k) = emitter.emit_density;
                            _T(i,j,k) = emitter.emit_temperature;
                        }
                    }
                }
            });

            compute_elements = _un._blockx*_un._blocky*_un._blockz;
            slice = _un._blockx*_un._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_un._blockx;
                uint bi = thread_idx%(_un._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;

                    if(i<_un._nx && j<_un._ny && k<_un._nz)
                    {
                        float w_x = ((float)i-_un._ox)*_h;
                        float w_y = ((float)j-_un._oy)*_h;
                        float w_z = ((float)k-_un._oz)*_h;
                        Vec3f world_pos(w_x, w_y, w_z);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _un(i,j,k) = emitter.emit_velocity(world_pos)[0];
                        }
                    }
                }
            });

            compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
            slice = _vn._blockx*_vn._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_vn._blockx;
                uint bi = thread_idx%(_vn._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                    {
                        float w_x = ((float)i-_vn._ox)*_h;
                        float w_y = ((float)j-_vn._oy)*_h;
                        float w_z = ((float)k-_vn._oz)*_h;
                        Vec3f world_pos(w_x, w_y, w_z);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _vn(i,j,k) = emitter.emit_velocity(world_pos)[1];
                        }
                    }
                }
            });

            compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
            slice = _wn._blockx*_wn._blocky;

            tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                uint bk = thread_idx/slice;
                uint bj = (thread_idx%slice)/_wn._blockx;
                uint bi = thread_idx%(_wn._blockx);

                for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                    {
                        float w_x = ((float)i-_wn._ox)*_h;
                        float w_y = ((float)j-_wn._oy)*_h;
                        float w_z = ((float)k-_wn._oz)*_h;
                        Vec3f world_pos(w_x, w_y, w_z);
                        float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                        if (sdf_value <= 0)
                        {
                            _wn(i,j,k) = emitter.emit_velocity(world_pos)[2];
                        }
                    }
                }
            });
        }
    }
}

void BimocqSolver::addBuoyancy(float dt)
{
    int compute_elements = _rho._blockx*_rho._blocky*_rho._blockz;
    int slice = _rho._blockx*_rho._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/_rho._blockx;
        uint bi = thread_idx%(_rho._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<_nx && j<_ny && k<_nz)
            {
                float density = _rho(i,j,k);
                float temperature = _T(i,j,k);
                float f = -dt*_alpha*density + dt*_beta*temperature;

                _vn(i,j,k) += 0.5*f;
            }
        }
    });
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/_rho._blockx;
        uint bi = thread_idx%(_rho._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<_nx && j>0&&j<_ny && k<_nz)
            {
                float density = _rho(i,j,k);
                float temperature = _T(i,j,k);
                float f = -dt*_alpha*density + dt*_beta*temperature;

                _vn(i,j+1,k) += 0.5*f;
            }
        }
    });
}

void BimocqSolver::setSmoke(float drop, float raise, const std::vector<Emitter> &emitters)
{
    _alpha = drop;
    _beta = raise;
    sim_emitter = emitters;
}

void BimocqSolver::blendBoundary(buffer3Df &field, const buffer3Df &blend_field)
{
    for(const auto &boundary : sim_boundary)
    {
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*boundary.b_sdf);

        int compute_elements = field._blockx*field._blocky*field._blockz;
        int slice = field._blockx*field._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/field._blockx;
            uint bi = thread_idx%(field._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<field._nx && j<field._ny && k<field._nz)
                {
                    float w_x = ((float)i-field._ox)*_h;
                    float w_y = ((float)j-field._oy)*_h;
                    float w_z = ((float)k-field._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    float background_value = boundary.b_sdf->background();
                    if (sdf_value > 0.f && sdf_value < background_value)
                    {
                        field(i,j,k) = blend_field(i,j,k);
                    }
                }
            }
        });
    }
}

void BimocqSolver::clearBoundary(buffer3Df field)
{
    int compute_elements = _rho._blockx*_rho._blocky*_rho._blockz;

    int slice = _rho._blockx*_rho._blocky;
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/_rho._blockx;
        uint bi = thread_idx%(_rho._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(_b_desc(i,j,k)==3)
            {
                field(i,j,k) = 0;
            }
        }
    });
}

void BimocqSolver::updateBoundary(int framenum, float dt)
{
    _b_desc.setZero();
    for (int k=0;k<_nz;k++)for(int j=0;j<_ny;j++)for(int i=0;i<_nx;i++)
    {
        //0:fluid;1:air;2:solid
        if(i<1) _b_desc(i,j,k) = 2;
        if(j<1) _b_desc(i,j,k) = 2;
        if(k<1) _b_desc(i,j,k) = 2;

        if(i>=_nx-1) _b_desc(i,j,k) = 2;
        if(j>=_ny-1) _b_desc(i,j,k) = 1;
        if(k>=_nz-1) _b_desc(i,j,k) = 2;
    }
    for(auto &boundary : sim_boundary)
    {
        Vec3f boundary_vel = boundary.vel_func(framenum);
        boundary.update(framenum, _h, dt);
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*boundary.b_sdf);

        int compute_elements = _b_desc._blockx*_b_desc._blocky*_b_desc._blockz;
        int slice = _b_desc._blockx*_b_desc._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_b_desc._blockx;
            uint bi = thread_idx%(_b_desc._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_b_desc._nx && j<_b_desc._ny && k<_b_desc._nz)
                {
                    float w_x = ((float)i-_b_desc._ox)*_h;
                    float w_y = ((float)j-_b_desc._oy)*_h;
                    float w_z = ((float)k-_b_desc._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0.f)
                    {
                        _b_desc(i,j,k) = 3;
                    }
                }
            }
        });

        compute_elements = _un._blockx*_un._blocky*_un._blockz;
        slice = _un._blockx*_un._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_un._blockx;
            uint bi = thread_idx%(_un._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_un._nx && j<_un._ny && k<_un._nz)
                {
                    float w_x = ((float)i-_un._ox)*_h;
                    float w_y = ((float)j-_un._oy)*_h;
                    float w_z = ((float)k-_un._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _usolid(i,j,k) = boundary_vel[0];
                    }
                }
            }
        });

        compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
        slice = _vn._blockx*_vn._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_vn._blockx;
            uint bi = thread_idx%(_vn._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                {
                    float w_x = ((float)i-_vn._ox)*_h;
                    float w_y = ((float)j-_vn._oy)*_h;
                    float w_z = ((float)k-_vn._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _vsolid(i,j,k) = boundary_vel[1];
                    }
                }
            }
        });

        compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
        slice = _wn._blockx*_wn._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_wn._blockx;
            uint bi = thread_idx%(_wn._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                {
                    float w_x = ((float)i-_wn._ox)*_h;
                    float w_y = ((float)j-_wn._oy)*_h;
                    float w_z = ((float)k-_wn._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _wsolid(i,j,k) = boundary_vel[2];
                    }
                }
            }
        });
    }
}

void BimocqSolver::setBoundary(const std::vector<Boundary> &boundaries)
{
    sim_boundary = boundaries;
}

float BimocqSolver::getCFL()
{
    max_v = 1e-4;
    for (uint k=0; k<_nz;k++) for (uint j=0; j<_ny;j++) for (uint i=0; i<_nx+1;i++)
    {
        if (fabs(_un(i,j,k))>max_v)
        {
            max_v = fabs(_un(i,j,k));
        }
    }
    for (uint k=0; k<_nz;k++) for (uint j=0; j<_ny+1;j++) for (uint i=0; i<_nx;i++)
    {
        if (fabs(_vn(i,j,k))>max_v)
        {
            max_v = fabs(_vn(i,j,k));
        }
    }
    for (uint k=0; k<_nz+1;k++) for (uint j=0; j<_ny;j++) for (uint i=0; i<_nx;i++)
    {
        if (fabs(_wn(i,j,k))>max_v)
        {
            max_v = fabs(_wn(i,j,k));
        }
    }
    return _h / max_v;
}

void BimocqSolver::projection()
{
    int ni = _nx;
    int nj = _ny;
    int nk = _nz;

    int system_size = ni*nj*nk;
    if(rhs.size() != system_size) {
        rhs.resize(system_size);
        pressure.resize(system_size);
        matrix.resize(system_size);
    }

    matrix.zero();
    rhs.assign(rhs.size(), 0);
    pressure.assign(pressure.size(), 0);
    //write boundary velocity;
    int compute_num = ni*nj*nk;
    int slice = ni*nj;
    tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
    {
        int k = thread_idx/slice;
        int j = (thread_idx%slice)/ni;
        int i = thread_idx%ni;
        if ( _b_desc(i,j,k)==2 || _b_desc(i,j,k)==3)//solid
        {
            _un(i,j,k) = _usolid(i,j,k);
            _un(i+1,j,k) = _usolid(i+1,j,k);
            _vn(i,j,k) = _vsolid(i,j,k);
            _vn(i,j+1,k) = _vsolid(i,j+1,k);
            _wn(i,j,k) = _wsolid(i,j,k);
            _wn(i,j,k+1) = _wsolid(i,j,k+1);
        }
    });


    //set up solver
    compute_num = ni*nj*nk;
    slice = ni*nj;
    tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
    {
        int k = thread_idx/slice;
        int j = (thread_idx%slice)/ni;
        int i = thread_idx%ni;
        if(i>=1 && i<ni-1 && j>=1 && j<nj-1 && k>=1 && k<nk-1)
        {
            int index = i + ni*j + ni*nj*k;

            rhs[index] = 0;
            pressure[index] = 0;

            if( _b_desc(i,j,k)==0 )//a fluid cell
            {

                //right neighbour
                if( _b_desc(i+1,j,k)==0 ) {//a fluid cell
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                    matrix.add_to_element(index, index + 1, -1.0/_h/_h);
                }
                else if( _b_desc(i+1,j,k)==1 )//an empty cell
                {
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                }
                rhs[index] -= _un(i+1,j,k) / _h;

                //left neighbour
                if( _b_desc(i-1,j,k)==0 ) {
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                    matrix.add_to_element(index, index - 1, -1.0/_h/_h);
                }
                else if( _b_desc(i-1,j,k)==1 ){

                    matrix.add_to_element(index, index, 1.0/_h/_h);
                }
                rhs[index] += _un(i,j,k) / _h;

                //top neighbour
                if( _b_desc(i,j+1,k)==0 ) {//a fluid cell
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                    matrix.add_to_element(index, index + ni, -1.0/_h/_h);
                }
                else if( _b_desc(i,j+1,k)==1 )//an empty cell
                {
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                }
                rhs[index] -= _vn(i,j+1,k) / _h;

                //bottom neighbour
                if( _b_desc(i,j-1,k)==0 ) {
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                    matrix.add_to_element(index, index - ni, -1.0/_h/_h);
                }
                else if( _b_desc(i,j-1,k)==1 ){

                    matrix.add_to_element(index, index, 1.0/_h/_h);
                }
                rhs[index] += _vn(i,j,k) / _h;
                //rhs[index] += _burn_div(i,j,k);



                //back neighbour
                if( _b_desc(i,j,k+1)==0 ) {//a fluid cell
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                    matrix.add_to_element(index, index + ni*nj, -1.0/_h/_h);
                }
                else if( _b_desc(i,j,k+1)==1 )//an empty cell
                {
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                }
                rhs[index] -= _wn(i,j,k+1) / _h;

                //front neighbour
                if( _b_desc(i,j,k-1)==0 ) {
                    matrix.add_to_element(index, index, 1.0/_h/_h);
                    matrix.add_to_element(index, index - ni*nj, -1.0/_h/_h);
                }
                else if( _b_desc(i,j,k-1)==1 ){

                    matrix.add_to_element(index, index, 1.0/_h/_h);
                }
                rhs[index] += _wn(i,j,k) / _h;


                //rhs[index] += _burn_div(i,j,k);

            }
        }
    });

    //Solve the system using a AMGPCG solver

    double tolerance;
    int iterations;
    //solver.set_solver_parameters(1e-6, 1000);
    //bool success = solver.solve(matrix, rhs, pressure, tolerance, iterations);
    bool success = AMGPCGSolve(matrix,rhs,pressure,1e-6,1000,tolerance,iterations,_nx,_ny,_nz);

    printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
    if(!success) {
        printf("WARNING: Pressure solve failed!************************************************\n");
    }

    //apply grad
    u_valid.assign(0);
    compute_num = _un._nx*_un._ny*_un._nz;
    slice = _un._nx*_un._ny;
    tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_un._nx;
        uint i = thread_idx%_un._nx;
        if(k<_un._nz && j<_un._ny && i<_un._nx-1 && i>0)
        {
            int index = i + j*ni + k*ni*nj;
            if(_b_desc(i,j,k) == 0 || _b_desc(i-1,j,k) == 0) {

                _un(i,j,k) -=  (float)(pressure[index] - pressure[index-1]) / _h ;
                u_valid(i,j,k) = 1;
            }

        }
    });

    v_valid.assign(0);
    compute_num = _vn._nx*_vn._ny*_vn._nz;
    slice = _vn._nx*_vn._ny;
    tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_vn._nx;
        uint i = thread_idx%_vn._nx;
        if(k<_vn._nz && j>0 && j<_vn._ny-1 && i<_vn._nx )
        {
            int index = i + j*ni + k*ni*nj;
            if(_b_desc(i,j,k) == 0 || _b_desc(i,j-1,k) == 0) {

                _vn(i,j,k) -=  (float)(pressure[index] - pressure[index-ni]) / _h ;
                v_valid(i,j,k) = 1;
            }

        }
    });

    w_valid.assign(0);
    compute_num = _wn._nx*_wn._ny*_wn._nz;
    slice = _wn._nx*_wn._ny;
    tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_wn._nx;
        uint i = thread_idx%_wn._nx;
        if(k>0 && k<_wn._nz-1 && j<_wn._ny && i<_wn._nx )
        {
            int index = i + j*ni + k*ni*nj;
            if(_b_desc(i,j,k) == 0 || _b_desc(i,j,k-1) == 0) {

                _wn(i,j,k) -=  (float)(pressure[index] - pressure[index-ni*nj]) / _h ;
                w_valid(i,j,k) = 1;
            }

        }
    });
    //write boundary velocity
    compute_num = ni*nj*nk;
    slice = ni*nj;
    tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/ni;
        uint i = thread_idx%ni;
        if ( _b_desc(i,j,k)==2 || _b_desc(i,j,k)==3)//solid
        {
            _un(i,j,k) = _usolid(i,j,k);
            u_valid(i,j,k) = 1;
            _un(i+1,j,k) = _usolid(i+1,j,k);
            u_valid(i+1,j,k) =1;
            _vn(i,j,k) = _vsolid(i,j,k);
            v_valid(i,j,k) = 1;
            _vn(i,j+1,k) = _vsolid(i,j+1,k);
            v_valid(i,j+1,k) = 1;
            _wn(i,j,k) = _wsolid(i,j,k);
            w_valid(i,j,k) = 1;
            _wn(i,j,k+1) = _wsolid(i,j,k+1);
            w_valid(i,j,k+1) =1;
        }
    });

    compute_num = _un._nx*_un._ny*_un._nz;
    slice = _un._nx*_un._ny;
    tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_un._nx;
        uint i = thread_idx%_un._nx;
        if(k<_un._nz&& j<_un._ny && i<_un._nx )
        {
            if(u_valid(i,j,k)==0)
            {
                _un(i,j,k) = 0;
            }
        }
    });

    compute_num = _vn._nx*_vn._ny*_vn._nz;
    slice = _vn._nx*_vn._ny;
    tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_vn._nx;
        uint i = thread_idx%_vn._nx;
        if(k<_vn._nz&& j<_vn._ny && i<_vn._nx )
        {
            if(v_valid(i,j,k)==0)
            {
                _vn(i,j,k) = 0;
            }
        }
    });

    compute_num = _wn._nx*_wn._ny*_wn._nz;
    slice = _wn._nx*_wn._ny;
    tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
    {
        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_wn._nx;
        uint i = thread_idx%_wn._nx;
        if(k<_wn._nz&& j<_wn._ny && i<_wn._nx )
        {
            if(w_valid(i,j,k)==0)
            {
                _wn(i,j,k) = 0;
            }
        }
    });

    //extrapolate(u_extrap,_un,u_valid);
    //extrapolate(v_extrap,_vn,v_valid);
    //extrapolate(w_extrap,_wn,w_valid);
}

void BimocqSolver::outputResult(uint frame, string filepath)
{
    writeVDB(frame, filepath, _h, _rho);
    int boundary_index = 0;
    for (auto &b : sim_boundary)
    {
        char file_name[256];
        sprintf(file_name,"%s/sim_boundary%02d_%04d.obj", filepath.c_str(), boundary_index, frame);
        std::string objname(file_name);

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec4I> quads;
        openvdb::tools::volumeToMesh<openvdb::FloatGrid>(*b.b_sdf, points, quads);
        writeObj(objname, points, quads);
        boundary_index += 1;
    }
}

void BimocqSolver::velocityReinitialize()
{
    _uprev.copy(_uinit);
    _vprev.copy(_vinit);
    _wprev.copy(_winit);
    // set current buffer as next initial buffer
    _uinit.copy(_un);
    _vinit.copy(_vn);
    _winit.copy(_wn);
}

void BimocqSolver::scalarReinitialize()
{
    _rhoprev.copy(_rhoinit);
    _Tprev.copy(_Tinit);
    // set current buffer as next initial buffer
    _rhoinit.copy(_rho);
    _Tinit.copy(_T);
}