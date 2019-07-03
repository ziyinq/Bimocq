find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(Half REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61; -gencode arch=compute_75,code=sm_75;
        -gencode arch=compute_75,code=compute_75; -lineinfo)
set(CUDA_SEPARABLE_COMPILATION ON)

add_library(cuda_lib GPU_Advection.h GPU_kernel.cu)
target_link_libraries(cuda_lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a)

set(SOURCE_FILES
        main.cpp
        GPU_kernel.cu
        Mapping.cpp
        BimocqSolver.cpp
        )

cuda_add_executable(BiMocq3D ${SOURCE_FILES})
target_link_libraries(BiMocq3D  ${TBB_LIBRARIES})
target_link_libraries(BiMocq3D  ${OPENVDB_LIBRARIES})
target_link_libraries(BiMocq3D  Half)
target_link_libraries(BiMocq3D ${Boost_LIBRARIES})
target_link_libraries(BiMocq3D rt ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart_static.a)
