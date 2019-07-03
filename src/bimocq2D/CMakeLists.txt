find_package(TBB REQUIRED)

add_executable(BiMocq2D main.cpp BimocqSolver2D.cpp ../utils/writeBMP.cpp)
target_link_libraries(BiMocq2D PRIVATE ${TBB_LIBRARIES})