find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

add_executable(BiMocq2D main.cpp BimocqSolver2D.cpp ../utils/writeBMP.cpp)
target_link_libraries(BiMocq2D ${TBB_LIBRARIES})
target_link_libraries(BiMocq2D ${Boost_LIBRARIES})