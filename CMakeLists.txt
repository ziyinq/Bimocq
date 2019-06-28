cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

project(BiMocq CXX CUDA)
message(STATUS "${CMAKE_BUILD_TYPE} Build")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DEIGEN_INITIALIZE_MATRICES_BY_NAN ")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # using regular Clang or AppleClang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -Wno-unused-parameter -march=native")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -Wno-unused-parameter -Wcast-align -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -march=native -fno-math-errno")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,defs")

    # enable link time optimization
    # We need two wrappers in order for LTO to work properly:
    # - gcc-ar: static library archiver
    # - gcc-ranlib: static library indexer
    # Without these wrappers, all sorts of undefined refernce errors
    # occur in gcc-4.9 due to "slim" LTO objects, and possibly
    # earlier versions for various reasons.
    if("${CMAKE_AR}" MATCHES "gcc-ar$")
        # Already using the gcc-ar wrapper.
        set(GCC_WRAPPER_AR "${CMAKE_AR}")
    else()
        # Replace ar with gcc-ar.
        string(REGEX REPLACE "ar$" "gcc-ar" GCC_WRAPPER_AR "${CMAKE_AR}")
    endif()
    if("${CMAKE_RANLIB}" MATCHES "gcc-ranlib$")
        # Already using the gcc-ranlib wrapper.
        set(GCC_WRAPPER_RANLIB "${CMAKE_RANLIB}")
    else()
        # Replace ranlib with gcc-ranlib.
        string(REGEX REPLACE "ranlib$" "gcc-ranlib" GCC_WRAPPER_RANLIB "${CMAKE_RANLIB}")
    endif()

    if(EXISTS "${GCC_WRAPPER_AR}" AND EXISTS "${GCC_WRAPPER_RANLIB}")
        # Found gcc binutils wrappers.
        set(CMAKE_AR "${GCC_WRAPPER_AR}")
        set(CMAKE_RANLIB "${GCC_WRAPPER_RANLIB}")
        set(HAS_BINUTILS_WRAPPERS 1)
    else()
        message(FATAL_ERROR "gcc binutils wrappers not found; cannot enable LTO.")
    endif()

    if(HAS_BINUTILS_WRAPPERS AND CMAKE_BUILD_TYPE MATCHES RELEASE)
        set(JIXIE_CXXFLAGS_LTO "-flto=8")
        set(JIXIE_LDFLAGS_LTO "${JIXIE_CXXFLAGS_LTO} -fuse-linker-plugin")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=all -Wextra -Wno-unused-parameter -march=native -std=c++14")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zm1024")
    add_definitions(-DNOMINMAX)
endif()


# Add new build types
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} --coverage")
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} --coverage")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "")
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "")

set(CMAKE_CXX_FLAGS_QUICK "${CMAKE_CXX_FLAGS} -O1")
set(CMAKE_EXE_LINKER_FLAGS_QUICK "")
set(CMAKE_SHARED_LINKER_FLAGS_QUICK "")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release
            CACHE STRING "Choose the type of build : Debug Release RelWithDebInfo MinSizeRel Coverage Quick."
            FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel" "Coverage" "Quick")
endif()

find_program(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full --error-exitcode=1 --suppressions=${CMAKE_CURRENT_SOURCE_DIR}/Scripts/valgrind.supp")

# options can be changed with ccmake or cmake-gui called on the build directory
# provided in ubuntu packages with cmake-curses-gui and cmake-qt-gui respectively
option(CREATE_LOCAL_MAKEFILES "Create Makefiles in the source directory to allow building from source" ON)

enable_testing()

add_subdirectory(src)

# build timer
# set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
