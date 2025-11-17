cmake_minimum_required(VERSION 3.26)
project(GNDS_QPBO_MIP_UD)

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Suppress the FindBoost module warning
cmake_policy(SET CMP0167 NEW)

# Set optimization and debug flags based on compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang")
    # Clang-specific flags (including Apple Clang on macOS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_PROFILE "-O3 -g -fno-omit-frame-pointer -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Using GNU")
    # GNU-specific flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -flto -funroll-loops -ftree-vectorize")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_PROFILE "-O3 -g -fno-omit-frame-pointer -march=native -mtune=native")
else()
    message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Enable IPO/LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if(supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "IPO/LTO enabled")
else()
    message(WARNING "IPO/LTO not supported: ${error}")
endif()

# Handle Boost - use CONFIG mode for modern CMake
find_package(Boost REQUIRED CONFIG)
if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
endif()

# Find Gurobi
find_package(GUROBI REQUIRED)
if(GUROBI_FOUND)
    message(STATUS "Gurobi found at: ${GUROBI_DIR}")
    message(STATUS "Gurobi include: ${GUROBI_INCLUDE_DIRS}")
    message(STATUS "Gurobi libraries: ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY}")
endif()

# Compile QPBO as a separate object to avoid template instantiation issues
add_library(qpbo_lib OBJECT 
    QPBO/QPBO.cpp
    QPBO/QPBO_extra.cpp
    QPBO/QPBO_maxflow.cpp
    QPBO/QPBO_postprocessing.cpp
)
target_include_directories(qpbo_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME} main.cpp $<TARGET_OBJECTS:qpbo_lib>)

if(Boost_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE 
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

# Add Gurobi include directories and link libraries
if(GUROBI_FOUND)
    target_include_directories(GNDS_QPBO_MIP_UD PRIVATE ${GUROBI_INCLUDE_DIRS})
    target_link_libraries(GNDS_QPBO_MIP_UD PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
endif()

# Add QPBO include directory
target_include_directories(GNDS_QPBO_MIP_UD PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Define QPBO as header-only
target_compile_definitions(GNDS_QPBO_MIP_UD PRIVATE)