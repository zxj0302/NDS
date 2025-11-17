cmake_minimum_required(VERSION 3.26)
project(NEG_DSD)

# Include common configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CommonConfig.cmake)

# Create executable
add_nds_executable(${PROJECT_NAME} main.cpp)

# Add QPBO library
add_qpbo_to_target(${PROJECT_NAME})

# Add Gurobi support
add_gurobi_to_target(${PROJECT_NAME})