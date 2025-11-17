# Common CMake configuration for all NDS subprojects
# Include this file in each subproject's CMakeLists.txt
# Note: cmake_minimum_required() must be called in each project's CMakeLists.txt before project()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
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

# Helper function to setup a basic executable with Boost
function(add_nds_executable TARGET_NAME SOURCE_FILES)
    add_executable(${TARGET_NAME} ${SOURCE_FILES})
    
    if(Boost_FOUND)
        target_include_directories(${TARGET_NAME} PRIVATE 
            ${Boost_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
        )
    endif()
endfunction()

# Helper function to setup Gurobi for a target
function(add_gurobi_to_target TARGET_NAME)
    # Add CMake module path if not already added
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
    
    # Find Gurobi
    find_package(GUROBI REQUIRED)
    if(GUROBI_FOUND)
        message(STATUS "Gurobi found at: ${GUROBI_DIR}")
        message(STATUS "Gurobi include: ${GUROBI_INCLUDE_DIRS}")
        message(STATUS "Gurobi libraries: ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY}")
        
        target_include_directories(${TARGET_NAME} PRIVATE ${GUROBI_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME} PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
    endif()
endfunction()

# Helper function to add QPBO library
function(add_qpbo_to_target TARGET_NAME)
    # Compile QPBO as a separate object to avoid template instantiation issues
    add_library(qpbo_lib OBJECT 
        ../QPBO/QPBO.cpp
        ../QPBO/QPBO_extra.cpp
        ../QPBO/QPBO_maxflow.cpp
        ../QPBO/QPBO_postprocessing.cpp
    )
    target_include_directories(qpbo_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_sources(${TARGET_NAME} PRIVATE $<TARGET_OBJECTS:qpbo_lib>)
endfunction()
