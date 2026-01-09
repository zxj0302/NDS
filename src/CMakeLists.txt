# /Users/zxj/Desktop/NDS/src/CMakeLists.txt
cmake_minimum_required(VERSION 3.26)
project(NDS VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Configuration Options
# ============================================================================
option(BUILD_ALL "Build all executables" ON)
option(BUILD_NEG_DSD "Build neg_dsd executable" ON)
option(BUILD_DCS_GREEDY "Build dcs_greedy executable" ON)
option(BUILD_CEP "Build cep executable" ON)
option(BUILD_EXACT "Build exact executable (requires Gurobi)" ON)
option(BUILD_MIQCP "Build miqcp executable (requires Gurobi)" ON)

option(USE_GUROBI "Enable Gurobi support for MIP-based algorithms" ON)
option(VERBOSE_BUILD "Show detailed build information" OFF)

option(ENABLE_LOG "Enable logging macro ENABLE_LOG" ON)
option(DISABLE_WARNINGS "Disable compiler warnings" OFF)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For IDE integration

# Compiler flags
if(DISABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-w)  # Disable all warnings
    endif()
    if(MSVC)
        add_compile_options(/W0)  # Disable all warnings on MSVC
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "==============================================")
message(STATUS "NDS - Negative Densest Subgraph Algorithms")
message(STATUS "==============================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "==============================================")

# ============================================================================
# Dependencies
# ============================================================================
# Boost (required for all)
find_package(Boost REQUIRED COMPONENTS json CONFIG)
if(Boost_FOUND)
    message(STATUS "✓ Boost found: ${Boost_VERSION}")
endif()

# Gurobi (optional, but required for MIP variants)
if(USE_GUROBI)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(GUROBI)
    if(GUROBI_FOUND)
        message(STATUS "✓ Gurobi found: ${GUROBI_DIR}")
    else()
        message(WARNING "✗ Gurobi not found. MIP-based algorithms will be disabled.")
        set(BUILD_EXACT OFF)
    endif()
endif()

# ============================================================================
# QPBO Library (shared object for QPBO-based algorithms)
# ============================================================================
add_library(qpbo_lib OBJECT
    external/QPBO/QPBO.cpp
    external/QPBO/QPBO_extra.cpp
    external/QPBO/QPBO_maxflow.cpp
    external/QPBO/QPBO_postprocessing.cpp
)
target_include_directories(qpbo_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(qpbo_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Build Executables
# ============================================================================
set(BUILD_COUNT 0)

# Helper function to conditionally build executables
function(add_nds_algorithm NAME REQUIRES_GUROBI REQUIRES_QPBO)
    string(TOUPPER ${NAME} NAME_UPPER)
    string(REPLACE "_" "" NAME_CLEAN ${NAME})
    
    if(BUILD_ALL OR BUILD_${NAME_UPPER})
        # Check dependencies
        if(${REQUIRES_GUROBI} AND NOT GUROBI_FOUND)
            message(STATUS "⊘ Skipping ${NAME}: Gurobi not found")
            return()
        endif()
        
        # Build the executable
        add_executable(${NAME} executables/${NAME}.cpp)

        # Compiler flags
        if(ENABLE_LOG)
            target_compile_definitions(${NAME} PRIVATE ENABLE_LOG)
        endif()
        
        # Include directories
        target_include_directories(${NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${Boost_INCLUDE_DIRS}
        )
        
        # Link Boost libraries
        find_library(BOOST_JSON_LIBRARY boost_json PATHS /opt/homebrew/lib /usr/local/lib REQUIRED)
        target_link_libraries(${NAME} PRIVATE ${BOOST_JSON_LIBRARY})
        
        # Link QPBO if needed
        if(${REQUIRES_QPBO})
            target_sources(${NAME} PRIVATE $<TARGET_OBJECTS:qpbo_lib>)
        endif()
        
        # Link Gurobi if needed
        if(${REQUIRES_GUROBI} AND GUROBI_FOUND)
            target_include_directories(${NAME} PRIVATE ${GUROBI_INCLUDE_DIRS})
            target_link_libraries(${NAME} PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
        endif()
        
        # Set output directory to /Users/zxj/Desktop/NDS/build
        set_target_properties(${NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build
        )
        
        math(EXPR BUILD_COUNT "${BUILD_COUNT}+1")
        set(BUILD_COUNT ${BUILD_COUNT} PARENT_SCOPE)
        message(STATUS "✓ Building ${NAME}")
    else()
        message(STATUS "⊘ Skipping ${NAME}: disabled")
    endif()
endfunction()

# Add all algorithms (using directory names)
add_nds_algorithm(neg_dsd FALSE FALSE)
add_nds_algorithm(dcs_greedy FALSE FALSE)
add_nds_algorithm(cep FALSE FALSE)
add_nds_algorithm(exact TRUE TRUE)
add_nds_algorithm(miqcp TRUE FALSE)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "==============================================")
message(STATUS "Configuration Summary:")
message(STATUS "  Total executables: ${BUILD_COUNT}")
message(STATUS "  Gurobi support: ${USE_GUROBI}")
if(BUILD_COUNT EQUAL 0)
    message(WARNING "No executables will be built!")
endif()
message(STATUS "==============================================")

# ============================================================================
# Custom Targets
# ============================================================================
# Clean all builds
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/../build
    COMMENT "Cleaning build directory"
)

# Install target (optional)
# Note: Targets will be installed based on which ones were actually built
if(TARGET neg_dsd)
    install(TARGETS neg_dsd RUNTIME DESTINATION bin)
endif()
if(TARGET dcs_greedy)
    install(TARGETS dcs_greedy RUNTIME DESTINATION bin)
endif()
if(TARGET cep)
    install(TARGETS cep RUNTIME DESTINATION bin)
endif()
if(TARGET exact)
    install(TARGETS exact RUNTIME DESTINATION bin)
endif()
if(TARGET miqcp)
    install(TARGETS miqcp RUNTIME DESTINATION bin)
endif()