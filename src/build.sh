#!/bin/bash

# Build script for NDS project
# Build directory is at /Users/zxj/Desktop/NDS/build

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/../build"
SRC_DIR="${SCRIPT_DIR}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Build type: Debug, Release (default), RelWithDebInfo"
    echo "  -c, --clean            Clean build directory before building"
    echo "  -a, --algorithm ALG    Build specific algorithm only (neg_dsd, dcs_greedy, cep, etc.)"
    echo "  -j, --jobs N           Number of parallel jobs (default: auto)"
    echo "  --no-gurobi            Disable Gurobi support"
    echo "  --verbose              Verbose build output"
    echo "  -h, --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # Build all in Release mode"
    echo "  $0 -t Debug                  # Build all in Debug mode"
    echo "  $0 -a cep                    # Build only cep"
    echo "  $0 -c                        # Clean and rebuild"
    echo "  $0 --no-gurobi              # Build without Gurobi"
}

# Default values
BUILD_TYPE="Release"
CLEAN=false
ALGORITHM=""
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
USE_GUROBI=ON
VERBOSE=OFF

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -a|--algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --no-gurobi)
            USE_GUROBI=OFF
            shift
            ;;
        --verbose)
            VERBOSE=ON
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure CMake options
CMAKE_OPTS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DUSE_GUROBI="${USE_GUROBI}"
    -DVERBOSE_BUILD="${VERBOSE}"
)

# Algorithm-specific build
if [ -n "$ALGORITHM" ]; then
    echo -e "${YELLOW}Building only ${ALGORITHM}...${NC}"
    # Convert to uppercase for CMake option
    ALGORITHM_UPPER=$(echo "$ALGORITHM" | tr '[:lower:]' '[:upper:]')
    CMAKE_OPTS+=(
        -DBUILD_ALL=OFF
        -DBUILD_${ALGORITHM_UPPER}=ON
    )
fi

# Configure
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake "${CMAKE_OPTS[@]}" "${SRC_DIR}"

# Build
echo -e "${GREEN}Building with ${JOBS} parallel jobs...${NC}"
cmake --build . --parallel ${JOBS}

echo -e "${GREEN}✓ Build complete!${NC}"
echo ""
echo "Executables are in: ${BUILD_DIR}"
ls -lh "${BUILD_DIR}" | grep -v "^d" | grep -v "^total" || true
