#!/bin/bash

# Script to compile all C++ projects in src directory
# Usage: ./compile.sh

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the list of projects to compile
PROJECTS=("CEP" "CEP_MIP" "CEP_QPBO" "DCSGreedy" "NEG_DSD")

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting compilation of all projects...${NC}\n"

# Compile each project
for PROJECT in "${PROJECTS[@]}"; do
    PROJECT_DIR="$SCRIPT_DIR/$PROJECT"
    BUILD_DIR="$PROJECT_DIR/build"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Compiling $PROJECT${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Check if project directory exists
    if [ ! -d "$PROJECT_DIR" ]; then
        echo -e "${RED}Warning: Directory $PROJECT_DIR does not exist. Skipping...${NC}\n"
        continue
    fi
    
    # Check if build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${RED}Warning: Build directory $BUILD_DIR does not exist. Skipping...${NC}\n"
        continue
    fi
    
    # Navigate to build directory
    cd "$BUILD_DIR"
    
    # Run Make
    echo "Running Make for $PROJECT..."
    if make; then
        echo -e "${GREEN}Build successful for $PROJECT${NC}\n"
    else
        echo -e "${RED}Build failed for $PROJECT${NC}\n"
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # Return to script directory
    cd "$SCRIPT_DIR"
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Compilation complete!${NC}"
echo -e "${BLUE}========================================${NC}"

# List all compiled executables
echo -e "\n${BLUE}Compiled executables:${NC}"
for PROJECT in "${PROJECTS[@]}"; do
    EXECUTABLE="$SCRIPT_DIR/$PROJECT/build/$PROJECT"
    if [ -f "$EXECUTABLE" ]; then
        echo -e "${GREEN}✓${NC} $EXECUTABLE"
    else
        echo -e "${RED}✗${NC} $EXECUTABLE (not found)"
    fi
done