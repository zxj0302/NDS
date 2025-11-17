# FindGUROBI.cmake
# Try to find GUROBI

# Use GUROBI_HOME environment variable if set
if(DEFINED ENV{GUROBI_HOME})
    set(GUROBI_DIR "$ENV{GUROBI_HOME}" CACHE PATH "Gurobi installation directory")
else()
    # Fallback default paths
    if(APPLE)
        set(GUROBI_DIR "/Library/gurobi1300/macos_universal2" CACHE PATH "Gurobi installation directory")
    elseif(UNIX)
        set(GUROBI_DIR "/opt/gurobi1300/linux64" CACHE PATH "Gurobi installation directory")
    elseif(WIN32)
        set(GUROBI_DIR "C:/gurobi1300/win64" CACHE PATH "Gurobi installation directory")
    endif()
endif()

find_path(GUROBI_INCLUDE_DIRS
    NAMES gurobi_c++.h
    HINTS ${GUROBI_DIR}/include
)

find_library(GUROBI_LIBRARY
    NAMES gurobi130 gurobi120 gurobi110 gurobi103 gurobi
    HINTS ${GUROBI_DIR}/lib
    NO_DEFAULT_PATH
)

find_library(GUROBI_CXX_LIBRARY
    NAMES gurobi_c++
    HINTS ${GUROBI_DIR}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG
    GUROBI_LIBRARY GUROBI_CXX_LIBRARY GUROBI_INCLUDE_DIRS)

mark_as_advanced(GUROBI_INCLUDE_DIRS GUROBI_LIBRARY GUROBI_CXX_LIBRARY)