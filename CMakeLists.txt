cmake_minimum_required(VERSION 3.10)
project(sndTridentCrossCheck CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set global output directories for libraries and executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find ROOT
find_package(ROOT REQUIRED COMPONENTS Core Tree Hist RIO)
include(${ROOT_USE_FILE})

# Check that SNDSW_ROOT is set
if(NOT DEFINED ENV{SNDSW_ROOT})
  message(FATAL_ERROR "SNDSW_ROOT environment variable is not set!")
endif()
set(SNDSW_ROOT $ENV{SNDSW_ROOT})

# Setup base directory of the software stack
get_filename_component(SNDSW_PARENT ${SNDSW_ROOT} DIRECTORY)
get_filename_component(SND_BASE ${SNDSW_PARENT} DIRECTORY)

# Setup include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/cuts
  ${SNDSW_ROOT}/include
  ${ROOT_INCLUDE_DIRS}
)

# Auto-detect FairRoot include directory
file(GLOB FAIRROOT_DIRS "${SND_BASE}/FairRoot/*")
if(FAIRROOT_DIRS)
  list(GET FAIRROOT_DIRS 0 FAIRROOT_PATH)
  include_directories(${FAIRROOT_PATH}/include)
  message(STATUS "Auto-detected FairRoot include: ${FAIRROOT_PATH}/include")
endif()

# Auto-detect VMC include directory
file(GLOB VMC_DIRS "${SND_BASE}/VMC/*")
if(VMC_DIRS)
  list(GET VMC_DIRS 0 VMC_PATH)
  include_directories(${VMC_PATH}/include/vmc)
  message(STATUS "Auto-detected VMC include: ${VMC_PATH}/include/vmc")
endif()

# Auto-detect fmt include directory
file(GLOB FMT_DIRS "${SND_BASE}/fmt/*")
if(FMT_DIRS)
  list(GET FMT_DIRS 0 FMT_PATH)
  include_directories(${FMT_PATH}/include)
  message(STATUS "Auto-detected fmt include: ${FMT_PATH}/include")
endif()

# Auto-detect FairLogger include directory
file(GLOB FAIRLOGGER_DIRS "${SND_BASE}/FairLogger/*")
if(FAIRLOGGER_DIRS)
  list(GET FAIRLOGGER_DIRS 0 FAIRLOGGER_PATH)
  include_directories(${FAIRLOGGER_PATH}/include)
  message(STATUS "Auto-detected FairLogger include: ${FAIRLOGGER_PATH}/include")
endif()

# Link directories
link_directories(
  ${SNDSW_ROOT}/lib
  ${ROOT_LIBRARY_DIRS}
)

add_subdirectory(cuts)

# Find sndsw dependencies
find_library(ANALYSIS_TOOLS_LIB snd_analysis_tools PATHS "${SNDSW_ROOT}/lib" REQUIRED)
find_library(SHIP_DATA_LIB ShipData PATHS "${SNDSW_ROOT}/lib" REQUIRED)

# Add executable
add_executable(tridentSample tridentSample.cxx)
target_link_libraries(tridentSample
  PRIVATE
  trident_cuts
  ${ANALYSIS_TOOLS_LIB}
  ${SHIP_DATA_LIB}
  ${ROOT_LIBRARIES}
)

# Set executable properties
set_target_properties(tridentSample PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
