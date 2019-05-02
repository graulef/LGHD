cmake_minimum_required(VERSION 3.6)
project(lghd_catkin)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenCV REQUIRED)
list(APPEND ThirdParty_LIBS ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

# TODO: Add FFTW as a working and required dependecy such that throws error during build process
find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS})


#############
# LIBRARIES #
#############
add_definitions(-std=c++11)
add_definitions(-DENABLE_STATISTICS=1)
add_definitions(-DENABLE_TIMING=1)

if(APPLE)
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
endif()
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")
add_definitions(-std=c++11)

cs_add_library(${PROJECT_NAME}_lib
 src/image_io.h
 src/math_functions.cc
 src/log_gabor_filter_bank.cc
 src/phase_congruency.cc
 src/lghd.cc
)

############
# BINARIES #
############

cs_add_executable(${PROJECT_NAME}
 src/main.cc
)

# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME} ${ITK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GSL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} fftw3f fftw3f_threads)
target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES})


#############
# QTCREATOR #
#############
FILE(GLOB_RECURSE LibFiles "include/*h" "config/*.yaml" "launch/*.launch" "rviz/*.rviz")
add_custom_target(headers SOURCES ${LibFiles})

##########
# EXPORT #
##########
cs_install()
cs_export()
