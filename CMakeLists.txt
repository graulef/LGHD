cmake_minimum_required(VERSION 2.8.3)
project(lghd_catkin)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
add_definitions(-std=c++11)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

find_package(FFTW3f REQUIRED)
include_directories(${FFTW3f_INCLUDE_DIRS})

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}_lib
 src/image_io.h
 src/math_functions.cc
 src/log_gabor_filter_bank.cc
 src/phase_congruency.cc
 src/lghd.cc
)


# TODO: Remove building main, kept for debugging
cs_add_executable(${PROJECT_NAME}
 src/main.cc
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS}})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME} ${ITK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GSL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} fftw3f fftw3f_threads)
target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES})


#############
# QTCREATOR #
#############
FILE(GLOB_RECURSE LibFiles "include/*h")
add_custom_target(headers SOURCES ${LibFiles})
  
##########
# EXPORT #
##########
cs_install()
cs_export()
