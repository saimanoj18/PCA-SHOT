cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(pca_shot)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")



# Adding PCL library
set(PCL_DIR "/home/sai/workspace/pcl-pcl-1.7.2/build/")
find_package(PCL 1.7.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})




###########################SHOT################################


add_executable (PCA_SHOT pcashot/PCA_SHOT.cpp)
target_link_libraries (PCA_SHOT ${PCL_LIBRARIES})


###############################ROPS#############################

add_executable (pca_rops pcarops/pca_rops.cpp)
target_link_libraries (pca_rops ${PCL_LIBRARIES})



###################################FPFH#########################

add_executable (pca_fpfh pcafpfh/pca_fpfh.cpp)
target_link_libraries (pca_fpfh ${PCL_LIBRARIES})







