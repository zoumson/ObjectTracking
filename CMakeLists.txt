cmake_minimum_required (VERSION 3.0)

PROJECT(tracking)

#Set the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


#Set ccache (for speeding up)
find_program(CCACHE_FOUND ccache) 
if(CCACHE_FOUND) 
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache) 
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) 
endif(CCACHE_FOUND) 
set (CMAKE_CXX_FLAGS "-DPCL_ONLY_CORE_POINT_TYPES=ON -DNO_EXPLICIT_INSTANTIATIONS") 

# Requires OpenCV
FIND_PACKAGE( OpenCV 4.0.0 REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

ADD_EXECUTABLE(camshiftTracker  ${PROJECT_SOURCE_DIR}/src/camshiftTracker.cpp )
TARGET_LINK_LIBRARIES( camshiftTracker ${OpenCV_LIBS} )

ADD_EXECUTABLE( coloredObjectTracker ${PROJECT_SOURCE_DIR}/src/coloredObjectTracker.cpp )
TARGET_LINK_LIBRARIES( coloredObjectTracker ${OpenCV_LIBS} )

ADD_EXECUTABLE( farnebackTracker  ${PROJECT_SOURCE_DIR}/src/farnebackTracker.cpp )
TARGET_LINK_LIBRARIES( farnebackTracker ${OpenCV_LIBS} )

ADD_EXECUTABLE( goodFeaturesToTrack ${PROJECT_SOURCE_DIR}/src/goodFeaturesToTrack.cpp )
TARGET_LINK_LIBRARIES( goodFeaturesToTrack ${OpenCV_LIBS} )

ADD_EXECUTABLE( harrisCornersTracker ${PROJECT_SOURCE_DIR}/src/harrisCornersTracker.cpp )
TARGET_LINK_LIBRARIES( harrisCornersTracker ${OpenCV_LIBS} )

ADD_EXECUTABLE( lucasKanadeTracker ${PROJECT_SOURCE_DIR}/src/lucasKanadeTracker.cpp )
TARGET_LINK_LIBRARIES( lucasKanadeTracker ${OpenCV_LIBS} )
