################################################################################
#
# 
# 
################################################################################
cmake_minimum_required (VERSION 2.4)
if(POLICY CMP0048)
cmake_policy(SET CMP0048 OLD)
endif()
project(calibration)

# Collect all sources
file(GLOB_RECURSE CALIBRATION_SOURCE_FILES "src/*.cpp" "src/*.h" "src/*.inl")

# Clasify sources according to folder structure.
# This creates a nice layout for the project in visual studio
# This macro is courtesy of http://www.cmake.org/pipermail/cmake/2013-November/056336.html
macro(GroupSources curdir)
	file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir}
		${PROJECT_SOURCE_DIR}/${curdir}/*)
	foreach(child ${children})
		if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			GroupSources(${curdir}/${child})
		else()
			string(REPLACE "/" "\\" groupname ${curdir})
			source_group(${groupname} FILES
				${PROJECT_SOURCE_DIR}/${curdir}/${child})
		endif()
	endforeach()
endmacro()

GroupSources(cjson)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
endif()


# Find 3rd party packages
find_package(OpenCV 3.1 REQUIRED HINTS /usr/local)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
LINK_DIRECTORIES(${OpenCV_LIB_DIR})


set(APP_NAME calibration)

add_executable(${APP_NAME} ${CALIBRATION_SOURCE_FILES})
target_include_directories(${APP_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

target_include_directories(${APP_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} LINK_PUBLIC ${OpenCV_LIBS})

target_link_libraries(${APP_NAME} LINK_PUBLIC -lpthread -lm )
