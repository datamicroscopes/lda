cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(microscopes_lda)

# default mode is Release
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -MD -std=c++0x")

# since we use distributions headers, we need to inherit their
# no-strict-aliasing warnings rule otherwise deal with compiler noise
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")

# this warning is annoying; who cares about unused parameters
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  # for anaconda builds
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7")
  # clang complains about register
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif()

# taken from distributions
#set(CMAKE_CXX_FLAGS_MATHOPT "-mfpmath=sse -msse4.1 -ffast-math -funsafe-math-optimizations")
set(CMAKE_CXX_FLAGS_MATHOPT "-mfpmath=sse -msse4.1")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS_MATHOPT}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG_MODE -fno-omit-frame-pointer")

# give our include dirs the most precedent
include_directories(include)

# followed by the EXTRA_* ones
if(DEFINED EXTRA_INCLUDE_PATH)
  include_directories(${EXTRA_INCLUDE_PATH})
endif()
if(DEFINED EXTRA_LIBRARY_PATH)
  link_directories(${EXTRA_LIBRARY_PATH})
endif()

find_package(Protobuf REQUIRED)
message(STATUS "found protobuf INC=${PROTOBUF_INCLUDE_DIRS}, LIB=${PROTOBUF_LIBRARIES}")
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(Distributions)
if(DISTRIBUTIONS_FOUND)
  message(STATUS "found distributions INC=${DISTRIBUTIONS_INCLUDE_DIRS}, LIB=${DISTRIBUTIONS_LIBRARY_DIRS}")
  include_directories(${DISTRIBUTIONS_INCLUDE_DIRS})
  link_directories(${DISTRIBUTIONS_LIBRARY_DIRS})
else()
  message(FATAL_ERROR "Could not find distributions")
endif()

find_package(MicroscopesCommon)
if(MICROSCOPES_COMMON_FOUND)
  message(STATUS "found microscopes_common INC=${MICROSCOPES_COMMON_INCLUDE_DIRS}, LIB=${MICROSCOPES_COMMON_LIBRARY_DIRS}")
  include_directories(${MICROSCOPES_COMMON_INCLUDE_DIRS})
  link_directories(${MICROSCOPES_COMMON_LIBRARY_DIRS})
else()
  message(FATAL_ERROR "Could not find microscopes_common")
endif()

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY microscopes DESTINATION cython FILES_MATCHING PATTERN "*.pxd" PATTERN "__init__.py")

set(MICROSCOPES_LDA_SOURCE_FILES src/lda/model.cpp)
add_library(microscopes_lda SHARED ${MICROSCOPES_LDA_SOURCE_FILES})
target_link_libraries(microscopes_lda ${PROTOBUF_LIBRARIES} distributions_shared microscopes_common)
install(TARGETS microscopes_lda LIBRARY DESTINATION lib)
