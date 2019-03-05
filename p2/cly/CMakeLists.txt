cmake_minimum_required(VERSION 3.8)
project(heat2D3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

enable_language("CUDA")

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

include_directories(
        
)

set(SOURCE_FILES
        
        )

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES} main.cu)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)

