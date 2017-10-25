set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  
project(cblas_test)
add_executable(cblas_test cblas_use.cpp)
include_directories(/usr/include)
target_link_libraries(cblas_test /usr/lib/libcblas.so)


