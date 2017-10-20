set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  
project(winograd_test)

add_executable(winograd_test src/winograd_test.cpp)

include_directories(/usr/include)

target_link_libraries(winograd_test /usr/lib/libcblas.so)




