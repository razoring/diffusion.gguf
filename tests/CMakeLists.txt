add_executable(test_utf8
    unit/test_utf8.cpp
)
target_include_directories(test_utf8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_utf8 PRIVATE diffusion_core)
add_test(NAME test_utf8 COMMAND test_utf8)

add_executable(test_jacobi_mock
    unit/test_jacobi_mock.cpp
)
target_include_directories(test_jacobi_mock PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_jacobi_mock PRIVATE diffusion_core)
add_test(NAME test_jacobi_mock COMMAND test_jacobi_mock)

add_executable(test_diff_device
    unit/test_diff_device.cu
)
target_include_directories(test_diff_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_diff_device PRIVATE diffusion_core)
add_test(NAME test_diff_device COMMAND test_diff_device)

add_executable(test_diffusion_mock
    unit/test_diffusion_mock.cpp
)
target_include_directories(test_diffusion_mock PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_diffusion_mock PRIVATE diffusion_core)
add_test(NAME test_diffusion_mock COMMAND test_diffusion_mock)

