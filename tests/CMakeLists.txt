add_executable(test_utf8
    unit/test_utf8.cpp
)
target_include_directories(test_utf8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_utf8 PRIVATE diffusion_core)
add_test(NAME test_utf8 COMMAND test_utf8)

if (GGML_CUDA)
    add_executable(test_diff_device
        unit/test_diff_device.cu
    )
    set_source_files_properties(unit/test_diff_device.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(test_diff_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(test_diff_device PRIVATE diffusion_core)
    add_test(NAME test_diff_device COMMAND test_diff_device)

    add_executable(test_latency_breakdown
        unit/test_latency_breakdown.cu
    )
    set_source_files_properties(unit/test_latency_breakdown.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(test_latency_breakdown PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(test_latency_breakdown PRIVATE diffusion_core)

    add_executable(test_active_query_breakdown
        unit/test_active_query_breakdown.cu
    )
    set_source_files_properties(unit/test_active_query_breakdown.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(test_active_query_breakdown PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(test_active_query_breakdown PRIVATE diffusion_core)

    add_executable(test_sparse_graph_transitions
        unit/test_sparse_graph_transitions.cu
    )
    set_source_files_properties(unit/test_sparse_graph_transitions.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(test_sparse_graph_transitions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(test_sparse_graph_transitions PRIVATE diffusion_core)
endif()

add_executable(test_diffusion_mock
    unit/test_diffusion_mock.cpp
)
target_include_directories(test_diffusion_mock PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_diffusion_mock PRIVATE diffusion_core)
add_test(NAME test_diffusion_mock COMMAND test_diffusion_mock)

add_executable(test_pos_remapping
    unit/test_pos_remapping.cpp
)
target_include_directories(test_pos_remapping PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_pos_remapping PRIVATE diffusion_core)
add_test(NAME test_pos_remapping COMMAND test_pos_remapping)
