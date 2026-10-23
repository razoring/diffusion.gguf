cmake_minimum_required(VERSION 3.18)
project(diffusion.gguf VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>")
endif()

# llama.cpp submodule options
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common utils" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama examples" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build llama tools" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama server" FORCE)

option(DIFFUSION_BUILD_ORACLE "Build upstream llama-server and llama-cli as test oracle" OFF)
if (DIFFUSION_BUILD_ORACLE)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "Build llama tools" FORCE)
    set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama server" FORCE)
endif()

option(DIFFUSION_PARITY_BUILD "Disable sgemm / BLAS optimizations for exact bit parity testing" OFF)
if (DIFFUSION_PARITY_BUILD)
    set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile sgemm" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "Disable BLAS" FORCE)
endif()

if (DIFFUSION_BUILD_ORACLE)
    add_subdirectory(third_party/llama.cpp)
else()
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
endif()

# Core library
add_library(diffusion_core STATIC
    src/utf8.cpp
    src/init_guess.cpp
    src/context_trie.cpp
    src/engine.cpp
    src/scheduler_ar.cpp
    src/scheduler_jacobi.cpp
    src/llama/backend_llama.cpp
    src/llama/tokenizer_llama.cpp
    src/llama/sampler_llama.cpp
    src/llama/chat_llama.cpp
    src/llama/dist_positional.cpp
)

target_include_directories(diffusion_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
)

target_link_libraries(diffusion_core PUBLIC
    llama
    llama-common
    ggml
)

# CLI executable
add_executable(diffusion-cli
    tools/cli/main.cpp
    tools/cli/args.cpp
    tools/cli/render_final.cpp
    tools/cli/render_live.cpp
    tools/cli/render_progress.cpp
    tools/cli/repl.cpp
)
target_link_libraries(diffusion-cli PRIVATE diffusion_core)


# Server executable
add_executable(diffusion-server
    tools/server/main.cpp
)
target_link_libraries(diffusion-server PRIVATE diffusion_core cpp-httplib)
if (WIN32)
    target_link_libraries(diffusion-server PRIVATE ws2_32)
endif()


# Converter executable
add_executable(diffusion-convert
    tools/convert/main.cpp
    tools/convert/ggjt_reader.cpp
    tools/convert/quant_repack.cpp
    tools/convert/gguf_emit.cpp
)
target_link_libraries(diffusion-convert PRIVATE diffusion_core)


enable_testing()
add_subdirectory(tests)
