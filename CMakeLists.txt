cmake_minimum_required(VERSION 3.28)
project(c4d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Performance baseline (§0): -O3, native arch, fast-math allowed (no
# cross-platform byte determinism required). Verified: GCC emits AVX-512/AVX2 +
# FMA across the DWT and quantizer (the hot vectorizable loops); the remaining
# scalar loops (rANS renorm, per-token codec) are inherently serial. GCC beats
# clang here. constexpr is already maxed for throughput (the only runtime table,
# the per-voxel SubbandMap, is a one-time process-lifetime build, not per-chunk).
set(C4D_OPT_FLAGS -O3 -march=native -ffast-math -funroll-loops)

option(C4D_SANITIZE "Build with ASan+UBSan" OFF)
option(C4D_BUILD_TESTS "Build tests" ON)
option(C4D_BUILD_CLI "Build the c4d CLI" ON)
option(C4D_LTO "Link-time optimization (helps the multi-TU CLI)" ON)

if(C4D_LTO AND NOT C4D_SANITIZE)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT C4D_IPO_OK)
  if(C4D_IPO_OK)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

# Header-only core library (§0: header-heavy, constexpr/consteval).
add_library(c4d INTERFACE)
target_include_directories(c4d INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(c4d INTERFACE cxx_std_26)

function(c4d_tune tgt)
  target_compile_options(${tgt} PRIVATE ${C4D_OPT_FLAGS})
  if(C4D_SANITIZE)
    target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -O1)
    target_link_options(${tgt} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

if(C4D_BUILD_CLI)
  add_executable(c4d_cli cli/main.cpp)
  target_link_libraries(c4d_cli PRIVATE c4d)
  set_target_properties(c4d_cli PROPERTIES OUTPUT_NAME c4d)
  c4d_tune(c4d_cli)

  # zarr v2 -> .c4d converter (tools/). Reads raw zarr v2 always; blosc-
  # compressed chunks need libblosc (auto-detected; otherwise raw-only with a
  # clear runtime message). Modeled on vc_zarr_recompress, self-contained.
  add_executable(zarr_to_c4d tools/zarr_to_c4d.cpp)
  target_link_libraries(zarr_to_c4d PRIVATE c4d)
  find_library(BLOSC_LIB blosc)
  find_path(BLOSC_INC blosc.h)
  if(BLOSC_LIB AND BLOSC_INC)
    target_compile_definitions(zarr_to_c4d PRIVATE C4D_WITH_BLOSC)
    target_include_directories(zarr_to_c4d PRIVATE ${BLOSC_INC})
    target_link_libraries(zarr_to_c4d PRIVATE ${BLOSC_LIB})
    message(STATUS "zarr_to_c4d: blosc support enabled (${BLOSC_LIB})")
  else()
    message(STATUS "zarr_to_c4d: libblosc not found — raw-zarr only")
  endif()
  c4d_tune(zarr_to_c4d)
endif()

# Optional head-to-head benchmark against c3d v1 (needs C3D_DIR pointing at a
# c3d checkout with build/libc3d.a). Off unless C3D_DIR is set.
option(C4D_BUILD_C3D_BENCH "Build the c4d-vs-c3d benchmark" OFF)
if(C4D_BUILD_C3D_BENCH AND C3D_DIR)
  find_package(OpenMP)
  add_executable(compare_c3d bench/compare_c3d.cpp)
  target_link_libraries(compare_c3d PRIVATE c4d ${C3D_DIR}/build/libc3d.a)
  target_include_directories(compare_c3d PRIVATE ${C3D_DIR})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(compare_c3d PRIVATE OpenMP::OpenMP_CXX)
  endif()
  c4d_tune(compare_c3d)
endif()

if(C4D_BUILD_TESTS)
  enable_testing()
  file(GLOB C4D_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
  foreach(src ${C4D_TEST_SRCS})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE c4d)
    c4d_tune(${name})
    add_test(NAME ${name} COMMAND ${name})
    # test_zarr_export drives the converter binary end-to-end: give it the path
    # and make it depend on the binary (only when the CLI/tool is built).
    if(name STREQUAL "test_zarr_export" AND C4D_BUILD_CLI)
      target_compile_definitions(${name} PRIVATE
        C4D_ZARR_TO_C4D_BIN="$<TARGET_FILE:zarr_to_c4d>")
      add_dependencies(${name} zarr_to_c4d)
    endif()
  endforeach()
endif()
