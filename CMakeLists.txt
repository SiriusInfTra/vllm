cmake_minimum_required(VERSION 3.21)

project(vllm_extensions LANGUAGES CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11" "3.12")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx908;gfx90a;gfx942;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.1.2")
set(TORCH_SUPPORTED_VERSION_ROCM_5X "2.0.1")
set(TORCH_SUPPORTED_VERSION_ROCM_6X "2.1.1")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
# Normally `torch.utils.cpp_extension.CUDAExtension` would add
# `libtorch_python.so` for linking against an extension. Torch's cmake
# configuration does not include this library (presumably since the cmake
# config is used for standalone C++ binaries that link against torch).
# The `libtorch_python.so` library defines some of the glue code between
# torch/python via pybind and is required by VLLM extensions for this
# reason. So, add it by manually with `find_library` using torch's
# installed library path.
#
find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.x
  if (ROCM_VERSION_DEV_MAJOR EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM_5X})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_ROCM_5X} "
      "expected for ROCMm 5.x build, saw ${Torch_VERSION} instead.")
  endif()

  # ROCm 6.x
  if (ROCM_VERSION_DEV_MAJOR EQUAL 6 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM_6X})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_ROCM_6X} "
      "expected for ROCMm 6.x build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `VLLM_GPU_ARCHES`.
#
override_gpu_arches(VLLM_GPU_ARCHES
  ${VLLM_GPU_LANG}
  "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Define extension targets
#

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc/cache_kernels.cu"
  "csrc/attention/attention_kernels.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/quantization/squeezellm/quant_cuda_kernel.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/moe_align_block_size_kernels.cu"
  "csrc/pybind.cpp")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_EXT_SRC
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/quantization/marlin/marlin_cuda_kernel.cu"
    "csrc/custom_all_reduce.cu")
endif()

define_gpu_extension_target(
  _C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  WITH_SOABI)

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/moe_ops.cpp"
  "csrc/moe/topk_softmax_kernels.cu")

define_gpu_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  WITH_SOABI)

#
# _punica_C extension
#

set(VLLM_PUNICA_EXT_SRC
  "csrc/punica/bgmv/bgmv_bf16_bf16_bf16.cu"
  "csrc/punica/bgmv/bgmv_bf16_bf16_fp16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp16_bf16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp16_fp16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp32_bf16.cu"
  "csrc/punica/bgmv/bgmv_bf16_fp32_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp16_bf16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp16_bf16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp32_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp16_fp32_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp32_bf16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp32_bf16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp16_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp16_fp16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp32_bf16.cu"
  "csrc/punica/bgmv/bgmv_fp32_fp32_fp16.cu"
  "csrc/punica/punica_ops.cc")

#
# Copy GPU compilation flags+update for punica
#
set(VLLM_PUNICA_GPU_FLAGS ${VLLM_GPU_FLAGS})
list(REMOVE_ITEM VLLM_PUNICA_GPU_FLAGS
  "-D__CUDA_NO_HALF_OPERATORS__"
  "-D__CUDA_NO_HALF_CONVERSIONS__"
  "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
  "-D__CUDA_NO_HALF2_OPERATORS__")

#
# Filter out CUDA architectures < 8.0 for punica.
#
if (${VLLM_GPU_LANG} STREQUAL "CUDA")
  set(VLLM_PUNICA_GPU_ARCHES)
  foreach(ARCH ${VLLM_GPU_ARCHES})
    string_to_ver(CODE_VER ${ARCH})
    if (CODE_VER GREATER_EQUAL 8.0)
      list(APPEND VLLM_PUNICA_GPU_ARCHES ${ARCH})
    endif()
  endforeach()
  message(STATUS "Punica target arches: ${VLLM_PUNICA_GPU_ARCHES}")
endif()

if (VLLM_PUNICA_GPU_ARCHES)
  define_gpu_extension_target(
    _punica_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_PUNICA_EXT_SRC}
    COMPILE_FLAGS ${VLLM_PUNICA_GPU_FLAGS}
    ARCHITECTURES ${VLLM_PUNICA_GPU_ARCHES}
    WITH_SOABI)
else()
  message(WARNING "Unable to create _punica_C target because none of the "
    "requested architectures (${VLLM_GPU_ARCHES}) are supported, i.e. >= 8.0")
endif()

#
# Add the `default` target which detects which extensions should be
# built based on platform/architecture.  This is the same logic that
# setup.py uses to select which extensions should be built and should
# be kept in sync.
#
# The `default` target makes direct use of cmake easier since knowledge
# of which extensions are supported has been factored in, e.g.
#
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../vllm ..
# cmake --build . --target default
#
add_custom_target(default)

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling C extension.")
  add_dependencies(default _C)
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  message(STATUS "Enabling moe extension.")
  add_dependencies(default _moe_C)

  # Enable punica if -DVLLM_INSTALL_PUNICA_KERNELS=ON or
  # VLLM_INSTALL_PUNICA_KERNELS is set in the environment and
  # there are supported target arches.
  if (VLLM_PUNICA_GPU_ARCHES AND
      (ENV{VLLM_INSTALL_PUNICA_KERNELS} OR VLLM_INSTALL_PUNICA_KERNELS))
    message(STATUS "Enabling punica extension.")
    add_dependencies(default _punica_C)
  endif()
endif()
