#!/bin/bash
# Build COLMAP (GPU + GUI) on WSL2 Ubuntu 22.04, CUDA 12.6, Ada GPU (sm_89).
# Pure system-wide C++ build -> installs to /usr/local. Does NOT touch any conda env;
# run it from base or any env, it doesn't matter.
# (cuDSS and pycolmap are Python/pip concerns -> install them separately into a chosen env.)
# Adapted from InstantSfM's Dockerfile.
set -euo pipefail

COLMAP_BRANCH=${COLMAP_BRANCH:-main}
CERES_VERSION=${CERES_VERSION:-2.1.0}
CUDA_ARCH=${CUDA_ARCH:-89}        # RTX 40xx (Ada) = 89
JOBS=${JOBS:-8}                   # nvcc is RAM-heavy; 8 is safe on 27GB. Raise with JOBS=$(nproc) if you have headroom.

echo "==> [0/3] apt build deps (incl. Qt5 for GUI)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git wget ca-certificates xz-utils \
  libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev \
  libsuitesparse-dev libmetis-dev liblapack-dev libblas-dev \
  libboost-filesystem-dev libboost-graph-dev libboost-program-options-dev \
  libboost-system-dev libfreeimage-dev libflann-dev liblz4-dev \
  libsqlite3-dev libcgal-dev libglew-dev libgl1 libgl1-mesa-dev libglib2.0-0 \
  libopencv-dev libopenimageio-dev openimageio-tools \
  libcurl4-openssl-dev libssl-dev \
  qtbase5-dev libqt5opengl5-dev libqt5svg5-dev   # <-- GUI support (Core/Widgets + OpenGL + Svg)

# Ubuntu's OpenImageIO CMake config references the OpenCV4 include dir even though
# COLMAP doesn't use it; create it so find_package(OpenImageIO) succeeds.
sudo mkdir -p /usr/include/opencv4

# Ubuntu 22.04's apt cmake is 3.22, but COLMAP's fetched faiss needs >=3.24.
# Install CMake 3.31.10 system-wide (shadows apt's via /usr/local/bin). Avoid 4.x:
# it hard-removes support for cmake_minimum_required < 3.5 and breaks some fetched deps.
CMAKE_VERSION=3.31.10
wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -O /tmp/cmake.tgz
sudo tar -xzf /tmp/cmake.tgz -C /opt
sudo ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake /usr/local/bin/cmake
sudo ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/ctest /usr/local/bin/ctest
sudo ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cpack /usr/local/bin/cpack
rm -f /tmp/cmake.tgz
cmake --version

echo "==> [1/3] Ceres Solver ${CERES_VERSION}"
cd /tmp
wget -q "http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz"
tar -zxf "ceres-solver-${CERES_VERSION}.tar.gz"
rm -rf ceres-build && mkdir ceres-build && cd ceres-build
cmake "../ceres-solver-${CERES_VERSION}" -GNinja \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON \
  -DMINIGLOG=OFF -DSUITESPARSE=OFF -DCXSPARSE=OFF
ninja
sudo ninja install
sudo ldconfig
cd /tmp && rm -rf ceres-build "ceres-solver-${CERES_VERSION}" "ceres-solver-${CERES_VERSION}.tar.gz"

echo "==> [2/3] COLMAP ${COLMAP_BRANCH} (GPU sm_${CUDA_ARCH}, GUI=ON)"
rm -rf /tmp/colmap
git clone --branch "${COLMAP_BRANCH}" --depth 1 https://github.com/colmap/colmap.git /tmp/colmap
cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
  -DGUI_ENABLED=ON \
  -DOPENGL_ENABLED=ON
cmake --build /tmp/colmap/build --parallel "${JOBS}"
sudo cmake --install /tmp/colmap/build
sudo ldconfig

echo "==> [3/3] verify"
colmap --help 2>&1 | head -3
echo
echo "Done. GUI test (needs WSLg display):  colmap gui"
echo "GPU is used automatically when CUDA is present; check with:  colmap --help | grep -i cuda || true"
rm -rf /tmp/colmap
