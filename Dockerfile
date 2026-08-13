FROM nvcr.io/nvidia/pytorch:25.03-py3

ARG TARGETARCH
ARG DEBIAN_FRONTEND=noninteractive

# Core build tooling and runtime libs needed by scikit-sparse / OpenCV / fused-ssim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git wget ca-certificates xz-utils \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev \
    libsuitesparse-dev libmetis-dev liblapack-dev libblas-dev \
    libboost-filesystem-dev libboost-graph-dev libboost-program-options-dev \
    libboost-system-dev libfreeimage-dev libflann-dev liblz4-dev \
    libopenimageio-dev openimageio-tools libopencv-dev libsqlite3-dev libcgal-dev libglew-dev libgl1 libgl1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build and install Ceres Solver (needed by pyceres)
ARG CERES_VERSION=2.1.0
RUN wget -q http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz \
    && tar --no-same-owner -zxf ceres-solver-${CERES_VERSION}.tar.gz \
    && mkdir ceres-build \
    && cd ceres-build \
    && cmake ../ceres-solver-${CERES_VERSION} -GNinja \
         -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON \
         -DMINIGLOG=OFF -DSUITESPARSE=OFF -DCXSPARSE=OFF \
    && ninja install \
    && cd / \
    && rm -rf ceres-build ceres-solver-${CERES_VERSION} ceres-solver-${CERES_VERSION}.tar.gz

# Build and install COLMAP from source instead of using the distro package.
ARG COLMAP_VERSION=main
ARG COLMAP_CUDA_ARCHITECTURES=89
RUN git clone --branch ${COLMAP_VERSION} --depth 1 https://github.com/colmap/colmap.git /tmp/colmap \
    && cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
         -DCMAKE_BUILD_TYPE=Release \
         "-DCMAKE_CUDA_ARCHITECTURES=${COLMAP_CUDA_ARCHITECTURES}" \
         -DGUI_ENABLED=OFF \
         -DOPENGL_ENABLED=OFF \
    && cmake --build /tmp/colmap/build \
    && cmake --install /tmp/colmap/build \
    && ldconfig \
    && colmap help >/dev/null \
    && rm -rf /tmp/colmap

# Install cuDSS via pip for CUDA 12
RUN pip install --no-cache-dir "nvidia-cudss-cu12<=0.7.1.6"

ENV CUDA_HOME=/usr/local/cuda
ENV FUSED_SSIM_FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.9"

WORKDIR /workspace/InstantSfM

# Install Python deps separately so code edits don't bust the cache.
COPY pyproject.toml README.md ./
RUN set -euo pipefail \
    && python -c "import tomllib, pathlib, importlib.metadata as m; py=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); inst={d.metadata['Name'].lower() for d in m.distributions()}; norm=lambda r: r.split('=')[0].split('>')[0].split('<')[0].split('!')[0].split('~')[0].split('@')[0].split(' ')[0].strip().lower(); reqs=[norm(r) if (norm(r) in inst and any(o in r for o in ['==','>=','<=','~='])) else r for r in py['project']['dependencies'] if not r.startswith('fused-ssim @ ')]; pathlib.Path('/tmp/requirements.txt').write_text('\\n'.join(reqs) + '\\n')" \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && git clone --depth=1 https://github.com/rahul-goel/fused-ssim /tmp/fused-ssim \
    && python -c "from pathlib import Path; p=Path('/tmp/fused-ssim/setup.py'); t=p.read_text(); t=t.replace('elif torch.mps.is_available():','elif False and torch.mps.is_available():'); t=t.replace('elif hasattr(torch, \\'xpu\\'):', 'elif False and hasattr(torch, \\'xpu\\'):'); p.write_text(t)" \
    && pip install --no-cache-dir --no-build-isolation /tmp/fused-ssim \
    && MAX_JOBS=$(nproc) pip install --no-cache-dir --no-build-isolation git+https://github.com/zitongzhan/bae.git

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}

# Install the project itself (editable, no deps) after source copy.
COPY instantsfm instantsfm
COPY demo.py demo.py
RUN sed -i 's/^license = .*/license = { text = "CC-BY-NC-4.0" }/' pyproject.toml \
    && pip install --no-cache-dir -e . --no-deps

# Bring in the rest of the project (assets, configs, etc.)
COPY . .

CMD ["/bin/bash"]
