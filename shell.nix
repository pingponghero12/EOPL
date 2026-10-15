{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
  },
}:

pkgs.mkShell {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    git
    cmake
    gnumake
    gcc12
    pkg-config
    # CUDA packages
    cudaPackages.cudatoolkit
    cudaPackages.cuda_cudart
    cudaPackages.libcurand
    cudaPackages.libcublas
    # NVIDIA driver
    linuxPackages.nvidia_x11
    # OpenGL libraries
    libGLU
    libGL
    # X11 libraries for CUDA samples
    xorg.libXi
    xorg.libXmu
    freeglut
    xorg.libXext
    xorg.libX11
    xorg.libXv
    xorg.libXrandr
    # Standard libraries
    zlib
    ncurses5
    stdenv.cc
    binutils

    (python312.withPackages (
      ps: with ps; [
        numpy
        scipy
        pandas
        matplotlib
        pytest
        scikit-learn
        seaborn
        pybind11
        scikit-build-core
        pip
      ]
    ))
  ];
  shellHook = ''
    # Set up CUDA paths
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export CUDA_ROOT=${pkgs.cudaPackages.cudatoolkit}

    # Set up compiler paths to use GCC 12
    export CC=${pkgs.gcc12}/bin/gcc
    export CXX=${pkgs.gcc12}/bin/g++
    export CUDAHOSTCXX=${pkgs.gcc12}/bin/g++
    export PATH=${pkgs.gcc12}/bin:$PATH

    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"

    # Set up library paths
    export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_cudart}/lib:$LIBRARY_PATH"

    # Include paths
    export C_INCLUDE_PATH="${pkgs.cudaPackages.cudatoolkit}/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="${pkgs.cudaPackages.cudatoolkit}/include:$CPLUS_INCLUDE_PATH"

    echo "CUDA environment loaded!"
    echo "GCC version: $(gcc --version | head -1)"
    echo "CUDA version: $(nvcc --version | grep release || echo 'nvcc not found')"
    echo "CUDA path: $CUDA_PATH"

    # Check if CUDA libraries exist
    if [ -d "$CUDA_PATH/lib64" ]; then
      echo "CUDA lib64 found: $CUDA_PATH/lib64"
      ls -la "$CUDA_PATH/lib64" | grep -E "(cudart|curand|cublas)" | head -5
    fi
  '';
}
