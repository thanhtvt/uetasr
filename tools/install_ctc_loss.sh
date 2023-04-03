mkdir -p extras
cd ./extras || exit

export CUDA_HOME=/usr/local/cuda

# Install ctc_loss
if [ ! -d warp-ctc ]; then
  git clone https://github.com/mxi-box/warp-ctc.git
fi

cd ./warp-ctc || exit
mkdir -p build && cd build || exit

if [ "$CUDA_HOME" ]; then
  cmake \
      -DUSE_NAIVE_KERNEL=on \
      -DCMAKE_C_COMPILER_LAUNCHER="$(which gcc)" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="$(which g++)"  \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" ..
else
  cmake \
      -DUSE_NAIVE_KERNEL=on \
      -DCMAKE_C_COMPILER_LAUNCHER="$(which gcc)" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="$(which g++)" ..
fi

make

cd ../tensorflow_binding || exit

if [ "$CUDA_HOME" ]; then
  CUDA="$CUDA_HOME" python3 setup.py install
else
  python3 setup.py install
fi

cd ../..
# fi

cd ..