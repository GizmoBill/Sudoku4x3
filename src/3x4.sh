nvcc --m64 --std c++17 --gpu-architecture=sm_72 --compiler-options -std=c++17,-march=armv8-a+simd,-Ofast,-Wno-format,-DJETSON --linker-options -pthread --include-path . -o sudoku3x4 bignumMT.cpp profile.cpp general.cpp timer.cpp Sudoku3x4.cpp gridCount.cu

