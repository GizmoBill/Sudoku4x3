// **************************
// *                        *
// *  Sudoku 3x4 Cuda Code  *
// *                        *
// **************************
//
// The following is a first cut at a Cuda GPU program for executing the main grid counting
// loop for the Sudoku 3x4 exact count. It makes very poor use of GPU resources, and is
// actually slower than running the CPU code on the Jetson's 8-core ARM v8.2 processors.
// The purpose of this first cut is to confirm that I understand the Nvidia tool chain
// and the most basic operations of Cuda. The code runs and gets the correct results.
// The first step in using the GPU properly will be to deal with the poor memory access
// pattern, which radically degrades the GPU's memory bandwidth and stalls the compute
// elements. I have a plan ...

// This file is edited in Visual Studio by fooling it into thinking it's a cpp file,
// then copied to the Jetson and renamed from cpp to cu. These definitions aid that
// fooling, and are not seen by the Cuda toolchain, because JETSON is defined.
#ifndef JETSON
#define JETSON
#define __global__
#endif

#include <stdint.h>
#include <cudapp.h>
#include "sudokuda.h"

constexpr uint32_t DoubleBoxCount = ColCompatibleCount * ColCompatibleCount;

struct BigTables
{
  CudaUnifiedMemory<int32_t [ColCodeCount][ColCodeCount]> gangCache = 9;

  CudaUnifiedMemory<uint16_t [ColCompatibleCount][2]> codeCompatTable = ColCodeCount;

  CudaUnifiedMemory<GridCountPacket> gcPackets = DoubleBoxCount;
}
*bigTables = nullptr;

void printDeviceProperties()
{
  int deviceId = cudaGetDevice(&deviceId);
  cudaDeviceProp dp;
  checkErr(cudaGetDeviceProperties(&dp, deviceId));

  printf("Clock rate %d kHz\n", dp.clockRate);
  printf("L2 cache size %d\n", dp.l2CacheSize);
  printf("Max blocks per multiprocessor %d\n", dp.maxBlocksPerMultiProcessor);
  printf("Max grid size %d.%d.%d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
  printf("Max block dimension %d.%d.%d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
  printf("Max threads per block %d\n", dp.maxThreadsPerBlock);
  printf("Max threads per multiprocessor %d\n", dp.maxThreadsPerMultiProcessor);
  printf("Multiprocessor count %d\n", dp.multiProcessorCount);
  printf("Reserved shared memory per block %d bytes\n", dp.reservedSharedMemPerBlock);
  printf("Shared memory per block %d bytes\n", dp.sharedMemPerBlock);
  printf("Shared memory per multiprocessor %d bytes\n", dp.sharedMemPerMultiprocessor);
  printf("Total global memory on device %lu bytes\n", dp.totalGlobalMem);
  printf("Warp size in threads %d \n", dp.warpSize);
}

void sudokudaEnd()
{
  delete bigTables;
}


void cudaCache(const int32_t cache[][ColCodeCount][ColCodeCount],
               const uint16_t codeCompatTable[][ColCompatibleCount][2])
{
  if (!bigTables)
    bigTables = new BigTables();
  bigTables->gangCache.copyTo(cache);
  bigTables->codeCompatTable.copyTo(codeCompatTable);
}

void cudaSetup(const GridCountPacket* gcPackets)
{
  bigTables->gcPackets.copyTo(gcPackets);
}

__global__
void count(const int32_t  gangCache[][ColCodeCount][ColCodeCount], 
           const uint16_t codeCompatTable[][ColCompatibleCount][2],
           const GridCountPacket* gcPackets,
           int box2, int box3, uint64_t* counts)
{
  uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t stride = blockDim.x * gridDim.x;

  uint16_t box7[ColCompatibleCount], box11[ColCompatibleCount];

  uint64_t count = 0;

  for (uint32_t box01 = threadIndex; box01 < DoubleBoxCount; box01 += stride)
  {
    const GridCountPacket& gcp0 = gcPackets[box01];

    if (gcp0.multiplier == 0)
      continue;

    int box01Other = gcp0.otherIndex;
    const GridCountPacket& gcp1 = gcPackets[box01Other];

    const int32_t (*band1Cache)[ColCodeCount] = gangCache[gcp0.cacheLevel];
    const int32_t (*band2Cache)[ColCodeCount] = gangCache[gcp1.cacheLevel];

    for (int b3 = 0; b3 < ColCompatibleCount; ++b3)
    {
      int b7  = codeCompatTable[box3][b3][0];
      int b11 = codeCompatTable[box3][b3][1];

      box7 [b3] = gcp0.doubleRename[b7 ];
      box11[b3] = gcp1.doubleRename[b11];
    }

    uint64_t partialCount = 0;
    for (int b2 = 0; b2 < ColCompatibleCount; ++b2)
    {
      int box6  = codeCompatTable[box2][b2][0];
      int box10 = codeCompatTable[box2][b2][1];

      box6  = gcp0.doubleRename[box6 ];
      box10 = gcp1.doubleRename[box10];

      for (int b3 = 0; b3 < ColCompatibleCount; ++b3)
        partialCount += (uint64_t)band1Cache[box6][box7[b3]] * band2Cache[box10][box11[b3]];
    }

    count += partialCount * gcp0.multiplier;
  }

  counts[threadIndex] = count;
}

uint64_t cudaCount(uint16_t box2, uint16_t box3)
{
  int blocks = 16;
  int threads = 32;

  CudaUnifiedMemory<uint64_t> counts = blocks * threads;

  count<<<blocks, threads>>>(bigTables->gangCache.mem(), bigTables->codeCompatTable.mem(),
                             bigTables->gcPackets.mem(), box2, box3, counts.mem());
  checkLaunch();
  safeSync();

  uint64_t total = 0;
  for (size_t i = 0; i < counts.numElements(); ++i)
    total += counts[i];

  return total;
}
