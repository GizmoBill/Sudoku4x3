// **************************
// *                        *
// *  Sudoku 3x4 Cuda Code  *
// *                        *
// **************************

// This file is edited in Visual Studio by fooling it into thinking it's a cpp file,
// then copied to the Jetson and renamed from cpp to cu. These definitions aid that
// fooling, and are not seen by the Cuda toolchain, because JETSON is defined.
#ifndef JETSON
#define JETSON
#define __global__

struct dim3
{
  uint32_t x, y, z;
  dim3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1)
    : x(x), y(y), z(z)
  {}
}
threadIdx, blockIdx, blockDim, gridDim;

#endif

#include <stdint.h>
#include <cudapp.h>
#include <vector>
#include <array>
#include "sudokuda.h"

constexpr uint32_t DoubleBoxCount = ColCompatibleCount * ColCompatibleCount;

dim3 threads(32);
dim3 blocks(88);

struct Box711
{
  uint16_t box7;
  uint16_t box11;
  Box711() = default;
  Box711(uint16_t box7, uint16_t box11) : box7(box7), box11(box11) {}
};

// Box711 box711[ceil(groupSize/Coalesce)][ColCompatibleCount][Coalesce];
// get(groupIndex, b3) = box711[groupIndex >> CoalesceBits][b3/Coalesce][groupIndex & CoalesceMaskLo]
// thread(groupIndex, b3) = groupIndex & 31;
// assume blockDim.x is power of 2
constexpr uint32_t CoalesceBits = 2;
constexpr uint32_t Coalesce = 1 << CoalesceBits;
constexpr uint32_t CoalesceMaskLo = Coalesce - 1;
constexpr uint32_t CoalesceMaskHi = ~CoalesceMaskLo;

// ****************
// *              *
// *  Big Tables  *
// *              *
// ****************
//
// Huge tables created by CPU code and copied to device memory here. Also some device
// and host arrays allocated here to avoid alloc/free operations in the counting loops.
// All device and host memory collected here so it can be constructed and destroyed
// easily.
struct BigTables
{
  // Device copy of BangGang::gangCache_, holding band counts at this stage
  CudaDeviceMemory<int32_t [ColCodeCount][ColCodeCount]> gangCache{ 9 };

  // Device copy of GridCounter::codeCompatTable_
  const uint16_t (*codeCompatTableHost)[ColCompatibleCount][2];
  CudaDeviceMemory<uint16_t [ColCompatibleCount][2]> codeCompatTableDev{ ColCodeCount };

  // Device copy of GridCounter::gcPackets_
  const GridCountPacket* gcPacketsHost;
  CudaDeviceMemory<GridCountPacket> gcPacketsDev{ DoubleBoxCount };

  // list of box3 codes for current group
  CudaDeviceMemory<uint16_t> box3;

  // Device box7 and box11 codes
  CudaDeviceMemory<Box711> box711;

  // Partial counts, effectively uint64_t counts[blocks][box2GroupSize]
  CudaDeviceMemory<uint64_t> counts;

  int groupSize() const { return box3.numElements(); }
}
*bigTables = nullptr;

void sudokudaEnd()
{
  delete bigTables;
}

// **************************
// *                        *
// *  Print GPU Properties  *
// *                        *
// **************************

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

// ************************************
// *                                  *
// *  Setup Functions Called by Host  *
// *                                  *
// ************************************

void gpuGrid(uint32_t blocksX, uint32_t threadsX, uint32_t threadsY)
{
  blocks.x = blocksX;
  threads.x = threadsX;
  threads.y = threadsY;
}

void gpuInit(const int32_t cache[][ColCodeCount][ColCodeCount],
             const uint16_t codeCompatTable[][ColCompatibleCount][2])
{
  if (!bigTables)
    bigTables = new BigTables();
  bigTables->gangCache.copyTo(cache);
  bigTables->codeCompatTableHost = codeCompatTable;
  bigTables->codeCompatTableDev.copyTo(codeCompatTable);
}

void gpuSetup(const GridCountPacket* gcPackets)
{
  bigTables->gcPacketsHost = gcPackets;
  bigTables->gcPacketsDev.copyTo(gcPackets);
}

__global__
void clearCounts(uint64_t* counts, int groupSize)
{
  uint64_t* p = counts + blockIdx.x * groupSize;
  for (int groupIndex = threadIdx.x; groupIndex < groupSize; groupIndex += blockDim.x)
    p[groupIndex] = 0;
}

void gpuGroup(int box2GroupSize, const uint16_t* box3List)
{
  bigTables->box3.alloc(box2GroupSize);
  bigTables->box3.copyTo(box3List);

  int32_t box711Size = ColCompatibleCount * ((box2GroupSize + CoalesceMaskLo) & CoalesceMaskHi);
  bigTables->box711.alloc(box711Size);
  bigTables->box711.clear();

  bigTables->counts.alloc(box2GroupSize * blocks.x);
  clearCounts<<<blocks, threads>>>(bigTables->counts.mem(), box2GroupSize);
}

void gpuAddGroup(uint64_t* groupCounts, int groupStride)
{
  std::vector<uint64_t> counts(bigTables->counts.numElements());
  bigTables->counts.copyFrom(counts.data());

  int groupSize = bigTables->groupSize();

  for (int groupIndex = 0; groupIndex < groupSize; ++groupIndex)
  {
    uint64_t count = 0;
    for (int block = 0; block < blocks.x; ++block)
      count += counts[block * groupSize + groupIndex];
    groupCounts[groupIndex * groupStride] += count;
  }
}

// ***********************
// *                     *
// *  Box7/Box11 Kernel  *
// *                     *
// ***********************
//
// gridDim.x must be a multiple of Coalesce
__global__
void box711(int groupSize, const uint16_t (*codeCompatTable)[ColCompatibleCount][2],
            const uint16_t* doubleRename0, const uint16_t* doubleRename1,
            const uint16_t* box3List, Box711* box711)
{
  box711 += blockIdx.x & CoalesceMaskLo;

  for (int groupIndex = blockIdx.x; groupIndex < groupSize; groupIndex += gridDim.x)
  {
    int box3 = box3List[groupIndex];
    const uint16_t (*compat)[2] = codeCompatTable[box3];

    Box711* box711Line = box711 + ColCompatibleCount * (CoalesceMaskHi & groupIndex);

    for (int b3 = threadIdx.x; b3 < ColCompatibleCount; b3 += blockDim.x)
    {
      int b7  = compat[b3][0];
      int b11 = compat[b3][1];
      Box711& bx = box711Line[b3 * Coalesce];
      bx.box7  = doubleRename0[b7 ];
      bx.box11 = doubleRename1[b11];
    }
  }
}

// **************************
// *                        *
// *  Main Counting Kernel  *
// *                        *
// **************************

__global__
void groupCount(const uint16_t (*codeCompatTable)[2],
                const uint16_t* doubleRename0, const uint16_t* doubleRename1,
                const int32_t (*cache0)[ColCodeCount], const int32_t (*cache1)[ColCodeCount],
                const Box711* box711, int groupSize, int multiplier, uint64_t* counts)
{
  uint64_t* p = counts + blockIdx.x * groupSize;
  box711 += threadIdx.x & CoalesceMaskLo;

  for (int b2 = blockIdx.x; b2 < ColCompatibleCount; b2 += gridDim.x)
  {
    int box6  = codeCompatTable[b2][0];
    int box10 = codeCompatTable[b2][1];

    box6  = doubleRename0[box6 ];
    box10 = doubleRename1[box10];

    const int32_t* band1CacheLine = cache0[box6 ];
    const int32_t* band2CacheLine = cache1[box10];

    for (int groupIndex = threadIdx.x; groupIndex < groupSize; groupIndex += blockDim.x)
    {
      const Box711* box711Line  = box711 + (groupIndex & CoalesceMaskHi) * ColCompatibleCount;
      uint64_t count = 0;
      for (int b3 = 0; b3 < Coalesce * ColCompatibleCount; b3 += Coalesce)
        count += (uint64_t)band1CacheLine[box711Line[b3].box7] * band2CacheLine[box711Line[b3].box11];

      p[groupIndex] += count * multiplier;
    }
  }
}

// *****************************
// *                           *
// *  Main Counting Host Code  *
// *                           *
// *****************************

void gpuMainCount(int box01, int box2)
{
  const GridCountPacket& gcp0 = bigTables->gcPacketsHost[box01];

  int box01Other = gcp0.otherIndex;
  const GridCountPacket& gcp1 = bigTables->gcPacketsHost[box01Other];

  int groupSize = bigTables->groupSize();

  Box711* box711List  = bigTables->box711.mem();

  const uint16_t* doubleRename0 = bigTables->gcPacketsDev[box01     ].doubleRename;
  const uint16_t* doubleRename1 = bigTables->gcPacketsDev[box01Other].doubleRename;

  box711<<<64, 64>>>(groupSize, bigTables->codeCompatTableDev.mem(),
                              doubleRename0, doubleRename1, bigTables->box3.mem(),
                              box711List);
  checkLaunch();

  const uint16_t (*codeCompatTable)[2] = bigTables->codeCompatTableDev[box2];

  const int32_t (*cache0)[ColCodeCount] = bigTables->gangCache[gcp0.cacheLevel];
  const int32_t (*cache1)[ColCodeCount] = bigTables->gangCache[gcp1.cacheLevel];

  int multiplier = bigTables->gcPacketsHost[box01].multiplier;

  uint64_t* counts = bigTables->counts.mem();
  //safeSync();

  groupCount<<<blocks, threads>>>(codeCompatTable, doubleRename0, doubleRename1,
                                  cache0, cache1, box711List,
                                  groupSize, multiplier, counts);
  checkLaunch();
  //safeSync();
}
