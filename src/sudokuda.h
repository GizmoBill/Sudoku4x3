// *******************************
// *                             *
// *  Sudoku 3x4 Cuda Interface  *
// *                             *
// *******************************

// This struct allows the C++ code to create and use the big tables, and also pass them
// to the GPU while keeping C++ and Cuda reasonably separate.
struct GridCountPacket
{
  // The DoubleBoxCount (346 * 346 = 119716) entries in the huge gcPackets table are sorted
  // by GridCounter::packetOrder_ into a run order that is slightly more favorable to data
  // cache hit rate than the natural order produced by GridCounter::setup_.
  int32_t  runOrder;

  // If this packet is used for band1, then otherIndex gives the index of the packet for
  // band2. 
  int32_t  otherIndex;

  // The index [0 .. 8] of BandGang::cache for the band count data
  uint8_t  cacheLevel;

  // 0, 1, or 2 to account for the two-way box4-box8 symmetry. See GridCounter::setup_.
  uint8_t  multiplier;

  // Rename box 6, 7, 10, or 11 for the double-mapping used to put boxes 4 and 5 in
  // canonical form for finding the gangster. This really should be a ColCode instead
  // of uint16_t, but this is read by Cuda and I didn't want to make Cuda swallow all
  // the C++ declarations, or split the C++ into header files that would separate the
  // definitions from the declarations. The code organization isn't ideal, but
  // reflects its evolution and my desire to avoid radical refactoring that might
  // introduce bugs.
  uint16_t doubleRename[5775];
};

#ifdef JETSON

// Since we're not giving the Cuda source all the C++ declarations
constexpr int32_t ColCodeCount = 5775;
constexpr int32_t ColCompatibleCount = 346;

void printDeviceProperties();

// Call this to deallocate all device and host memory used by the GPU
void sudokudaEnd();

// Set this number of blocks and theads to use for gpuGroup, gpuAddGroup, and gpuMainCount.
// All three of those functions must use the same values.
void gpuGrid(uint32_t blocksX, uint32_t threadsX, uint32_t threadsY = 1);

// Called in the GridCounter constructor to copy the bane counts in the gangster cache,
// and the column code compatability table, to device memory.
void gpuInit
(
  const int32_t bandCounts[][ColCodeCount][ColCodeCount],
  const uint16_t codeCompatTable[][ColCompatibleCount][2]
);

// Called for each of the 9 gangster sets to copy the GridCountPackets to device memory
void gpuSetup(const GridCountPacket* gcPackets);

// Called for each box2 group to set the group size and allocate host and device memory
void gpuGroup(int box2GroupSize, const uint16_t* box3List);

// Called at the end of a box2 group to add the results to the groupCounts array,
// at CPU thread 0
void gpuAddGroup(uint64_t* groupCounts, int groupStride);

// Main count looops to enumerate one GridCountPacket index (box01) for one box2 group
void gpuMainCount(int box01/*Start, int box01End*/, int box2);

#endif
