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
  // cache hit rate than the natural order produced by GridCounter::setup_. Keeping the
  // original index in each GridCountPacket allows us to fix the sorted otherIndex elements.
  // It is not otherise used.
  int32_t  originalIndex;

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
  // the C++ definitions, or split the C++ into header files that would separate the
  // definitions from the declarations. The code organization isn't ideal, but
  // reflects its evolution and my desire to avoid radical refactoring that might
  // introduce bugs.
  uint16_t doubleRename[5775];
};

#ifdef JETSON

constexpr int32_t ColCodeCount = 5775;
constexpr int32_t ColCompatibleCount = 346;

void printDeviceProperties();

void sudokudaEnd();

void cudaCache
(
  const int32_t bandCounts[][ColCodeCount][ColCodeCount],
  const uint16_t codeCompatTable[][ColCompatibleCount][2]
);

void cudaSetup
(
  const GridCountPacket* gcPackets
);

uint64_t cudaCount(uint16_t box2, uint16_t box3);

#endif
