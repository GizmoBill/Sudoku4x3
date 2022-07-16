// COPYRIGHT (C) 2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

//  ---------------------------------
//  | 0 4 8 |       |       |       |
//  | 1 5 9 |       |       |       |
//  | 2 6 A |       | c     |       | 0
//  | 3 7 B |       | o     |       |
//  |-----------------l-------------|    B
//  |       | rows  | u     |       |    A
//  |       |       | m     |       |    N
//  |       |       | n     |       | 1  D
//  |       |       | s     |       |    S
//  |-------------------------------|
//  |       |       |       |       |
//  |       |       |       |       |
//  |       |       |       |       | 2
//  |       |       |       |       |
//  ---------------------------------
//      0       1       2       3
//               STACKS

#include <bignumMT.h>
#include <iListNew.h>
#include <timer.h>
#include <profile.h>
#include <rope.h>
#include <general.h>
#include <symmetrytable.h>

#include "sudokuda.h"

#include <vector>
#include <array>
#include <mutex>
#include <fstream>
#include <cmath>
#include <filesystem>   // requires C++17

using namespace BillGeneral;

typedef NumList<uint64_t> IList64;

// ***********************
// *                     *
// *  Utility Functions  *
// *                     *
// ***********************

template<typename T>
inline T minEq(T& dst, T x)
{
  dst = std::min(dst, x);
  return dst;
}

template<typename T>
inline void sort(T& lo, T& hi)
{
  // Hopefully the compiler will generate branchless min/max instructions, to avoid the
  // high cost of branches that would have worst-case branch predication success
  T tmp = std::min(lo, hi);
  hi = std::max(lo, hi);
  lo = tmp;
}

template<typename T>
constexpr T factorial(int n)
{
  T f = 1;
  for (int i = 2; i <= n; ++i)
    f *= (T)i;
  return f;
}

template<typename T>
constexpr T ipower(T n, uint32_t e)
{
  T p = 1;
  while (true)
  {
    if ((e & 1) != 0)
      p *= n;
    e >>= 1;
    if (e != 0)
      n *= n;
    else
      break;
  }
  return p;
}

template<typename T>
constexpr T combin(int n, int s)
{
  T c = 1;
  for (int i = 0; i < s; ++i, --n)
    c *= (T)n;
  return c / factorial<T>(s);
}

inline void indentLine(int indent)
{
  printf("%*s", indent, "");
}

std::string commas(uint64_t n)
{
  if (n == 0)
    return "0";

  int triplets[8];
  int stackIndex = 0;
  while (n > 0)
  {
    triplets[stackIndex++] = n % 1000;
    n /= 1000;
  }

  std::string s;
  const char* tripletFormat = "%d";
  while (stackIndex > 0)
  {
    s += strFormat(tripletFormat, triplets[--stackIndex]);
    if (stackIndex > 0)
      s += ',';
    tripletFormat = "%03d";
  }

  return s;
}

// ***************
// *             *
// *  Box Codes  *
// *             *
// ***************
//
// While there are 12! boxes, there are only 12! / (3!^4 * 4!) distinct ways a box can
// influence another box in the same band, and 12! / (4!^3 * 3!) ways to influence a
// box in the same stack. Each of these ways is given a unique uint16_t code of type
// RowCode = BoxCode<false> for bands and ColCode = BoxCode<true> for stacks. The use
// of these template classes instead of raw uint16_t values has a number of advantages:
//
// * The compiler enforces proper use of the values, preventing programming errors
//   caused by confusing band and stack, as well as all of the other uses for
//   integer values throughout the code.
// * The types serve as documentation for the reader.
// * The template provides effieient versions of both code types with mostly the same
//   source code and only a few specializations, avoiding the sin of copy/paste
//   programming and all of the opportunities for bugs that come with it.

template<bool COL>
class BoxSets;

template<bool COL>
class BoxCode
{
public:
  static constexpr int Count =
    factorial<uint32_t>(12) / (COL ? ipower<uint32_t>(factorial<uint32_t>(4), 3) * factorial<uint32_t>(3)
                                   : ipower<uint32_t>(factorial<uint32_t>(3), 4) * factorial<uint32_t>(4));

  // Initialize the list of BoxSets corresponding to each BoxCode. Throws
  // std::runtime_error if the wrong number of codes are found or the canonical box does
  // not have code 0 (both would be due to bugs).
  static void init();

  // Default uninitialized construction is allowed so that large tables can be constructed
  // without wasting time on pointless initializations.
  BoxCode() = default;

  // An integer can only be converted to a BoxCode explicitly. Otherwise the compiler type checking
  // would be defeated.
  explicit BoxCode(int code) : code_((uint16_t)code) {}

  // A BoxSet can be converted to its code implicitly. It is always safe to do so. Time
  // is O(log(Count)). Throws std::runtime_error if sets is not in canonical form.
  BoxCode(const BoxSets<COL>& sets);

  // Retrieve the uint16_t code when needed.
  uint16_t operator()() const { return code_; }

  // Operator overloads to make BoxCode work like an integer.
  bool operator< (BoxCode code) const { return code_ < code.code_; }

  bool operator<=(BoxCode code) const { return code_ <= code.code_; }

  bool operator!=(BoxCode code) const { return code_ != code.code_; }

  bool operator==(BoxCode code) const { return code_ == code.code_; }

  BoxCode& operator++() { ++code_; return *this; }

  // Is the code in a valid range? Used to terminate for loops.
  bool isValid() const { return code_ < Count; }

private:
  // It is necessary that code 0 be the canonical box. This happens naturally (i.e.
  // sort order) for column codes, but not row codes. The following value is used
  // as an offset for row codes, and was determined emperically.
  static constexpr uint16_t canonicalRowCode = 13439;

  // The code
  uint16_t code_;

  // The sorted list of BoxSets corresponding to the codes
  static std::vector<uint64_t> canon_;

  // Get the BoxSets corresponding to code i. O(1)
  static BoxSets<COL> canon(int i);

  friend class BoxSets<COL>;
};

using RowCode = BoxCode<false>;
using ColCode = BoxCode<true>;

template<bool COL>
std::vector<uint64_t> BoxCode<COL>::canon_;

template<bool COL>
BoxSets<COL> BoxCode<COL>::canon(int i)
{
  if (!COL)
    i = (i + canonicalRowCode) % Count;
  return BoxSets<COL>(canon_[i]);
}

template<bool COL>
BoxCode<COL>::BoxCode(const BoxSets<COL>& sets)
{
  auto p = std::lower_bound(canon_.begin(), canon_.end(), sets());
  if (p != canon_.end() && *p == sets())
  {
    size_t index = p - canon_.begin();
    if (!COL)
      index = (index + (Count - canonicalRowCode)) % Count;
    code_ = (uint16_t)index;
  }
  else
    throw std::runtime_error("Sets not found in canon");
}

// **************
// *            *
// *  Box Sets  *
// *            *
// **************
//
// A RowSets = BoxSets<false> is four sets, each containing three elements drawn from the
// twelve symbols. Being sets, order doesn't matter, since we are only considering
// influence on other boxes in the band. Likewise, a ColSets = BoxSets<true> is three
// sets of four elements. Sets are represented as 12-bit fields in a uint64_t. With
// this representation, compatibility checking is trivial and extremely fast.
//
// Sometimes the sets have a fewer or greater number of elements. For example, two
// compatible sets can be combined by logical OR.
//
// A BoxSets is in canonical form if the sets are in non-descending order from MSBs to
// LSBs.
//
// The template class has the same advantages as described above for BoxCode.

template<bool COL>
class BoxSets
{
public:
  static constexpr int N = COL ? 3 : 4;

  // The sets of the canonical box. Note: avoid the ull suffix to make uint64_t literals,
  // because the resulting type is not portable. In particular, Microsoft and GCC
  // disagree.
  static constexpr BoxSets canonSets() { return BoxSets((uint64_t)(COL ? 0x00F0F0F00 : 0x111222444888)); }

  // Default uninitialized construction is allowed so that large tables can be constructed
  // without wasting time on pointless initializations.
  BoxSets() = default;

  // An integer can only be converted to a BoxSets explicitly. Otherwise the compiler type checking
  // would be defeated.
  explicit constexpr BoxSets(uint64_t sets) : sets_(sets) {}

  BoxSets(int) = delete;   // prevent the above constructor from being used for ints

  // Construct a BoxSets<true> from three sets. This is implemented as a template specialization;
  // a specialization for BoxSets<false> is not defined and so cannot accidentially be used.
  BoxSets(int col0, int col1, int col2);

  // Construct a BoxSets<false> from four sets. This is implemented as a template specialization;
  // a specialization for BoxSets<true> is not defined and so cannot accidentially be used.
  BoxSets(int row0, int row1, int row2, int row3);

  // A BoxCode<COL> can be implicitly converted to a BoxSets<COL>. O(1)
  BoxSets(const BoxCode<COL>& code) : sets_(BoxCode<COL>::canon(code())()) {}

  // Convert this BoxSets to canonical form.
  BoxSets makeCanonical() const;

  // Retrieve the uint64_t value when needed.
  uint64_t operator()() const { return sets_; }

  // Set intersection
  BoxSets operator&(const BoxSets& sets) const { return BoxSets(sets_ & sets.sets_); }

  // Set union
  BoxSets operator|(const BoxSets& sets) const { return BoxSets(sets_ | sets.sets_); }

  BoxSets& operator|=(const BoxSets& sets) { sets_ |= sets.sets_; return *this; }

  // Comparison operators
  bool operator==(const BoxSets& sets) const { return sets_ == sets.sets_; }

  bool operator!=(const BoxSets& sets) const { return sets_ != sets.sets_; }

  bool operator<(const BoxSets& sets) const { return sets_ < sets.sets_; }

  // Return the sets whose elemets are not in this sets
  BoxSets invert() const
  {
    return BoxSets(sets_ ^ (uint64_t)(COL ? 0xFFFFFFFFF : 0xFFFFFFFFFFFF));
  }

  // Two BoxSets b1 and b2 are compatible if (b1 & b2).isCompatible is true.
  bool isCompatible() const { return sets_ == 0; }

  // Return a printable representation
  std::string toString() const;

private:
  uint64_t sets_;
};

using RowSets = BoxSets<false>;
using ColSets = BoxSets<true>;

template<>
inline BoxSets<true>::BoxSets(int col0, int col1, int col2)
{
  sets_ = (uint64_t)col0 << 24 | (uint64_t)col1 << 12 | col2;
}

template<>
inline BoxSets<false>::BoxSets(int row0, int row1, int row2, int row3)
{
  sets_ = (uint64_t)row0 << 36 | (uint64_t)row1 << 24 | (uint64_t)row2 << 12 | row3;
}

template<>
BoxSets<false> BoxSets<false>::makeCanonical() const
{
  int r3 = (int)sets_ & 0xFFF;
  int r2 = ((int)sets_ >> 12) & 0xFFF;
  int r1 = (int)((sets_ >> 24) & 0xFFF);
  int r0 = (int)((sets_ >> 36) & 0xFFF);

  sort(r0, r1);
  sort(r2, r3);
  sort(r0, r2);
  sort(r1, r3);
  sort(r1, r2);
  return BoxSets<false>(r0, r1, r2, r3);
}

template<>
BoxSets<true> BoxSets<true>::makeCanonical() const
{
  int c2 = (int)sets_ & 0xFFF;
  int c1 = ((int)sets_ >> 12) & 0xFFF;
  int c0 = (int)((sets_ >> 24) & 0xFFF);

  if (c0 < c1)
  {
    if (c1 < c2)
      return *this;
    else
      if (c0 < c2)
        return BoxSets(c0, c2, c1);
      else
        return BoxSets(c2, c0, c1);
  }
  else
  {
    if (c0 < c2)
      return BoxSets(c1, c0, c2);
    else
      if (c1 < c2)
        return BoxSets(c1, c2, c0);
      else
        return BoxSets(c2, c1, c0);
  }
}

template<>
void BoxCode<false>::init()
{
  ProfileTree::start("RowCode Init");

  const int bB = 0x800;
  for (int bA = bB >> 1; bA; bA >>= 1)
    for (int b9 = bA >> 1; b9; b9 >>= 1)
    {
      int row3 = b9 | bA | bB;
      int b8 = bB >> 1;
      while (b8 & row3)
        b8 >>= 1;
      for (int b7 = b8 >> 1; b7; b7 >>= 1)
      {
        if (b7 & row3)
          continue;
        for (int b6 = b7 >> 1; b6; b6 >>= 1)
        {
          if (b6 & row3)
            continue;
          int row2 = b6 | b7 | b8;
          int row23 = row2 | row3;
          int b5 = bB >> 1;
          while (b5 & row23)
            b5 >>= 1;
          for (int b4 = b5 >> 1; b4; b4 >>= 1)
          {
            if (b4 & row23)
              continue;
            for (int b3 = b4 >> 1; b3; b3 >>= 1)
            {
              if (b3 & row23)
                continue;
              int row1 = b3 | b4 | b5;
              int row0 = (row1 | row23) ^ 0xFFF;
              canon_.push_back(BoxSets<false>(row0, row1, row2, row3)());
            }
          }
        }
      }
    }


  std::sort(canon_.begin(), canon_.end());

  if (canon_.size() != Count)
    throw std::runtime_error("Unexpected row canon count");

  if (BoxCode(BoxSets<false>((uint64_t)0x111222444888))() != 0)
    throw std::runtime_error("Unexpected canonical row code");

  ProfileTree::stop(Count);
}

template<>
void BoxCode<true>::init()
{
  ProfileTree::start("ColCode Init");
  const int bB = 0x800;
  for (int bA = bB >> 1; bA; bA >>= 1)
    for (int b9 = bA >> 1; b9; b9 >>= 1)
      for (int b8 = b9 >> 1; b8; b8 >>= 1)
      {
        int col2 = b8 | b9 | bA | bB;
        int b7 = bB >> 1;
        while (b7 & col2)
          b7 >>= 1;
        for (int b6 = b7 >> 1; b6; b6 >>= 1)
        {
          if (b6 & col2)
            continue;
          for (int b5 = b6 >> 1; b5; b5 >>= 1)
          {
            if (b5 & col2)
              continue;
            for (int b4 = b5 >> 1; b4; b4 >>= 1)
            {
              if (b4 & col2)
                continue;
              int col1 = b4 | b5 | b6 | b7;
              int col0 = (col1 | col2) ^ 0xFFF;
              canon_.push_back(BoxSets<true>(col0, col1, col2)());
            }
          }
        }
      }

  std::sort(canon_.begin(), canon_.end());

  if (canon_.size() != Count)
    throw std::runtime_error("Unexpected column canon count");

  if (BoxCode(BoxSets<true>((uint64_t)0x00F0F0F00))() != 0)
    throw std::runtime_error("Unexpected canonical column code");

  ProfileTree::stop(Count);
}

template<bool COL>
std::string BoxSets<COL>::toString() const
{
  std::string s;
  for (int i = N - 1; i >= 0; --i)
  {
    if (s.size() != 0)
      s += '.';
    s += strFormat("%03X", (int)(sets_ >> (12 * i)) & 0xFFF);
  }
  return s;
}

// **********************************************************
// *                                                        *
// *  Row Quadruplet, Column Triplet Permutation Generator  *
// *                                                        *
// **********************************************************
//
// Efficient permutations of row quadruplets and column triplets. Initializing a QP
// object takes some time, but executing one is fast, so the idea is to set one up and
// use it many times. Here "permutation" means a permutation of the twelve symbols.
// Each of the four sets of a quadruplet, or three sets of a triplet, is processed
// identically, since each chunk has a bit for each symbol and it is the symbols that
// are undergoing permutation. To execute a permutation, each bit independently is
// unchanged, shifted left, or shifted right.

class QP
{
  // Table of shift counts for the 24 permutations of the three columns of the
  // canonical box. Each permutation has 4 shift counts, for the 4 elements
  // of each column.
  static int ctab_[4][24];

  // Table of shift counts for the six permutations of the four rows of the
  // canonical box. Each permutation has 3 shift counts, for the 3 elements
  // of each row.
  static int rtab_[3][6];

  struct
  {
    int shift;      // amount to shift left or right
    uint64_t mask;  // select bits to shift by specified amount
  }
  pos_[11],         // bits to be shifted left
  neg_[11];         // bits to be shifted right
  int np_;          // number of entries in the left shift table
  int nn_;          // number of entries in the right shift table
  uint64_t zm_;     // select bits to be left unchanged

  void clear_() { np_ = nn_ = 0; zm_ = (uint64_t)0xFFFFFFFFFFFF;}
  void enter_(int shift, int bit);

public:
  // Set to map src quadruplet to dst quadruplet. Note that there may be many
  // permutations that would effect such a map, one is selected arbitrarily.
  // Note the use of templates to insure that one cannot accidentially map
  // between row sets and column sets
  template<bool COL>
  void rename(BoxSets<COL> src, BoxSets<COL> dst);

  // Set for the nth permutation of column c of the canonical box. The other columns
  // are set for no permutation. For example, col(1, n) sets permutation n of the elements
  // {4, 5, 6, 7}. col() handles 13,824 of the 82,944 canonical column permutations.
  // 0 <= n < 24, 0 <= c < 3
  void col(int c, int n);

  // Set for the nth permutation of all columns simultaneously, as opposed to col()
  // which permutes the elements of one column.
  // 0 <= n < 6
  void cols(int n);

  // Set for the nth permutation of row r of the canonical box. The other rows
  // are set for no permutation. For example, row(2, n) sets permutation n of the elements
  // {2, 6, A}. row() handles 1296 of the 31,104 canonical row permutations.
  // 0 <= n < 6, 0 <= r < 4
  void row(int r, int n);

  // Set for the nth permutation of all rows simultaneously, as opposed to row()
  // which permutes the elements of one row.
  // 0 <= n < 24
  void rows(int n);

  // Template alias of row or col for template functions that work both ways
  template<bool COL>
  void set(int s, int n);

  // Template alias of rows or cols for template functions that work both ways
  template<bool COL>
  void sets(int n);

  // Map quadruplet according to permutation that has been set. Neither argument nor result
  // need be in canonical form
  template<bool COL>
  BoxSets<COL> map(BoxSets<COL>) const;

  // Like map, except result is put in canonical form.
  template<bool COL>
  BoxSets<COL> mapc(BoxSets<COL> sets) const { return map(sets).makeCanonical(); }

  // Print to stdout a representation of this permutation
  void print(int indent = 0) const;
};

void QP::enter_(int shift, int bit)
{
  uint64_t mask = (uint64_t)0x001001001001 << bit;
#if 0
  // This version makes one entry per bit that needs shifting.
  if (shift > 0)
  {
    pos_[np_  ].shift = shift;
    pos_[np_++].mask = mask;
    zm_ &= ~(mask << shift);
  }
  else if (shift < 0)
  {
    neg_[nn_  ].shift = -shift;
    neg_[nn_++].mask = mask;
    zm_ &= ~(mask >> -shift);
  }
#else
  // This version combines bits whose shift counts are the same, resulting in slower
  // execution for enter_ but faster for doing a permutation. Seems slightly slower
  // overall.
  if (shift > 0)
  {
    zm_ &= ~(mask << shift);
    for (int i = 0; i < np_; ++i)
      if (pos_[i].shift == shift)
      {
        pos_[i].mask |= mask;
        return;
      }
    pos_[np_  ].shift = shift;
    pos_[np_++].mask  = mask;
  }
  else if (shift < 0)
  {
    zm_ &= ~(mask >> -shift);
    for (int i = 0; i < nn_; ++i)
      if (neg_[i].shift == shift)
      {
        neg_[i].mask |= mask;
        return;
      }
    neg_[nn_  ].shift = -shift;
    neg_[nn_++].mask  =  mask;
  }
#endif
}

template<bool COL>
BoxSets<COL> QP::map(BoxSets<COL> sets) const
{
  uint64_t dst = sets() & zm_;
  for (int i = 0; i < np_; ++i)
    dst |= (sets() & pos_[i].mask) << pos_[i].shift;
  for (int i = 0; i < nn_; ++i)
    dst |= (sets() & neg_[i].mask) >> neg_[i].shift;
  return BoxSets<COL>(dst);
}

template<bool COL>
void QP::rename(BoxSets<COL> srcSets, BoxSets<COL> dstSets)
{
  clear_();

  uint64_t src = srcSets();
  uint64_t dst = dstSets();

  int s = 0, d = 0;
  for (int i = 0; i < 12; ++i)   // do all 12 bits of src and dst
  {
    while (!(src & 1))          // find next src bit
    {
      src >>= 1;
      ++s;
    }
    while (!(dst & 1))          // find next dst bit
    {
      dst >>= 1;
      ++d;
    }
    enter_(d - s, s % 12);      // map src to dst
    src >>= 1;                  // toss out the bits we just did
    dst >>= 1;
    ++s;
    ++d;
  }
}

// Bit shift counts for 24 permutations of three columns of canonical box
int QP::ctab_[4][24] =
{
  { 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3 },
  { 0,  0,  1,  1,  2,  2, -1, -1,  1,  1,  2,  2, -1, -1,  0,  0,  2,  2, -1, -1,  0,  0,  1,  1 },
  { 0,  1, -1,  1, -1,  0,  0,  1, -2,  1, -2,  0, -1,  1, -2,  1, -2, -1, -1,  0, -2,  0, -2, -1 },
  { 0, -1,  0, -2, -1, -2,  0, -1,  0, -3, -1, -3,  0, -2,  0, -3, -2, -3, -1, -2, -1, -3, -2, -3 }
};

// Bit shift counts for six permutations of four rows of canonical box
int QP::rtab_[3][6] =
{
  { 0,  0,  1,  1,  2,  2},
  { 0,  1, -1,  1, -1,  0},
  { 0, -1,  0, -2, -1, -2}
};

void QP::col(int c, int n)
{
  clear_();
  for (int r = 0; r < 4; ++r)
    enter_(ctab_[r][n], 4 * c + r);
}

void QP::cols(int n)
{
  clear_();
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 4; ++r)
      enter_(4 * rtab_[c][n], 4 * c + r);
}

void QP::row(int r, int n)
{
  clear_();
  for (int c = 0; c < 3; ++c)
    enter_(4 * rtab_[c][n], 4 * c + r);
}

void QP::rows(int n)
{
  clear_();
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 4; ++r)
      enter_(ctab_[r][n], 4 * c + r);
}

template<>
inline void QP::set<false>(int s, int n)
{
  row(s, n);
}

template<>
inline void QP::set<true>(int s, int n)
{
  col(s, n);
}

template<>
inline void QP::sets<false>(int n)
{
  rows(n);
}

template<>
inline void QP::sets<true>(int n)
{
  cols(n);
}

void QP::print(int indent) const
{
  indentLine(indent);
  printf(" 0 1 2 3 4 5 6 7 8 9 A B\n");
  indentLine(indent);
  for (int i = 0; i < 12; ++i)
  {
    int j = i;
    for (int s = 0; s < np_; ++s)
      if (((pos_[s].mask >> i) & 1) != 0)
      {
        j = i + pos_[s].shift;
        break;
      }
    for (int s = 0; s < nn_; ++s)
      if (((neg_[s].mask >> i) & 1) != 0)
      {
        j = i - neg_[s].shift;
        break;
      }
    printf("%2X", j);
  }
  printf("\n");
}

// ************************
// *                      *
// *  Permutation Tables  *
// *                      *
// ************************
//
// This class uses QP to create static permutation lookup tables needed for rapidly
// finding gangsters and determining their properties. Unlike QP it works with BoxCodes
// instead of BoxSets. Note that since the grid enumeration is based on band gangsters,
// only the COL version (ColTables) is actually used in the counting. The !COL version
// (RowTables) could be used to find stack gangsters, and was so used early in the
// development when it was still unclear which approach would be best. Having both
// versions requires specializing init(), but all other code is shared.

template<bool COL>
class PermutationTables
{
public:
  static constexpr int Codes     = BoxCode< COL>::Count;
  static constexpr int Sets      = BoxSets< COL>::N;
  static constexpr int SetSize   = BoxSets<!COL>::N;
  static constexpr int SetPerms  = factorial<int>(SetSize);
  static constexpr int SetsPerms = factorial<int>(Sets);

  // Initialize all tables
  static void init();

  // Given a box specified by code, return the code of the box after the specified
  // permutaion of the specified set of the symbols of the canonical box. This is
  // the BoxCode version of QP::set.
  static BoxCode<COL> mapSet(int set, int permutation, BoxCode<COL> code)
  {
    return setTable_[set][permutation][code()];
  }

  // Given a box specified by code, return the code of the box after the specified
  // permutaion of all sets of the symbols of the canonical box. This is the BoxCode
  // version of QP::sets.
  static BoxCode<COL> mapSets(int permutation, BoxCode<COL> code)
  {
    return setsTable_[permutation][code()];
  }

  // Return a partial sets, containing only the specified set of a box specified
  // by code, and permuted as specified by perm.
  // 0 <= code < Codes, 0 <= set < Sets, 0 <= perm < SetPerms
  static BoxSets<!COL> flipSets(BoxCode<COL> code, int set, int perm)
  {
    return flipTable_[code()][set][perm];
  }

  // Perform some consistency checks on the tables, print report to stdout if any
  // errors found.
  static void verify();

private:
  static BoxCode<COL> setTable_[Sets][SetPerms][Codes];
  static BoxCode<COL> setsTable_[SetsPerms][Codes];

  static BoxSets<!COL> flipTable_[Codes][Sets][SetPerms];
};

using RowTables = PermutationTables<false>;
using ColTables = PermutationTables<true>;

template<bool COL>
BoxCode<COL> PermutationTables<COL>::setTable_[Sets][SetPerms][Codes];

template<bool COL>
BoxCode<COL> PermutationTables<COL>::setsTable_[SetsPerms][Codes];

template<bool COL>
BoxSets<!COL> PermutationTables<COL>::flipTable_[Codes][Sets][SetPerms];

template<>
void RowTables::init()
{
  // Compute rowTable_[4][6][RowCode::Count]
  ProfileTree::start("Row Tables");
  QP qp;

  for (int r = 0; r < Sets; ++r)
  {
    // For all rows r and RowCodes i, permutation 0 maps the box to itself
    for (int i = 0; i < Codes; ++i)
      setTable_[r][0][i] = RowCode(i);

    // For the other 5 permutations p, get the RowSets corresponding to RowCode i, map it
    // to sets in canonical form, then convert back to code code
    for (int p = 1; p < SetPerms; ++p)
    {
      qp.row(r, p);
      for (int i = 0; i < Codes; ++i)
        setTable_[r][p][i] = RowCode(qp.mapc(RowSets(RowCode(i))));
    }
  }

  // Same as above, but for rowsTable_[24][RowCode::Count]
  for (int i = 0; i < Codes; ++i)
    setsTable_[0][i] = RowCode(i);

  for (int p = 1; p < SetsPerms; ++p)
  {
    qp.rows(p);
    for (int i = 0; i < Codes; ++i)
      setsTable_[p][i] = RowCode(qp.mapc(RowSets(RowCode(i))));
  }

  // Flip table. First do the canonical box. 
  for (int r = 0; r < Sets; ++r)
  {
    ColSets sets((uint64_t)0x001010100 << r);
    for (int p = 0; p < SetPerms; ++p)
    {
      qp.row(r, p);
      flipTable_[0][r][p] = qp.map(sets);
    }
  }

  // Then do the others by mapping from the canonical box
  for (int i = 1; i < Codes; ++i)
  {
    qp.rename<false>(RowSets::canonSets(), RowCode(i));
    for (int r = 0; r < Sets; ++r)
      for (int p = 0; p < SetPerms; ++p)
        flipTable_[i][r][p] = qp.map(flipTable_[0][r][p]);
  }

  ProfileTree::stop(24 * Codes);
}

template<>
void ColTables::init()
{
  ProfileTree::start("Column Tables");
  QP qp;

  for (int c = 0; c < Sets; ++c)
  {
    // For all columns c and ColCodes i, permutation 0 maps the box to itself
    for (int i = 0; i < Codes; ++i)
      setTable_[c][0][i] = ColCode(i);

    // For the other 23 permutations p, get the ColSets corresponding to ColCode i, map it
    // to sets in canonical form, then convert back to code code
    for (int p = 1; p < SetPerms; ++p)
    {
      qp.col(c, p);
      for (int i = 0; i < Codes; ++i)
        setTable_[c][p][i] = ColCode(qp.mapc(ColSets(ColCode(i))));
    }
  }

  // Same as above, but for setsTable_[SetsPerms][Codes]
  for (int i = 0; i < Codes; ++i)
    setsTable_[0][i] = ColCode(i);

  for (int p = 1; p < SetsPerms; ++p)
  {
    qp.cols(p);
    for (int i = 0; i < Codes; ++i)
      setsTable_[p][i] = ColCode(qp.mapc(ColSets(ColCode(i))));
  }

  // Flip table. First do the canonical box. 
  for (int c = 0; c < Sets; ++c)
  {
    RowSets sets((uint64_t)0x001002004008 << (4 * c));
    for (int p = 0; p < SetPerms; ++p)
    {
      qp.col(c, p);
      flipTable_[0][c][p] = qp.map(sets);
    }
  }

  // Then do the others by mapping from the canonical box
  for (int i = 1; i < Codes; ++i)
  {
    qp.rename<true>(ColSets::canonSets(), ColCode(i));
    for (int c = 0; c < Sets; ++c)
      for (int p = 0; p < SetPerms; ++p)
        flipTable_[i][c][p] = qp.map(flipTable_[0][c][p]);
  }

  ProfileTree::stop(24 * Codes);
}

template<bool COL>
void PermutationTables<COL>::verify()
{
  QP qp;
  int errors = 0;

  for (int s = 0; s < Sets; ++s)
    for (int p = 0; p < SetPerms; ++p)
    {
      qp.set<COL>(s, p);
      for (int i = 0; i < Codes; ++i)
      {
        BoxCode<COL> code = PermutationTables<COL>::mapSet(s, p, BoxCode<COL>(i));
        if (BoxSets<COL>(code)() != qp.mapc(BoxSets<COL>(BoxCode<COL>(i)))())
          ++errors;
      }
    }
  if (errors > 0)
    printf("%d %sTable errors\n", errors, COL ? "col" : "row");

  errors = 0;
  for (int p = 0; p < SetsPerms; ++p)
  {
    qp.sets<COL>(p);
    for (int i = 0; i < Codes; ++i)
    {
      BoxCode<COL> code = PermutationTables<COL>::mapSets(p, BoxCode<COL>(i));
      if (BoxSets<COL>(code)() != qp.mapc(BoxSets<COL>(BoxCode<COL>(i)))())
        ++errors;
    }
  }
  if (errors > 0)
    printf("%d %ssTable errors\n", errors, COL ? "col" : "row");
}

// ***********************
// *                     *
// *  Permutation Nodes  *
// *                     *
// ***********************
//
// Nodes make the finding of gangsters, and the grid enumeration, possible within the
// time and memory constraints of available machines circa 2022. The box codes
// comprise equivalence classes obtained by the canonical permutations (those that map
// code 0 to code 0). Each class has a representative member defined as the lowest code
// in the class. For each of the codes, a Node gives the identity of representative
// member of its class, and a list of which of the permutations map the code to its
// representative (and also code 0 to 0).
//
// Note that the product of the class size and the permutation list size equals the
// total number of canonical permutations. Since code 0 corresponds to the canonical
// box, all permutations by definition are in the list and the class has only the one
// element. The Node for this class doesn't store the list since it's large and there's
// no need.
//
// These nodes define the primary gangster configurations. This is why there are 9
// for band gangsters. Every band gangster has box 0 code 0 and box 1 code [0 .. 8].
//
// Here is some information about the classes:
//
// Row Nodes (12 classes, 6^4 * 24 = 31104 permutations)
// class RowCode     RowSets       Size  Permutations
//   0:     0     111.222.444.888     1     31104
//   1:     1     111.222.448.884    54       576
//   2:     5     111.224.40A.8C0   648        48
//   3:     7     111.224.448.882   216       144
//   4:    32     111.248.482.824   144       216
//   5:    71     112.205.428.8C0  3888         8
//   6:    72     112.205.448.8A0  3888         8
//   7:    86     112.20C.481.860  3888         8
//   8:    89     112.221.40C.8C0   243       128
//   9:    97     112.224.448.881   486        64
//  10:   113     112.244.409.8A0   648        48
//  11:   530     124.209.482.850  1296        24
//
// Column Nodes (9 classes, 24^3 * 6 = 82,944 permutations)
// class ColCode     ColSets     Size  Permutations
//   0:     0       00F.0F0.F00     1    82944
//   1:     1       00F.170.E80    48     1728
//   2:     9       00F.330.CC0    54     1536
//   3:    36       017.168.E80   576      144
//   4:    39       017.1E0.E08   128      648
//   5:    44       017.328.CC0  1728       48
//   6:   324       033.30C.CC0   216      384
//   7:   325       033.344.C88  1296       64
//   8:  2537       113.264.C88  1728       48
//
// As above for PermutationTables, the COL version is used and the !COL version could
// be used for a scheme based on stack gangsters. Only one function must be specialized,
// all other code is shared between COL and !COL.

template<bool COL>
class CodeNode
{
public:
  // Specify one permutation that maps a code to its class representative and also
  // maps code 0 to 0.
  struct PermutationSet
  {
    uint8_t set[BoxSets<COL>::N];
    uint8_t sets;

    // Execute the specified permutation.
    BoxCode<COL> permute(BoxCode<COL>) const;
  };

  // Find all the nodes. Throws std::runtime_error if a code can't find its
  // class representative (due to bugs).
  static void init();

  // Perform some consistency checks, print report to stdout if any errors found.
  static void verify();

  // Print the above "information about the classes".
  static void print();

  // A list of the class reps and the size of each class
  static NumList<BoxCode<COL>> nodeList;

  // For the specified code, return its node.
  static const CodeNode& nodes(BoxCode<COL> code) { return nodes_[code()]; }

  // The box code of the class representative
  BoxCode<COL> repCode = BoxCode<COL>(std::numeric_limits<uint16_t>::max());

  // The index of the class representative
  int repIndex;

  // The list of permutations that map this node to its class rep
  std::vector<PermutationSet> permutations;

private:
  // Table of nodes
  static CodeNode nodes_[BoxCode<COL>::Count];

  // Find all the permtations
  static void allPermutations_(int codeIndex);

  // Enter each found permutation, determine class rep
  void enter_(BoxCode<COL> code, const PermutationSet& pset);
};

using RowNode = CodeNode<false>;
using ColNode = CodeNode<true>;

template<bool COL>
CodeNode<COL> CodeNode<COL>::nodes_[BoxCode<COL>::Count];

template<bool COL>
NumList<BoxCode<COL>> CodeNode<COL>::nodeList;

template<>
void CodeNode<false>::allPermutations_(int rowCodeIndex)
{
  PermutationSet pset;
  for (pset.set[0] = 0; pset.set[0] < 6; ++pset.set[0])
  {
    RowCode r0 = RowTables::mapSet(0, pset.set[0], RowCode(rowCodeIndex));
    for (pset.set[1] = 0; pset.set[1] < 6; ++pset.set[1])
    {
      RowCode r1 = RowTables::mapSet(1, pset.set[1], r0);
      for (pset.set[2] = 0; pset.set[2] < 6; ++pset.set[2])
      {
        RowCode r2 = RowTables::mapSet(2, pset.set[2], r1);
        for (pset.set[3] = 0; pset.set[3] < 6; ++pset.set[3])
        {
          RowCode r3 = RowTables::mapSet(3, pset.set[3], r2);
          for (pset.sets = 0; pset.sets < 24; ++pset.sets)
          {
            RowCode r4 = RowTables::mapSets(pset.sets, r3);
            nodes_[rowCodeIndex].enter_(r4, pset);
          }
        }
      }
    }
  }
}

template<>
void CodeNode<true>::allPermutations_(int colCodeIndex)
{
  PermutationSet pset;
  for (pset.set[0] = 0; pset.set[0] < 24; ++pset.set[0])
  {
    ColCode c0 = ColTables::mapSet(0, pset.set[0], ColCode(colCodeIndex));
    for (pset.set[1] = 0; pset.set[1] < 24; ++pset.set[1])
    {
      ColCode c1 = ColTables::mapSet(1, pset.set[1], c0);
      for (pset.set[2] = 0; pset.set[2] < 24; ++pset.set[2])
      {
        ColCode c2 = ColTables::mapSet(2, pset.set[2], c1);
        for (pset.sets = 0; pset.sets < 6; ++pset.sets)
        {
          ColCode c3 = ColTables::mapSets(pset.sets, c2);
          nodes_[colCodeIndex].enter_(c3, pset);
        }
      }
    }
  }
}

template<bool COL>
void CodeNode<COL>::init()
{
  ProfileTree::start(COL ? "Column Nodes" : "Row Nodes");
  for (int index = 1; index < BoxCode<COL>::Count; ++index)
  {
    allPermutations_(index);
    nodeList.enter(nodes_[index].repCode);
  }
  nodes_[0].repCode = BoxCode<COL>(0);
  nodeList.enter(BoxCode<COL>(0));
  nodeList.sort();

  for (int i = 0; i < BoxCode<COL>::Count; ++i)
  {
    size_t index;
    if (nodeList.lookup(nodes_[i].repCode, index))
      nodes_[i].repIndex = (int)index;
    else
      throw std::runtime_error("Can't find node code");
  }

  ProfileTree::stop(1296 * 24);
}

template<bool COL>
void CodeNode<COL>::verify()
{
  constexpr int maxErrors = 8;
  int errors = 0;

  for (BoxCode<COL> r(0); r.isValid(); ++r)
  {
    const CodeNode& node = CodeNode::nodes(r);

    for (const PermutationSet& perm : node.permutations)
    {
      BoxCode<COL> repCode = perm.permute(r);
      if (repCode != node.repCode)
      {
        if (++errors <= maxErrors)
        {
          printf("Node %d permutation ", r());
          for (uint8_t p : perm.set)
            printf("%d.", p);
          printf(" maps to %d, expected %d\n", repCode(), node.repCode());
        }
      }
    }
  }

  if (errors > 0)
    printf("%d total node errors\n", errors);
}

template<bool COL>
void CodeNode<COL>::print()
{
  printf("\n%s Nodes\n", COL ? "Column" : "Row");
  QP qp;
  for (int node = 0; node < (int)nodeList.size(); ++node)
  {
    BoxCode<COL> code = nodeList[node];
    int permutations = (int)nodes(code).permutations.size();
    BoxSets<COL> sets = BoxCode<COL>(code);

    qp.rename(sets, BoxSets<COL>::canonSets());
    BoxCode<COL> renamedCode0 = qp.mapc(BoxSets<COL>::canonSets());

    printf("  %2d: %4d  %s  %4d  %4d  -> %4d -> %4d\n",
           node, code(), sets.toString().c_str(), nodeList.count(node), permutations,
           renamedCode0(), nodes(renamedCode0).repCode());
  }
}

template<bool COL>
void CodeNode<COL>::enter_(BoxCode<COL> code, const PermutationSet& pset)
{
  if (code <= repCode)
  {
    if (code < repCode)
    {
      repCode = code;
      permutations.clear();
    }

    permutations.push_back(pset);
  }
}

template<bool COL>
BoxCode<COL> CodeNode<COL>::PermutationSet::permute(BoxCode<COL> code) const
{
  for (int i = 0; i < BoxSets<COL>::N; ++i)
    code = PermutationTables<COL>::mapSet(i, set[i], code);
  return PermutationTables<COL>::mapSets(sets, code);
}

// ******************************************
// *                                        *
// *  Sets and Codes Compatible with a Box  *
// *                                        *
// ******************************************

template<bool COL>
class BoxCompatible
{
public:
  // Emperically determined
  static constexpr int Count = COL ? 346 : 13833;

  // Initialize static tables. Throws std::runtime_error if the number of compatible
  // boxes found does not match Count. Shared by both versions.
  static void init();

  // List of Count BoxSets compatible with the canonical box
  static std::vector<BoxSets<COL>> canon;

  // Return the index of sets in canon. Throws std::runtime_error if sets not
  // in canon. O(log(Count))
  static int canonCode(BoxSets<COL> sets);

  // Make a list of BoxSets that are compatible with the specified sets. Throws
  // std::runtime_error if the number of compatible boxes found does not match Count
  static void makeSets(BoxSets<COL> sets, std::vector<BoxSets<COL>>& list);

  // Perform some consistency checks, print report to stdout if any errors found.
  static void verify();

private:
  // Specialized to initialize oneSet_ for COL or !COL
  static void makeOneSet_();

  // These 12-bit sets hold all possible subsets of the 12 symbols that are of
  // size 3 (COL) or 4 (!COL, i.e. rows).
  static int oneSet_[combin<int>(12, BoxSets<!COL>::N)];
};

using RowCompatible = BoxCompatible<false>;
using ColCompatible = BoxCompatible<true>;

template<bool COL>
int BoxCompatible<COL>::oneSet_[combin<int>(12, BoxSets<!COL>::N)];

template<bool COL>
std::vector<BoxSets<COL>> BoxCompatible<COL>::canon;

template<>
void RowCompatible::makeOneSet_()
{
  int r = 0;
  for (int i = 0; i < 10; ++i)
    for (int j = i + 1; j < 11; ++j)
      for (int k = j + 1; k < 12; ++k)
        oneSet_[r++] = (1 << i) | (1 << j) | (1 << k);
}

template<>
void RowCompatible::makeSets(RowSets sets, std::vector<RowSets>& list)
{
  constexpr int N = combin<int>(12, 3);

  int set[4];
  uint64_t bits = sets();
  for (int i = 3; i >= 0; --i, bits >>= 12)
    set[i] = (int)bits & 0xFFF;

  list.clear();
  for (int i0 = 0; i0 < N; ++i0)
  {
    int r0 = oneSet_[i0];
    if ((r0 & set[0]) == 0)
      for (int i1 = 0; i1 < N; ++i1)
      {
        int r1 = oneSet_[i1];
        if (((r0 | set[1]) & r1) == 0)
        {
          for (int i2 = 0; i2 < N; ++i2)
          {
            int r2 = oneSet_[i2];
            if (((r0 | r1 | set[2]) & r2) == 0)
            {
              int r3 = (r0 | r1 | r2) ^ 0xFFF;
              if ((r3 & set[3]) == 0)
                list.push_back(RowSets(r0, r1, r2, r3));
            }
          }
        }
      }
  }

  if (list.size() != Count)
    throw std::runtime_error("Unexpected row compatible count");
}

template<>
void ColCompatible::makeOneSet_()
{
  int c = 0;
  for (int r0 = 0; r0 < 9; ++r0)
    for (int r1 = r0 + 1; r1 < 10; ++r1)
      for (int r2 = r1 + 1; r2 < 11; ++r2)
        for (int r3 = r2 + 1; r3 < 12; ++r3)
          oneSet_[c++] = (1 << r0) | (1 << r1) | (1 << r2) | (1 << r3);
}

template<>
void ColCompatible::makeSets(ColSets sets, std::vector<ColSets>& list)
{
  constexpr int N = combin<int>(12, 4);

  int set[3];
  uint64_t bits = sets();
  for (int i = 2; i >= 0; --i, bits >>= 12)
    set[i] = (int)bits & 0xFFF;

  list.clear();
  for (int i0 = 0; i0 < N; ++i0)
  {
    int c0 = oneSet_[i0];
    if ((c0 & set[0]) == 0)
      for (int i1 = 0; i1 < N; ++i1)
      {
        int c1 = oneSet_[i1];
        if (((c0 | set[1]) & c1) == 0)
        {
          int c2 = (c0 | c1) ^ 0xFFF;
          if ((c2 & set[2]) == 0)
            list.push_back(ColSets(c0, c1, c2));
        }
      }
  }

  if (list.size() != Count)
    throw std::runtime_error("Unexpected column compatible count");
}

template<bool COL>
void BoxCompatible<COL>::init()
{
  ProfileTree::start("BoxCompatible Init");
  makeOneSet_();
  makeSets(BoxSets<COL>::canonSets(), canon);
  std::sort(canon.begin(), canon.end());
  ProfileTree::stop(combin<int>(12, BoxSets<COL>::N));
}

template<bool COL>
int BoxCompatible<COL>::canonCode(BoxSets<COL> sets)
{
  auto p = std::lower_bound(canon.begin(), canon.end(), sets);
  if (p != canon.end() && *p == sets)
    return (int)(p - canon.begin());

  throw std::runtime_error("Sets not found in BoxCompatible::canon");
}

template<bool COL>
void BoxCompatible<COL>::verify()
{
  int errors = 0;
  for (const BoxSets<COL>& sets : canon)
    if ((sets & BoxSets<COL>::canonSets())() != 0)
      ++errors;
  if (errors > 0)
    printf("%d %s compatible errors\n", errors, COL ? "column" : "row");
}


// ********************
// *                  *
// *  Band Gangsters  *
// *                  *
// ********************
//
// Bands fall into equivalence classes under permutations that leave unchanged the
// structure of their compatibility with the other two bands:
//    * symbols (12!)
//    * elements of a column (4! each)
//    * columns of a box (3! each)
//    * boxes in the band (4!)
// Considering band compatibility, any band can be specified by the 5775 ColCodes of
// each of its four boxes (ColCodes ignore column elements order and column order in a
// box). Each equivalence class has a class representative, called a gangster (for
// historical reasons), defined as the lowest band in a lexigraphical ordering of the
// ColCodes. If the four boxes are b0, b1. b2, and b3, then for a gangster
//    * b0 must he ColCode 0
//    * b1 must be the ColCode of one of the 9 ColNodes. Any higher code for b1 could
//      be mapped to that ColCode by a permutation of the entire band, since by definition
//      of nodes such a permutation leaves code 0 unchanged.
//    * b2 <= b3
// This is why the gangCache is 9 levels of a 5775x5775 symmetry table (SymTable).
//
// Finding all the gangsters quickly relies on the above facts. We need find the class
// rep for 9 * 5776 * 5775 / 2 = 150,103,800 cases. When a gangster is found, it and
// up to 6 simple equivalents are stored in the gangCache, so that most of those cases
// are found in the cache and need not be recomputed. Only a little over a million
// cases miss the cache (< 1%), resulting in a little over four million calls to
// gangCode_, which has to actually consider lots of permutations. The actual numbers
// vary slightly when running multiple parallel threads, due to variations in the
// order in which gangsters are found and the cache is filled. Note that cache filling
// is thread-safe because if multiple threads try to fill the same cache element, they
// will be writing the same value and so write order doesn't matter.

// Determined emperically by Pettersen in 2006, verified here (the code finds the
// number by independent means).
constexpr int GangNum = 144578;

// Everything we know about a gangster. This structure is filled in from a variety
// of sources by BandGang::countAll and PCountFinder::find. It is is used to
//    * monitor progress during grid counting;
//    * compute the final result;
//    * create a human-readable record of every gangster;
//    * hold a translated version of Pettersen's results for comparison.
//
// The original Pettersen 2006 gangsters can be compared to the current (Silver 2022)
// ones. They are in a different order, so they can only be matched up by comparing
// nodeIndex, gangMembers, and bandCount. The two gangster sets must just be
// permutations of each other, because math is math. nodeIndex is an odd case because
// the two versions agree exactly on 7 of the 9 (what I call) nodeIndex values, but two
// are intermngled and have different sizes (sum of the two sizes is the same). The
// key to efficient comparison is to define a lexigraphic sort order on nodeIndex
// (slightly modified), gangMembers, and bandCount. Sort is then O(N logN) and
// comparison is O(N).
struct Gangster
{
  int      nodeIndex;     // Which of the 9 primary gangster configurations [0 .. 8]
  uint32_t gangIndex;     // [0 .. 144577]
  uint32_t gangMembers;   // equivalence class size
  uint32_t bandCount;     // band configurations of each member of the class
  uint64_t gridCount;     // band 1 & 2 configurations given this gangster in band 0
  double   time;          // min execution time among machines that counted this one

  // Lexigraphic sort order
  bool operator<(const Gangster& g) const
  {
    int ni0 = fixTable_[nodeIndex];
    int ni1 = fixTable_[g.nodeIndex];
    if (ni0 < ni1)
      return true;

    if (ni0 == ni1)
    {
      if (bandCount < g.bandCount)
        return true;

      if (bandCount == g.bandCount)
      {
        if (gangMembers < g.gangMembers)
          return true;

        if (gangMembers == g.gangMembers)
          return gridCount < g.gridCount;
      }
    }

    return false;
  }

  // Gangsters are the same if gamgMembers and bandCount match
  bool same(const Gangster& g) const
  {
    return gangMembers == g.gangMembers && bandCount == g.bandCount;
  }

  // Gangsters are equivalent if they are same() and gridCount matches
  bool equivalent(const Gangster& g) const
  {
    return same(g) && gridCount == g.gridCount;
  }

  // Write this to specified file
  void write(FILE* file) const;

  // Write specified list of these to file of specified name. File created or
  // overwritten. Throws std::runtime_error if file can't be opened.
  static void writeAll(const std::vector<Gangster>&, const std::string& filename);

  // Count and print number of duplcate gangsters, meaning same(), in specified
  // sorted list. Emperically there are none.
  static void duplicates(const std::vector<Gangster>&);

  // Count and print number of !equivalent() gangsters in sorted lists
  static void compare(const std::vector<Gangster>& pGang, const std::vector<Gangster>& sGang);

private:
  // Treat the two mismatched nodeIndex cases as one combined case
  static int fixTable_[9];
};

int Gangster::fixTable_[9] = { 0, 1, 2, 3, 4, 4, 6, 7, 8 };

void Gangster::write(FILE* file) const
{
  fprintf(file, "%d %6d %7d %7d %20llu %7.3f\n",
          nodeIndex, gangIndex, bandCount, gangMembers, gridCount, time);
}

void Gangster::writeAll(const std::vector<Gangster>& gang, const std::string& filename)
{
  FILE* out = fopen(filename.c_str(), "w");
  if (!out)
    throw std::runtime_error(strFormat("Can't write %s", filename.c_str()));

  for (const Gangster& g : gang)
    g.write(out);

  fclose(out);
}

void Gangster::duplicates(const std::vector<Gangster>& gang)
{
  uint32_t dups = 0;
  for (uint32_t gi = 1; gi < GangNum; ++gi)
    if (gang[gi].same(gang[gi - 1]))
      ++dups;
  printf("%u duplicates\n", dups);
}

void Gangster::compare(const std::vector<Gangster>& pGang, const std::vector<Gangster>& sGang)
{
  uint32_t gangsters = 0;
  uint32_t mismatches = 0;

  int pi = 0;
  for (const Gangster& sg : sGang)
    if (sg.gridCount > 0)
    {
      ++gangsters;

      while (pGang[pi] < sg)
        ++pi;

      if (pi >= GangNum || !sg.equivalent(pGang[pi]))
        ++mismatches;
    }

  printf("%u mismatches in %u gangsters\n", mismatches, gangsters);
}

// ****************************************
// *                                      *
// *  Find Gangsters and Get Census Data  *
// *                                      *
// ****************************************

// One of the 9 primary gangster configurations. 
struct GangSet
{
  int startIndex;       // gangster index of first gangster in the GangSet
  int count;            // number of gangsters in the GangSet
  int uniqueBoxCodes;   // how many ColCodes of the 5775 are used by gangsters?
  int gcdBandCounts;    // greatest common divisor of all band counts
};

// Result of the grid counting of each gangster, non-volatile so counting can be
// started and stopped
struct CountPacket
{
  uint32_t gangIndex;
  int thread;           // threadIndex in the Rope that did the count
  uint64_t count;       // total band 1 & 2 configuration count / 8
  double time;          // elapsed time in seconds divided by thread count

  // File I/O to make non-volatile:

  // Read from specified FILE. Old-style counts are too big for uint64_t and are first
  // divided by 64. New-style have an "*" prefix and are already divided by 64.
  // Return true if a CountPacket has been read, false for EOF. Throws
  // std::runtime_error for a bad gangIndex or an old-style count not divisible
  // by 64.
  bool read(FILE*); 

  // write to specified FILE
  void write(FILE*) const;  
};

bool CountPacket::read(FILE* file)
{
  char countBuf[32];
  if (fscanf(file, " %u %s %d %lf",
             &gangIndex, &countBuf, &thread, &time) == 4)
  {
    if (gangIndex >= GangNum)
      throw std::runtime_error(strFormat("Bad gangIndex %d", gangIndex));

    if (countBuf[0] == '*')
      count = std::strtoull(countBuf + 1, nullptr, 10);
    else
    {
      Bignum n(countBuf);
      Bignum rem;
      n.div(n, 64, rem);
      if (rem != 0)
        throw std::runtime_error(strFormat("Count for gang %u not divisible by 64", gangIndex));
      count = (uint64_t)n.makeint();
    }

    return true;
  }
  else
  {
    if (feof(file))
    {
      fclose(file);
      return false;
    }

    fclose(file);
    throw std::runtime_error("Bad count file format");
  }
}

void CountPacket::write(FILE* file) const
{
  fprintf(file, "%6d *%llu %3d %7.3f\n", gangIndex, count, thread, time);
}

// The central object for finding gangsters, computing gangMembers and bandCounts,
// holding the gangCache, holding gangster data, doing the final summation, and
// other housekeeping functions.
class BandGang
{
public:
  // Each of the 9 cache levels, one for each nodeIndex
  using CacheLevel = SymTable<int32_t, ColCode::Count, false>;

private:
  // See like-named public member
  ColCode renameTable_[ColCode::Count][ColCode::Count];

  // Consider a band with ColCodes 0, b1, b2, b3. Suppose that one of the boxes b1, b2, or b3,
  // are mapped to ColCode 0, yielding rename(b1, 0), 0, rename(b1, b2), rename(b1, b3). There
  // are only 9 possible values of rename(b1, 0), corresponding to the 9 ColCodes, because
  // P(0 <- b1) = P(0 <- b1.repCode) * P(b1.repCode <- b1), and P(b1.repCode <- b1) maps 0
  // to 0. So every band (0, b1, b2, b3) must have 1 - 6 members of the same equivalence
  // class in this cache, depending on how many of b1, b2, and b3 match. For example, if they
  // are all different then there are 3 ways to choose the box to be mapped to 0 and the
  // other 2 boxes can be swapped or not, total 6 cache entries.
  //
  // The cache holds different values over time.
  //    * initialized to -1 (no gangsters found)
  //    * temporary index issued in order of finding, or -1 if not yet found
  //    * code order index after all gangsters found
  //    * bandCounts for grid counting
  // The temporary find-order index is stored in gangMembers_ during finding. This list is
  // kept sorted (on gangCode) for fast lookup. The find-order index stays with its gangCode,
  // and is used to replace cache entires with the permanent code-order index.
  CacheLevel gangCache_[9];

  // The 9 GangSets
  GangSet gangSets_[9];

  // Used by parallel thread functions for mutually exclusive code sections
  std::mutex waitLock;

  // Gangster identity (code), member counts.
  IList gangMembers_;

  // Gangster band configuration counts
  uint32_t bandCounts_[GangNum];

  // Cache statistics for finding
  uint64_t codeCalls_, cacheMiss_;

  // Used during countBands and countMembers for progress reports
  Timer countTimer_;

  // A Carton holds the three non-code-zero boxes of a band, and related info, to
  // support efficient finding of the band's gangster (class representative)
  struct Carton
  {
    ColCode boxes[3];   // the three boxes
    int box0RepIndex;   // node index of class rep of boxes[0]
    int bestRepIndex;   // lowest node index of class reps of boxes
    CacheLevel* cache;  // the cache level for boxes[0]
  };

  // Look for the representative member of a gangster class containing boxes 0,b1,b2,b3 by
  // considering the 82,944 canonical permutations. Uses the Node objects so that far fewer
  // permutations need be considered. Update code if a better representative is found, i.e.
  // one with a lower numerical value. bestRepIndex is the best one expected, so that
  // higher ones can just be ignored.
  void gangCode_(ColCode boxes[3], int bestRepIndex, int& code);

  // Parallel thread iteration for the gangster finder. box01 = 5775 * box0RepCode + box1
  void findIteration_(int box01, int threadIndex);

  // Replace the temporary index issued in order of finding with the permanant code order
  // index (gangIndex).
  void fixCache_();

  // Find the number of bands that have the specified four ColCodes. This will only be used
  // for gangsters, so boxes[0] will be 0 and boxes[1] will be a node code, but those facts
  // are not used and so this in principal could count any band.
  static uint32_t bandCount_(ColCode boxes[4]);

  // Parallel thread iteration for the band counter.
  void bandIteration_(int gangIndex, int threadIndex);

  // For the parallel gangMembers counter, each thread has a private vector of GangNum counts
  // to avoid race conditions. Atomic counters are too slow. These are summed after all threads
  // have finished.
  std::vector<std::vector<uint32_t>> tempMembers_;

  // Parallel thread iteration for gangMembers. b1 is one of 5775 box1 ColCodes. Consider all
  // b1 <= b2 <= b3 combinations and add counts to tempMembers_. The b1.b2.b3 band is mapped
  // to its gangster by looking in the gangCache, which is full by now.
  void membersIteration_(int b1, int threadIndex);

public:
  // Initialize the renameTable_ and the gangCache_.
  BandGang();

  // Return number of gangsters found so far. This is poorly named.
  int size() const { return (int)gangMembers_.size(); }

  // Return the temporary find-order index of the gangster (class representative)
  // corresponding to a band with 4 equivalent Cartons of 3 boxes each. Call
  // gangCode_ as necessary. Update gangCache. Issue new temporary find-order
  // indices (under a mutex) when a new gangster is found.
  int gangIndex(Carton ct[4]);

  // Return the gangster code for the specified boxes
  static int makeGangCode(int box1NodeIndex, ColCode box2, ColCode box3)
  {
    sort(box2, box3);
    return (box1NodeIndex << 26) | (box2() << 13) | box3();
  }

  // Get the ColCodes of a gangster code
  static int box1NodeIndex(int gangCode) { return gangCode >> 26; }
  static ColCode box1(int gangCode) { return ColNode::nodeList[box1NodeIndex(gangCode)]; }
  static ColCode box2(int gangCode) { return ColCode((gangCode >> 13) & 0x1FFF); }
  static ColCode box3(int gangCode) { return ColCode(gangCode & 0x1FFF); }

  // Map code b1 by a permutation that renames the symbols such that a box with
  // code b0 would become the canonical box (code 0).
  // rename(b0, b1) is equivalent to mapping with QP::rename(b0, ColSets::canonSets())
  // but for codes instead of sets
  ColCode rename(ColCode b0, ColCode b1) const { return renameTable_[b0()][b1()]; }

  // Get read-only specified cache level (level is nodeIndex)
  const CacheLevel& cache(int level) const { return gangCache_[level]; }

  // Get IList of gangster codes and number of class members
  const IList& gangMembers() const { return gangMembers_; }

  // Get bandCount of gangster of specified index
  uint32_t bandCount(int gangIndex) const { return bandCounts_[gangIndex]; }

  // Get one of the 9 GangSet objects
  const GangSet& gangSet(int setIndex) const { return gangSets_[setIndex]; }

  // Write all gangster properties to specified file (usually bandGang.txt).
  // index, code, gangMembers, bandCount
  void writeFile(const std::string& filename) const;

  // Read all gangster properties from specified file (usually bandGang.txt). Verify that
  // indices are all present and in order, and codes match computed values. If
  // !verifyMembers, store the read gangMembers, otherwise verify that computed values
  // match those read. If !verifyCounts store the read bandCounts, otherwise verify that
  // computed values match those read. Throw std::runtine_error if any verification fails.
  // Since the gangCache is very large and just a few seconds to compute, it is not
  // stored. The gangster codes come from that, and so are always verified, kind of
  // a startup self-diagnostic. Computing gangMembers is like 30 seconds and bandCounts
  // like 10 minutes (4 cores 8 threads 2 GHz).
  void readFile(const std::string& filename, bool verifyMembers, bool verifyCounts);

  // Find all gangsters with specified number of threads, fill cache with gangIndex values.
  // Throws std::runtime_error if number found does not match GangNum. Several seconds.
  void find(int threads);

  // Compute all bandCounts with specified number of threads. 10 minutes
  void countBands(int threads);

  // Compute all gangMembers with specified number of threads. 30 seconds
  void countMembers(int threads);

  // Replace all gangIndex values in the cache with the corresponding bandCount, divided
  // by 8 (the GCD of all bandCounts) so the grid counts will fit in 64 bits.
  void replaceCacheCodesWithBandCounts();

  // Find the 9 GangSets. Expects all gangsters found and sorted in gangMembers_.
  void computeGangSets();

  // Find and read all grid count files (gridCount*) in the current directory into
  // the specified vector. Print count progress report. Throw std::runtime_error
  // if a duplicate gangIndex is found with a mismatching count. If all gangsters
  // have been counted, compute and print final result.
  void countAll(std::vector<Gangster>& gang, const std::string& baseFilename) const;

  // Cache performance stats
  std::string printStats() const;
  
  // What fraction of cache is filled?
  double cacheFill() const;

  // What fraction of specified cache level is filled?
  double cacheFill(int level) const;

  // Perform some consistency checks, print report to stdout if any errors found.
  void verifyTables() const;
};

// *****************************
// *                           *
// *  BandGang Infrastructure  *
// *                           *
// *****************************

BandGang::BandGang()
  : gangMembers_(GangNum)
{
  ProfileTree::start("BandGang Construct");

  gangMembers_.keepSorted(true);

  // Rename table, fast version using nodes. First compute just the node 0,
  // which is just the identity permutation
  for (int j = 0; j < ColCode::Count; ++j)
    renameTable_[0][j] = ColCode(j);

  // Now the rest of the nodes
  QP qp;
  for (size_t i = 1; i < ColNode::nodeList.size(); ++i)
  {
    ColCode nodeCode = ColNode::nodeList[i];
    qp.rename(ColSets(nodeCode), ColSets::canonSets());
    ColCode mapped0 = qp.mapc(ColSets(ColCode(0)));
    const ColNode::PermutationSet& p0 = ColNode::nodes(mapped0).permutations[0];
    mapped0 = p0.permute(mapped0);
    //printf("%4d -> %4d\n", nodeCode(), mapped0());
    for (int j = 0; j < ColCode::Count; ++j)
      renameTable_[nodeCode()][j] = p0.permute(qp.mapc(ColSets(ColCode(j))));
  }

  // Compute the rest of renameTable_ by mapping the representative members
  // P[0 <- i](j) = P[0 <- rep] * P[rep <- i](j). Note that there are many
  // permutations that could map a code to 0, but this particular choice
  // results in best cache hit rate and is necessary for double rename
  for (ColCode c0(0); c0.isValid(); ++c0)
  {
    const ColNode& node = ColNode::nodes(c0);
    if (node.repCode != c0)
      for (ColCode c1(0); c1.isValid(); ++c1)
        renameTable_[c0()][c1()] = rename(node.repCode, node.permutations[0].permute(c1));
  }

  // Clear the cache
  for (int i = 0; i < 9; ++i)
    gangCache_[i].setAll(-1);

  ProfileTree::stop();
}

void BandGang::verifyTables() const
{
  ProfileTree::start("Verify BandGang Tables");

  // Verify rename table
  bool cols[ColCode::Count];
  int errors = 0;

  for (ColCode c0(0); c0.isValid(); ++c0)
  {
    for (bool& c : cols)
      c = false;

    if (rename(c0, c0) != ColCode(0))
      ++errors;

    if (rename(c0, ColCode(0)) != ColNode::nodes(c0).repCode)
      ++errors;

    for (ColCode c1(0); c1.isValid(); ++c1)
    {
      ColCode code = rename(c0, c1);
      if (cols[code()])
        ++errors;
      else
        cols[code()] = true;
    }
  }
  if (errors > 0)
    printf("%d Band Rename Table errors\n", errors);

  ProfileTree::stop();
}

double BandGang::cacheFill(int level) const
{
  if ((uint32_t)level >= 9)
    return 0;

  uint32_t n = 0;
  const CacheLevel& cache = gangCache_[level];
  for (int c0 = 0; c0 < ColCode::Count; ++c0)
    for (int c1 = 0; c1 <= c0; ++c1)
      n += (int)(cache.get(c0, c1) >= 0);
  return (double)n / (ColCode::Count * (ColCode::Count + 1) / 2);
}

double BandGang::cacheFill() const
{
  double fill = 0;
  for (int i = 0; i < 9; ++i)
    fill += cacheFill(i);
  return fill / 9;
}

void BandGang::fixCache_()
{
  ProfileTree::start("Fix gang cache");
  std::vector<uint32_t> liveIndices(size());
  for (int i = 0; i < size(); ++i)
    liveIndices[gangMembers().count(i)] = i;

  for (int level = 0; level < 9; ++level)
  {
    CacheLevel& cache = gangCache_[level];
    for (int c0 = 0; c0 < ColCode::Count; ++c0)
      for (int c1 = c0; c1 < ColCode::Count; ++c1)
        cache.set(c0, c1, liveIndices[cache.get(c0, c1)]);
  }

  ProfileTree::stop(9 * ColCode::Count * ColCode::Count);
}

void BandGang::replaceCacheCodesWithBandCounts()
{
  ProfileTree::start("Replace cache codes");

  for (int level = 0; level < 9; ++level)
  {
    CacheLevel& cache = gangCache_[level];
    for (int c0 = 0; c0 < ColCode::Count; ++c0)
      for (int c1 = c0; c1 < ColCode::Count; ++c1)
      {
        uint32_t gangIndex = (uint32_t)cache.get(c0, c1);   // unsigned compare
        if (gangIndex >= GangNum)
          throw std::runtime_error("Bad gangster index in cache");

        uint32_t n = bandCount(gangIndex);
        if ((n & 7) != 0)
          throw std::runtime_error("Band count not divisible by 8");

        cache.set(c0, c1, (int32_t)(n >> 3));
      }
  }

  ProfileTree::stop(9 * ColCode::Count * ColCode::Count);
}

void BandGang::computeGangSets()
{
  int setIndex = -1;
  for (int gi = 0; gi < size(); ++gi)
  {
    int si = box1NodeIndex(gangMembers()[gi]);
    if (setIndex != si)
    {
      if (setIndex >= 0)
        gangSets_[setIndex].count = gi - gangSets_[setIndex].startIndex;

      setIndex = si;
      gangSets_[si].startIndex = gi;
    }
  }
  gangSets_[setIndex].count = size() - gangSets_[setIndex].startIndex;

  for (int si = 0; si < 9; ++si)
  {
    IList codeSet;
    codeSet.keepSorted(true);

    int64_t gcd = bandCount(gangSets_[si].startIndex);
    for (int gi = 0; gi < gangSets_[si].count; ++gi)
    {
      int gangIndex = gangSets_[si].startIndex + gi;
      int code = gangMembers()[gangIndex];
      codeSet.enter(box1(code)());
      codeSet.enter(box2(code)());
      codeSet.enter(box3(code)());

      gcd = Math::gcd(gcd, bandCount(gangIndex));
    }

    gangSets_[si].uniqueBoxCodes = (int)codeSet.size();
    gangSets_[si].gcdBandCounts = (int)gcd;
  }
}

std::string BandGang::printStats() const
{
  return commas(cacheMiss_) + " cache misses " + commas(codeCalls_) + " code calls";
}

// ***************************************
// *                                     *
// *  Band Gangster Multi-Thread Finder  *
// *                                     *
// ***************************************

void BandGang::gangCode_(ColCode boxes[3], int bestRepIndex, int& code)
{
  // Examine all 3 boxes
  for (int b = 0; b < 3; ++b)
  {
    // If the repIndex is not best, this can't be box1 of a gangster so ignore
    const ColNode& node = ColNode::nodes(boxes[b]);
    if (node.repIndex == bestRepIndex)
    {
      // Get the other two boxes
      ColCode b1 = boxes[(b + 1) % 3];
      ColCode b2 = boxes[(b + 2) % 3];

      // Get the repCodes and the best gangCode we can possibly get, to know if
      // we can't do any better and so should stop looking
      ColCode b1Rep = ColNode::nodes(b1).repCode;
      ColCode b2Rep = ColNode::nodes(b2).repCode;
      int bestPossibleCode = makeGangCode(bestRepIndex, b1Rep, b2Rep);

      if (bestRepIndex > 0)
        // If the bestRepIndex (b0) is not 0, the permutation list is valid. Map b1 and
        // b2 with every one and see if we get better (lower) codes. Note tht these
        // map 0 to 0 and b0 to b0.
        for (const auto& p : node.permutations)
        {
          if (code <= bestPossibleCode)
            break;

          ColCode b1Try = p.permute(b1);
          ColCode b2Try = p.permute(b2);
          int tryCode = makeGangCode(bestRepIndex, b1Try, b2Try);
          minEq(code, tryCode);
        }
      else if (b1Rep() == 0 || b2Rep() == 0)
        // There is no permutation list for node 0, all canonical permutations map
        // 0 to 0. If either of the other boxes are 0, get get the bestPossibleCode
        minEq(code, bestPossibleCode);
      else
      {
        // Here there is also no permutation list for b0. Use b1's list on b2 and b2's
        // list on b1 to try to find better code
        if (b1Rep <= b2Rep)
          for (const auto& p : ColNode::nodes(b1).permutations)
          {
            if (code <= bestPossibleCode)
              break;

            ColCode b2Try = p.permute(b2);
            int tryCode = makeGangCode(bestRepIndex, b1Rep, b2Try);
            minEq(code, tryCode);
          }

        if (b2Rep <= b1Rep)
          for (const auto& p : ColNode::nodes(b2).permutations)
          {
            if (code <= bestPossibleCode)
              break;

            ColCode b1Try = p.permute(b1);
            int tryCode = makeGangCode(bestRepIndex, b1Try, b2Rep);
            minEq(code, tryCode);
          }
      }
    }
  }
}

int BandGang::gangIndex(Carton ct[4])
{
  // Find best rep index (lowest) over all 4 Cartons
  int bestRepAll = 9;
  for (int i = 0; i < 4; ++i)
  {
    int bestRep = std::min(ct[i].box0RepIndex, ColNode::nodes(ct[i].boxes[1]).repIndex);
    minEq(bestRep, ColNode::nodes(ct[i].boxes[2]).repIndex);
    ct[i].bestRepIndex = bestRep;
    minEq(bestRepAll, bestRep);
  }

  // Call gangCode_ as needed to find gangCode
  int bestCode = std::numeric_limits<int>::max();
  int codeCalls = 0;
  for (int i = 0; i < 4; ++i)
    if (ct[i].bestRepIndex == bestRepAll)
    {
      gangCode_(ct[i].boxes, bestRepAll, bestCode);
      ++codeCalls;
    }

  int g;
  {
    // Under mutex, see if code already found. If not issue new found-order index and
    // add it. If so get its found-order index.
    std::lock_guard<std::mutex> lk(waitLock);
    size_t index;
    if (gangMembers_.lookup(bestCode, index))
      g = (int)gangMembers_.count(index);
    else
    {
      g = (int)gangMembers_.size();
      gangMembers_.append(bestCode, g);
    }

    ++codeCalls_ += codeCalls;
    ++cacheMiss_;
  }

  // Update cache
  for (int b = 0; b < 4; ++b)
    ct[b].cache->set(ct[b].boxes[1](), ct[b].boxes[2](), g);

  return g;
}

void BandGang::findIteration_(int box01, int threadIndex)
{
  // Load b0 and b1 for Carton 0
  Carton ct[4];
  ct[0].box0RepIndex = box01 / ColCode::Count;
  ct[0].boxes[0] = ColNode::nodeList[ct[0].box0RepIndex];
  ct[0].boxes[1] = ColCode(box01 % ColCode::Count);
  ct[0].cache = &gangCache_[ct[0].box0RepIndex];

  // Consider all b2 >= b1 for Carton 0
  for (ct[0].boxes[2] = ct[0].boxes[1]; ct[0].boxes[2].isValid(); ++ct[0].boxes[2])
  {
    // If Carton 0 hits the cache we're done with this b2
    if (ct[0].cache->get(ct[0].boxes[1](), ct[0].boxes[2]()) >= 0)
      continue;

    // If not, try the other cartons by mapping the other 3 boxes to 0
    // Quit as soon as the cache is hit
    for (int b = 1; b < 4; ++b)
    {
      ColCode base = ct[0].boxes[b - 1];
      ct[b].boxes[0] = rename(base, ColCode(0));
      ct[b].boxes[1] = rename(base, ct[0].boxes[b % 3]);
      ct[b].boxes[2] = rename(base, ct[0].boxes[(b + 1) % 3]);
      ct[b].box0RepIndex = ColNode::nodes(ct[b].boxes[0]).repIndex;
      ct[b].cache = &gangCache_[ct[b].box0RepIndex];

      int possibleGangIndex = ct[b].cache->get(ct[b].boxes[1](), ct[b].boxes[2]());
      if (possibleGangIndex >= 0)
      {
        for (int bb = 0; bb < b; ++bb)
          ct[bb].cache->set(ct[bb].boxes[1](), ct[bb].boxes[2](), possibleGangIndex);
        goto cacheHit;
      }
    }

    // No cache hit, do the hard work and update the cache
    gangIndex(ct);
  cacheHit:;
  }

  fprintf(stderr, "\r%d.%4d [%6d]; %s",
          ct[0].box0RepIndex, ct[0].boxes[1](), size(), printStats().c_str());
}

void BandGang::find(int threads)
{
  ProfileTree::start("Band Gangsters");
  codeCalls_ = 0;
  cacheMiss_ = 0;
  Rope<BandGang> rope(this, &BandGang::findIteration_);
  rope.run(9 * ColCode::Count, threads);
  ProfileTree::stop();

  fprintf(stderr, "\n");
  printf("%s\n", printStats().c_str());

  if (size() != GangNum)
    throw std::runtime_error(strFormat("Unexpected gangster count %d", size()));
  
  fixCache_();
}

// ******************
// *                *
// *  Band Counter  *
// *                *
// ******************
//
// 10 nested loops to explore the 4!^10 permutations of columns 1-10. Column 0
// is held fixed and column 11 has only one possible compatible arrangement.
// Nesting terminates as soon as an incompatibility is found, otherwise this
// would take way too long. The loops are nested to jump from box to box so
// that incompatibilities are found in outer loops--columns within a box
// are always compatible.
uint32_t BandGang::bandCount_(ColCode boxes[4])
{
  uint32_t band = 0;
  RowSets s0 = ColTables::flipSets(boxes[0], 0, 0);
  for (int p1 = 0; p1 < 24; ++p1)
  {
    RowSets s1 = ColTables::flipSets(boxes[0], 1, p1) | s0;
    for (int p3 = 0; p3 < 24; ++p3)
    {
      RowSets s3 = ColTables::flipSets(boxes[1], 0, p3);
      if ((s3 & s1).isCompatible())
      {
        s3 |= s1;
        for (int p6 = 0; p6 < 24; ++p6)
        {
          RowSets s6 = ColTables::flipSets(boxes[2], 0, p6);
          if ((s6 & s3).isCompatible())
          {
            s6 |= s3;
            for (int p9 = 0; p9 < 24; ++p9)
            {
              RowSets s9 = ColTables::flipSets(boxes[3], 0, p9);
              if ((s9 & s6).isCompatible())
              {
                s9 |= s6;
                for (int p2 = 0; p2 < 24; ++p2)
                {
                  RowSets s2 = ColTables::flipSets(boxes[0], 2, p2);
                  if ((s2 & s9).isCompatible())
                  {
                    s2 |= s9;
                    for (int p4 = 0; p4 < 24; ++p4)
                    {
                      RowSets s4 = ColTables::flipSets(boxes[1], 1, p4);
                      if ((s4 & s2).isCompatible())
                      {
                        s4 |= s2;
                        for (int p7 = 0; p7 < 24; ++p7)
                        {
                          RowSets s7 = ColTables::flipSets(boxes[2], 1, p7);
                          if ((s7 & s4).isCompatible())
                          {
                            s7 |= s4;
                            for (int p10 = 0; p10 < 24; ++p10)
                            {
                              RowSets s10 = ColTables::flipSets(boxes[3], 1, p10);
                              if ((s10 & s7).isCompatible())
                              {
                                s10 |= s7;
                                for (int p5 = 0; p5 < 24; ++p5)
                                {
                                  RowSets s5 = ColTables::flipSets(boxes[1], 2, p5);
                                  if ((s5 & s10).isCompatible())
                                  {
                                    s5 |= s10;
                                    for (int p8 = 0; p8 < 24; ++p8)
                                    {
                                      RowSets s8 = ColTables::flipSets(boxes[2], 2, p8);
                                      if ((s8 & s5).isCompatible())
                                        ++band;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return band;
}

void BandGang::bandIteration_(int gangIndex, int threadIndex)
{
  ColCode boxes[4];
  int gangCode = gangMembers()[gangIndex];
  boxes[0] = ColCode(0);
  boxes[1] = BandGang::box1(gangCode);
  boxes[2] = BandGang::box2(gangCode);
  boxes[3] = BandGang::box3(gangCode);
  bandCounts_[gangIndex] = bandCount_(boxes);

  double elapsedSec = countTimer_.elapsedSeconds();
  double estimatedSec = elapsedSec * size() / (gangIndex + 1);
  fprintf(stderr, "\r%6d  %6.1f/%6.1f", gangIndex, estimatedSec - elapsedSec, estimatedSec);
}

void BandGang::countBands(int threads)
{
  ProfileTree::start("Band Counter");
  countTimer_.start();
  Rope<BandGang> rope(this, &BandGang::bandIteration_);
  rope.run(size(), threads);
  ProfileTree::stop(size());
  fprintf(stderr, "\n");
}

// **************************
// *                        *
// *  Gang Members Counter  *
// *                        *
// **************************

void BandGang::membersIteration_(int b1, int threadIndex)
{
  ColCode box1(b1);
  ColCode box0 = rename(box1, ColCode(0));
  const CacheLevel& cache = gangCache_[ColNode::nodes(box0).repIndex];

  int gangIndex = cache.get(0, 0);
  ++tempMembers_[threadIndex][gangIndex];

  for (int b2 = b1 + 1; b2 < ColCode::Count; ++b2)
  {
    ColCode box2 = rename(box1, ColCode(b2));
    gangIndex = cache.get(0, box2());
    tempMembers_[threadIndex][gangIndex] += 3;
    gangIndex = cache.get(box2(), box2());
    tempMembers_[threadIndex][gangIndex] += 3;
  }

  for (int b2 = b1 + 1; b2 < ColCode::Count; ++b2)
  {
    ColCode box2 = rename(box1, ColCode(b2));
    for (int b3 = b2 + 1; b3 < ColCode::Count; ++b3)
    {
      ColCode box3 = rename(box1, ColCode(b3));
      gangIndex = cache.get(box2(), box3());
      tempMembers_[threadIndex][gangIndex] += 6;
    }
  }

  double elapsedSec = countTimer_.elapsedSeconds();
  double estimatedSec = elapsedSec * ColCode::Count / (b1 + 1);
  fprintf(stderr, "\r%4d  %6.1f/%6.1f", b1, estimatedSec - elapsedSec, estimatedSec);
}

void BandGang::countMembers(int threads)
{
  ProfileTree::start("Gang Counter");

  // Init temp vectors for each thread
  tempMembers_.resize(threads);
  for (std::vector<uint32_t>& vec : tempMembers_)
    vec.resize(size(), 0);

  // Count members
  countTimer_.start();
  Rope<BandGang> rope(this, &BandGang::membersIteration_);
  rope.run(ColCode::Count, threads);
  fprintf(stderr, "\n");

  // Sum and clear the temps
  for (int g = 0; g < size(); ++g)
  {
    uint32_t n = tempMembers_[0][g];
    for (size_t t = 1; t < tempMembers_.size(); ++t)
      n += tempMembers_[t][g];
    gangMembers_.count(g, n);
  }
  tempMembers_.clear();

  ProfileTree::stop();
}

// *************************************
// *                                   *
// *  Gangster Persistance (File I/O)  *
// *                                   *
// *************************************

void BandGang::writeFile(const std::string& filename) const
{
  ProfileTree::start("Write gangsters");

  std::FILE* file = fopen(filename.c_str(), "w");

  for (int g = 0; g < size(); ++g)
  {
    int code = gangMembers()[g];
    fprintf(file, "%6d, %d.%4d.%4d, %9u, %9u\n",
            g, BandGang::box1NodeIndex(code), BandGang::box2(code)(), BandGang::box3(code)(),
            gangMembers().count(g), bandCount(g));
  }
  fclose(file);

  ProfileTree::stop(GangNum);
}

void BandGang::readFile(const std::string & filename, bool verifyMembers, bool verifyCounts)
{
  ProfileTree::start("Read/verify gangsters");
  std::FILE* file = fopen(filename.c_str(), "r");

  for (uint32_t i = 0; i < GangNum; ++i)
  {
    uint32_t index, box1, box2, box3, members, count;
    if (fscanf(file, " %u, %u. %u. %u, %u, %u",
               &index, &box1, &box2, &box3, &members, &count) == 6)
    {
      if (i != index)
        throw std::runtime_error(strFormat("Read gangster index %d, expected %d\n", index, i));

      int code = makeGangCode(box1, ColCode(box2), ColCode(box3));
      if (code != gangMembers()[i])
        throw std::runtime_error(strFormat("Read gangster %d code %d expected %d",
                                            i, code, gangMembers()[i]));

      if (verifyMembers)
      {
        if (members != gangMembers().count(i))
          throw std::runtime_error(strFormat("Read gangster %d members %d expected %d",
                                             i, members, gangMembers().count(i)));
      }
      else
        gangMembers_.count(i, members);

      if (verifyCounts)
      {
        if (count != bandCount(i))
          throw std::runtime_error(strFormat("Read gangster %d count %d expected %d",
                                             i, count, bandCount(i)));
      }
      else
        bandCounts_[i] = count;
    }
    else
      throw std::runtime_error(strFormat("Gangster read error at index %d\n", i));
  }

  fclose(file);

  ProfileTree::stop(GangNum);
}

// ******************************************************
// *                                                    *
// *  Read All Grid Count Records, Compute Final Count  *
// *                                                    *
// ******************************************************

void BandGang::countAll(std::vector<Gangster>& gang, const std::string& baseFilename) const
{
  ProfileTree::start("Count all");

  uint64_t maxCount = 0;
  uint32_t maxIndex = 0;
  std::string maxFile;

  int files = 0;
  int gangsters = 0;
  double totalTime = 0;

  gang.resize(GangNum);

  uint32_t progress[9];
  for (uint32_t& p : progress)
    p = 0;

  ProfileTree::start("Get gangster properties");
  for (uint32_t gi = 0; gi < GangNum; ++gi)
  {
    Gangster& g = gang[gi];
    g.gangIndex = gi;
    g.bandCount = bandCount(gi);
    g.gangMembers = gangMembers().count(gi);
    g.gridCount = 0;
    g.nodeIndex = box1NodeIndex(gangMembers()[gi]);
  }
  ProfileTree::stop(GangNum);

  for (auto const& file : std::filesystem::directory_iterator{ "." })
  {
    std::string filename = file.path().filename().u8string();
    if (file.path().extension() == ".txt" && filename.substr(0, baseFilename.size()) == baseFilename)
    {
      ProfileTree::start("Read count file");

      std::FILE* file = fopen(filename.c_str(), "r");
      if (!file)
        throw std::runtime_error(strFormat("Can't open %s\n", filename.c_str()));

      ++files;

      while (true)
      {
        CountPacket cp;
        if (cp.read(file))
        {
          if (gang[cp.gangIndex].gridCount == 0)
          {
            gang[cp.gangIndex].gridCount = cp.count;
            gang[cp.gangIndex].time      = cp.time ;

            int baseIndex = box1NodeIndex(gangMembers()[cp.gangIndex]);
            ++progress[baseIndex];
            ++gangsters;

            if (cp.count > maxCount)
            {
              maxIndex = cp.gangIndex;
              maxFile = filename;
              maxCount = cp.count;
            }

            if (cp.time >= 100)
              printf("Anomolous time %.2fs at gangster %u\n", cp.time, cp.gangIndex);
            totalTime += cp.time;
          }
          else
          {
            if (gang[cp.gangIndex].gridCount != cp.count)
              throw std::runtime_error(strFormat("Count mismatch gangster %u", cp.gangIndex));
            minEq(gang[cp.gangIndex].time, cp.time);
          }
        }
        else
          break;
      }

      ProfileTree::stop();
    }
  }

  int bits = 0;
  for (uint64_t count = maxCount; count > 0; count >>= 1, ++bits);

  printf("Max count %llu needs %d bits, gangster %u in file %s\n",
         maxCount, bits, maxIndex, maxFile.c_str());

  printf("%d files, %d gangsters examined in %.2f hours\n",
         files, gangsters, totalTime / 3600);

  ProfileTree::start("Progress report");
  printf("\nProgress:\n");
  for (int i = 0; i < 9; ++i)
  {
    printf("  %d  %6u/%6u  %5.1f%%",
           i, progress[i], gangSet(i).count, 100.0 * progress[i] / gangSet(i).count);
    if (progress[i] < (uint32_t)gangSet(i).count)
    {
      const char* prefix = "  needs ";
      bool mode = true;
      for (uint32_t gangOffset = 0; gangOffset < (uint32_t)gangSet(i).count; ++gangOffset)
      {
        bool newMode = gang[gangSet(i).startIndex + gangOffset].gridCount != 0;
        if (newMode != mode)
        {
          if (newMode)
            printf("%u", gangOffset - 1);
          else
          {
            printf("%s %u - ", prefix, gangOffset);
            prefix = ", ";
          }
          mode = newMode;
        }
      }

      if (!mode)
        printf("%u", gangSet(i).count - 1);
    }

    printf("\n");
  }
  printf("     %6u/%6u  %5.1f%%\n",
         gangsters, GangNum, 100.0 * gangsters / GangNum);

  ProfileTree::stop();

  if (gangsters == GangNum)
  {
    ProfileTree::start("Total grid configurations");
    Bignum total = 0;
    for (const Gangster& g : gang)
      total += (Bignum)g.gangMembers * g.gridCount * g.bandCount;

    total *= (Bignum)ipower<uint64_t>(8, 2)                   // band 1 & 2 bandCounts were divided by 8
             * ipower<uint64_t>(factorial<uint64_t>(4), 3)    // column 0 of each band was held fixed
             * ColCode::Count                                 // box0 could be any other ColCode
             * ipower<uint64_t>(factorial<uint64_t>(3), 4);   // 4 stacks have 3! column permutations

    printf("%s ~= %.6e\n", total.decimal(true).c_str(), total.makeDouble());

    ProfileTree::stop();
  }

  ProfileTree::stop();
}

// *************************
// *                       *
// *  Count Complete Grid  *
// *                       *
// *************************

// For each gangster in band0 we want to consider the 346^4 compatible box combinations in
// band1. For each of those, there is only one compatible combination in band2. We want to
// sum the product of the bandCounts of band1 band2 for all those combinations. Those band
// counts are found in gangCache, so we need to convert band1 and band2 to their gangster
// (class representative).
//
// Every gangster in a GangSet has the same box0 and box1 (band0). This means that there are
// 346*346 possible compatible box4 and box5 combinations in band1, and the same
// combinations for box8 and box9 in band2. Those fixed 119,716 possibilities are encoded
// in tables by setup_() and used for every gangster of the GangSet.
//
// If box4 and box8 are swapped the result is the same. If box4 < box8 we count once and
// set the multiplier_ table entry to 2. If box4 = box8 we could once and set multiplier_
// to 1. If box8 < box4 we set multiplier_ to 0 and skip that case. This reduces the work
// by roughly half.
//
// The main counting loops consider 173 * 347 * 346 * 346 ColCode configurations of
// band1 (boxes 4-7) and band2 (boxes 8-11). Each has to be mapped to its gangster. The
// mapping goes like this:
//    b4 -> b4' = rename(b5, b4) -> b4" = 0
//    b5 -> b5' = 0              -> b5" = rename(b4', 0)
//    b6 -> b6' = rename(b5, b6) -> b6" = rename(b4', b6')
//    b7 -> b7' = rename(b5, b7) -> b7" = rename(b4', b7')
// Note that b5" = rename(b4', 0) must be one of the 9 codes because
// P(0 <- b4') = P(0 <- b4'.repCode) * P(b4'.repCode <- b4'), and P(b4'.repCode <- b4')
// maps 0 to 0. We arrange that those 9 codes are the 9 node codes. So we can map
// an arbirtary band to its gangster with the composition of two renames each for b6/b10
// and b7/b11. That composition is stored in the huge double rename table. Only the b7/b11
// double renames are needed in the inner loop, the b6/b10 double renames are in the
// next-outer loop.
//
// cacheTable_, multiplier_, and doubleRenameTable_ are set for box4/box5 of band1.
// box8/box9 of band2 must also appear in those tables, but at a different index.
// The otherBandIndex_ table gives that index.

class GridCounter
{
public:
  // Initialize internal gangser-independent table
  GridCounter(const BandGang& gang, bool gpuEnable);

  // Count all gangsters in specified one of the 9 GangSets, using specified number of
  // threads, appending results as found to file countFileBase_gangSetMark.txt, where
  // gangSetMark is gangIndex followed by '-' if countBackwards. countBackwards means
  // reverse gangIndex order. If countStartIndex > 0 start count, forward or back, there.
  // countBackwards and countStartOffset come from the command line, allowing the
  // operator to more easily employ multiple computers working in parallel.
  //
  // If verify, enable verification mode. In this mode gangsters thathave already been
  // counted are recounted and verified. Otherwise gangsters already counted are skipped,
  // allowing easy stop/restart of counting.
  //
  // threads is the number of CPU threads, which can be 0 to mean don't do counting with
  // CPU threads. If 0 and GPU is not enabled, counting is skipped--just setup is run.
  //
  // Throws std::runtime_error if any of a varient of errors (i.e. bugs) are detected.
  void count(int gangSet, int threads, const std::string& countFileBase,
             bool verify, bool countBackwards, int countStartOffset,
             bool box2GroupMode);

private:
  static constexpr uint32_t DoubleBoxCount = ipower<uint32_t>(ColCompatible::Count, 2);

  std::string countFile_;   // full path
  const BandGang& gang_;
  int gangSetIndex_;        // which set we're working on
  const GangSet* gangSet_;

  int threadCount_;         // number of threads at work, to determine per-thread times
  std::mutex waitLock;      // lock for appending the results file, printing progress

  std::vector<uint64_t> counts_;  // Counts from previous or current run, for skipping and
                                  // verification
  Bignum totalCount_;
  double totalTime_;
  bool verifyMode_;
  bool countBackwards_;
  int countSkip_;
  bool gpuEnable_;
  uint32_t activeBox01Count_;

  // Big tables. See comments in sudokuda.h
  ColCode codeCompatTable_[ColCode::Count][ColCompatible::Count][2];
  GridCountPacket gcPackets[DoubleBoxCount];

  void setup_(int gangSetIndex);
  bool packetOrder_(int gcpIndex0, int gcpIndex1) const;
  void readCountFile_();
  void countIteration_(int gangCode, int threadIndex);
  void printMultiplierHist_() const;

  // *** New box2 group counting data, for good data cache hit rate. ***

  // A box2Group is a sequence of consecutive gangsters in a GangSet that have the
  // same box2 codes. Being in the same GangSet, they also have the same box0
  // and box1 codes.

  int box2GroupStartOffset_;  // Offset in current GangSet of first gangster in the group
  int box2GroupSize_;         // Number of gangsters in the group

  // Return the gangster code of the gangster at the specified index in the group
  int32_t box2GroupGangCode_(int groupIndex) const
  {
    return gang_.gangMembers()[gangSet_->startIndex + box2GroupStartOffset_ + groupIndex];
  }

  // Return the box2 ColCode of the group
  ColCode box2GroupCode_() const { return BandGang::box2(box2GroupGangCode_(0)); }

  // A vector of the box3 codes of the group
  std::vector<ColCode> box2GroupBox3_;

  // A place to accumulate partial counts, effectively box2GroupCounts_[box2GroupSize_][threadCount_]
  // Each CPU thread needs a set of counts to avoid race conditions
  std::vector<uint64_t> box2GroupCounts_;

  // Add the specified count to box2GroupCounts_ for the specified groupIndex and threadIndex
  void box2GroupCountsAdd_(int groupIndex, int threadIndex, uint64_t count)
  {
    box2GroupCounts_[groupIndex * threadCount_ + threadIndex] += count;
  }

  // Get the total count for the specified groupIndex by summing the thread partial counts
  uint64_t box2GroupCount_(int groupIndex)
  {
    uint64_t count = 0;
    uint64_t* p = box2GroupCounts_.data() + groupIndex * threadCount_;
    for (int thread = 0; thread < threadCount_; ++thread)
      count += p[thread];
    return count;
  }

  // The iteration function for the Rope threads.
  void box2GroupIteration_(int box01, int threadIndex);

  // The wrangler handles the Rope
  void box2GroupWrangler_();
};

GridCounter::GridCounter(const BandGang& gang, bool gpuEnable)
  : gang_(gang), gpuEnable_(gpuEnable)
{
  ProfileTree::start("Construct GridCounter");

  // codeCompatible
  std::vector<ColSets> list;
  for (ColCode c(0); c.isValid(); ++c)
  {
    ColCompatible::makeSets(c, list);
    for (int i = 0; i < ColCompatible::Count; ++i)
    {
      codeCompatTable_[c()][i][0] = list[i].makeCanonical();
      codeCompatTable_[c()][i][1] = (ColSets(c) | list[i]).invert().makeCanonical();
    }
  }

  ProfileTree::stop(ColCode::Count * ColCompatible::Count);

#ifdef JETSON
  if (gpuEnable)
  {
    ProfileTree::start("Band counts, compat table -> GPU");
    gpuInit((int32_t (*)[ColCode::Count][ColCode::Count])&gang.cache(0),
            (uint16_t(*)[ColCompatibleCount][2])codeCompatTable_);
    ProfileTree::stop(9 * ColCode::Count * ColCode::Count);
  }
#endif
}

void GridCounter::readCountFile_()
{
  // Read countFile if any
  totalTime_ = 0;
  std::FILE* file = fopen(countFile_.c_str(), "r");
  if (file)
  {
    while (true)
    {
      CountPacket cp;
      if (cp.read(file))
      {
        int gangOffset = cp.gangIndex - gangSet_->startIndex;
        if (gangOffset < 0 || gangOffset >= gangSet_->count)
          throw std::runtime_error(strFormat("Bad gangIndex %d in file %s",
                                             cp.gangIndex, countFile_.c_str()));

        if (counts_[gangOffset] == 0)
          counts_[gangOffset] = cp.count;
        else if (counts_[gangOffset] != cp.count)
          throw std::runtime_error(strFormat("Mismatch of duplicate gangster count, index %d, file %s",
                                             cp.gangIndex, countFile_.c_str()));

        totalCount_ += cp.count;
        totalTime_ += cp.time;
      }
      else
        break;
    }
  }

  printf("Read count file %s, total time so far %.2f hours\n",
         countFile_.c_str(), totalTime_ / 3600.0);
}

bool GridCounter::packetOrder_(int gcpIndex0, int gcpIndex1) const
{
  const GridCountPacket& gcp0 = gcPackets[gcpIndex0];
  const GridCountPacket& gcp1 = gcPackets[gcpIndex1];

  if (gcp0.multiplier == 0)
    return false;
  if (gcp1.multiplier == 0)
    return true;

  int level00 = gcp0.cacheLevel;
  int level01 = gcPackets[gcp0.otherIndex].cacheLevel;
  sort(level00, level01);

  int level10 = gcp1.cacheLevel;
  int level11 = gcPackets[gcp1.otherIndex].cacheLevel;
  sort(level10, level11);

  if (level00 < level10)
    return true;
  if (level00 == level10)
  {
    if ((level00 & 1) == 0)
      return level01 < level11;
    else
      return level01 > level11;
  }
  return false;
}

void GridCounter::setup_(int gangSetIndex)
{
  ProfileTree::start("GridCounter Setup");

  gangSetIndex_ = gangSetIndex;
  gangSet_ = &gang_.gangSet(gangSetIndex_);

  counts_.clear();
  counts_.resize(gangSet_->count, 0);

  readCountFile_();

  // Make tables
  ProfileTree::start("Big tables");
  ColCode box1 = ColNode::nodeList[gangSetIndex];

  for (GridCountPacket& gcp : gcPackets)
    gcp.otherIndex = -1;

  int di = 0;
  for (int c0 = 0; c0 < ColCompatible::Count; ++c0)
  {
    ColCode box4 = codeCompatTable_[0][c0][0];
    ColCode box8 = codeCompatTable_[0][c0][1];

    std::vector<int> x0;
    for (int i = 0; i < ColCompatible::Count; ++i)
      if (codeCompatTable_[0][i][0] == box8)
        x0.push_back(i);
    if (x0.size() == 0)
      throw std::runtime_error("Can't find box8");

    for (int c1 = 0; c1 < ColCompatible::Count; ++c1, ++di)
    {
      ColCode box5 = codeCompatTable_[box1()][c1][0];
      ColCode box9 = codeCompatTable_[box1()][c1][1];

      ColCode box4a = gang_.rename(box5, box4);
      ColCode box5a = gang_.rename(box4a, ColCode(0));

      // ColCode box8a = gang_.rename(box9, box8);
      // ColCode box9a = gang_.rename(box8a, ColCode(0));

      GridCountPacket& gcp = gcPackets[di];

      size_t cacheIndex;
      if (ColNode::nodeList.lookup(box5a, cacheIndex))
        gcp.cacheLevel = (uint8_t)cacheIndex;
      else
        throw std::runtime_error("Can't find cache index");

      for (ColCode cx(0); cx.isValid(); ++cx)
        gcp.doubleRename[cx()] = gang_.rename(box4a, gang_.rename(box5, cx))();

      gcp.multiplier = (uint8_t)(box4 <= box8) + (uint8_t)(box4 < box8);

      std::vector<int> x1;
      for (int i = 0; i < ColCompatible::Count; ++i)
        if (codeCompatTable_[box1()][i][0] == box9)
          x1.push_back(i);
      if (x1.size() == 0)
        throw std::runtime_error("Can't find box9");

      for (int b0 : x0)
        for (int b1 : x1)
        {
          int otherIndex = b0 * ColCompatible::Count + b1;
          if (gcPackets[otherIndex].otherIndex == -1)
          {
            gcPackets[otherIndex].otherIndex = di;
            goto otherOK;
          }
        }
      throw std::runtime_error("Error constructing otherBandIndex_");
    otherOK:;
    }
  }
  ProfileTree::stop(DoubleBoxCount);

  ProfileTree::start("Sort");

  // Sort for preferred run order. multiply 0 elements must all be after the others.
  std::vector<int> gcpIndices(DoubleBoxCount);
  for (int i = 0; i < DoubleBoxCount; ++i)
    gcpIndices[i] = i;
  std::sort(gcpIndices.begin(), gcpIndices.end(),
            [&](int index0, int index1) { return packetOrder_(index0, index1); });
  for (int i = 0; i < DoubleBoxCount; ++i)
    gcPackets[i].runOrder = gcpIndices[i];

  // Get number of box01 indices are active (multiplier > 0)
  for (activeBox01Count_ = 0;
       gcPackets[gcPackets[activeBox01Count_].runOrder].multiplier > 0;
       ++activeBox01Count_);

  ProfileTree::stop();

#ifdef JETSON
  if (gpuEnable_)
  {
    ProfileTree::start("Tables -> GPU");
    gpuSetup(gcPackets);
    ProfileTree::stop();
  }
#endif

  ProfileTree::stop();
}

void GridCounter::countIteration_(int gangOffset, int threadIndex)
{
  if (gangOffset < countSkip_)
    return;

  if (countBackwards_)
    gangOffset = gangSet_->count - gangOffset - 1;

  bool verifying = false;
  if (counts_[gangOffset] != 0)
  {
    verifying = verifyMode_;
    if (!verifying)
      return;
  }

  Timer timer(true);

  int gangIndex = gangSet_->startIndex + gangOffset;
  int gangCode = gang_.gangMembers()[gangIndex];

  ColCode box2 = BandGang::box2(gangCode);
  ColCode box3 = BandGang::box3(gangCode);
  uint16_t box7[ColCompatible::Count], box11[ColCompatible::Count];

  uint64_t count = 0;

  for (uint32_t runIndex = 0; runIndex < activeBox01Count_; ++runIndex)
  {
    uint32_t box01 = gcPackets[runIndex].runOrder;
    const GridCountPacket& gcp0 = gcPackets[box01];

    int box01Other = gcp0.otherIndex;
    const GridCountPacket& gcp1 = gcPackets[box01Other];

    const BandGang::CacheLevel* band1Cache = &gang_.cache(gcp0.cacheLevel);
    const BandGang::CacheLevel* band2Cache = &gang_.cache(gcp1.cacheLevel);

    for (int b3 = 0; b3 < ColCompatible::Count; ++b3)
    {
      int b7  = codeCompatTable_[box3()][b3][0]();
      int b11 = codeCompatTable_[box3()][b3][1]();

      box7 [b3] = gcp0.doubleRename[b7 ];
      box11[b3] = gcp1.doubleRename[b11];
    }

    uint64_t partialCount = 0;
    for (int b2 = 0; b2 < ColCompatible::Count; ++b2)
    {
      int box6  = codeCompatTable_[box2()][b2][0]();
      int box10 = codeCompatTable_[box2()][b2][1]();

      box6  = gcp0.doubleRename[box6 ];
      box10 = gcp1.doubleRename[box10];

      // This is the inner loop, executed 1.04e15 times. Two sequential memory fetches (excellent
      // data cache locality). Two memory references from the gangCache, good but not excellent locality.
      // One 64-bit multiply-accumulate.
      for (int b3 = 0; b3 < ColCompatible::Count; ++b3)
        partialCount += (uint64_t)band1Cache->get(box6, box7[b3]) * band2Cache->get(box10, box11[b3]);
    }

    count += partialCount * gcp0.multiplier;
  }

  double threadTime = timer.elapsedSeconds() / threadCount_;

  bool verified = false;
  if (verifying)
    verified = counts_[gangOffset] == count;
  else
    counts_[gangOffset] = count;

  std::lock_guard<std::mutex> lk(waitLock);

  if (!verifying)
  {
    totalCount_ += count;
    totalTime_ += threadTime;

    std::FILE* file = fopen(countFile_.c_str(), "a");
    if (file)
    {
      CountPacket cp;
      cp.gangIndex = gangIndex;
      cp.count = count;
      cp.thread = threadIndex;
      cp.time = threadTime;
      cp.write(file);
      fclose(file);
    }
    else
      throw std::runtime_error("Can't write count file");
  }
  else
  {
    if (!verified)
    {
      printf("\n%d.%d (%d)\n", gangSetIndex_, gangOffset, gangIndex);
      throw std::runtime_error("Count verification failure");
    }
  }

  fprintf(stderr, "\r%d/%d  %5.1f/%.1f %s",
          gangOffset, gangSet_->count, threadTime, totalTime_, totalCount_.decimal(true).c_str());
}

void GridCounter::count(int gangSet, int threads, const std::string& countFileBase,
                        bool verify, bool countBackwards, int countStartOffset,
                        bool box2GroupMode)
{
  ProfileTree::start("Grid counter");

  countFile_ = countFileBase + "_" + std::to_string(gangSet);
  if (countBackwards)
    countFile_ += '-';
  countFile_ += ".txt";

  verifyMode_ = verify;
  threadCount_ = threads + (int)gpuEnable_;
  countBackwards_ = countBackwards;

  const GangSet& gset = gang_.gangSet(gangSet);
  countSkip_ = countStartOffset;
  if (countBackwards && countStartOffset > 0)
    countSkip_ = gset.count - countStartOffset - 1;

  setup_(gangSet);
  //printMultiplierHist_();

  if (threadCount_ > 0)
  {
    ProfileTree::start("Main count loop");
    if (!box2GroupMode)
    {
      Rope<GridCounter> rope(this, &GridCounter::countIteration_);
      rope.run(gset.count, threads);
    }
    else
    {
      box2GroupWrangler_();
    }
    ProfileTree::stop(gset.count - countSkip_);
    fprintf(stderr, "\n");
  }

  ProfileTree::stop();
}

void GridCounter::printMultiplierHist_() const
{
  IList h;
  for (const GridCountPacket& gcp : gcPackets)
    h.enter(gcp.multiplier);
  h.print("Grid count multipliers");
}

// **************************
// *                        *
// *  Box 2 Group Counting  *
// *                        *
// **************************

void GridCounter::box2GroupIteration_(int runIndex, int threadIndex)
{
  int32_t box01 = gcPackets[runIndex].runOrder;

  ColCode box2 = box2GroupCode_();

#ifdef JETSON
  if (gpuEnable_ && threadIndex == 0)
  {
    gpuMainCount(box01, box2());
  }
  else
#endif

  {
    const GridCountPacket& gcp0 = gcPackets[box01];

    int box01Other = gcp0.otherIndex;
    const GridCountPacket& gcp1 = gcPackets[box01Other];

    std::vector<std::array<uint16_t, ColCompatible::Count>> box7 (box2GroupSize_);
    std::vector<std::array<uint16_t, ColCompatible::Count>> box11(box2GroupSize_);
    for (int groupIndex = 0; groupIndex < box2GroupSize_; ++groupIndex)
    {
      ColCode box3 = box2GroupBox3_[groupIndex];
      for (int b3 = 0; b3 < ColCompatible::Count; ++b3)
      {
        int b7 = codeCompatTable_[box3()][b3][0]();
        int b11 = codeCompatTable_[box3()][b3][1]();

        box7 [groupIndex][b3] = gcp0.doubleRename[b7];
        box11[groupIndex][b3] = gcp1.doubleRename[b11];
      }
    }

    for (int b2 = 0; b2 < ColCompatible::Count; ++b2)
    {
      int box6  = codeCompatTable_[box2()][b2][0]();
      int box10 = codeCompatTable_[box2()][b2][1]();

      box6  = gcp0.doubleRename[box6 ];
      box10 = gcp1.doubleRename[box10];

      const int32_t* band1CacheLine = gang_.cache(gcp0.cacheLevel).address(box6 , 0);
      const int32_t* band2CacheLine = gang_.cache(gcp1.cacheLevel).address(box10, 0);

      for (int groupIndex = 0; groupIndex < box2GroupSize_; ++groupIndex)
      {
        uint64_t count = 0;
        for (int b3 = 0; b3 < ColCompatible::Count; ++b3)
          count += (uint64_t)band1CacheLine[box7[groupIndex][b3]] * band2CacheLine[box11[groupIndex][b3]];
        box2GroupCountsAdd_(groupIndex, threadIndex, count * gcPackets[box01].multiplier);
      }
    }
  }

  if (runIndex % 100 == 0)
    fprintf(stderr, "\r%5d-%5d/%5d: %5d/%d",
            box2GroupStartOffset_, box2GroupStartOffset_ + box2GroupSize_ - 1, gangSet_->count,
            runIndex, activeBox01Count_);
}

void GridCounter::box2GroupWrangler_()
{
  int direction;
  if (!countBackwards_)
  {
    box2GroupStartOffset_ = countSkip_;
    direction = 1;
  }
  else
  {
    box2GroupStartOffset_ = gangSet_->count - countSkip_ - 1;
    direction = -1;
  }

  Timer setTimer(true);
  int gangstersEnumerated = 0;

  while (0 <= box2GroupStartOffset_ && box2GroupStartOffset_ < gangSet_->count)
  {
    if (!verifyMode_ && counts_[box2GroupStartOffset_] > 0)
    {
      box2GroupStartOffset_ += direction;
      continue;
    }

    Timer groupTimer(true);

    // find group size
    box2GroupSize_ = 0;
    for (int gangOffset = box2GroupStartOffset_;
         0 <= gangOffset && gangOffset < gangSet_->count;
         gangOffset += direction)
    {
      int groupOffset = gangOffset - box2GroupStartOffset_;
      if (BandGang::box2(box2GroupGangCode_(groupOffset)) != box2GroupCode_())
        break;
      if (verifyMode_ || counts_[gangOffset] == 0)
        box2GroupSize_ = direction * groupOffset + 1;
    }

    if (countBackwards_)
      box2GroupStartOffset_ -= box2GroupSize_ - 1;

    // Initialize partial counts
    box2GroupCounts_.resize(box2GroupSize_ * threadCount_);
    for (uint64_t& count : box2GroupCounts_)
      count = 0;

    // Fetch box3 codes
    box2GroupBox3_.resize(box2GroupSize_);
    for (int groupIndex = 0; groupIndex < box2GroupSize_; ++groupIndex)
      box2GroupBox3_[groupIndex] = BandGang::box3(box2GroupGangCode_(groupIndex));

    // Set gpu box2 group
#ifdef JETSON
    if (gpuEnable_)
      gpuGroup(box2GroupSize_, (const uint16_t*)box2GroupBox3_.data());
#endif

    // Run all active box01 indices on all threads
    Rope<GridCounter> rope(this, &GridCounter::box2GroupIteration_);
    rope.run(activeBox01Count_, threadCount_);

    // Add box2 group
#ifdef JETSON
    if (gpuEnable_)
      gpuAddGroup(box2GroupCounts_.data(), threadCount_);
#endif

    double secondsPerGangsterGroup = groupTimer.elapsedSeconds() / box2GroupSize_;

    gangstersEnumerated += box2GroupSize_;
    double secondsPerGangsterSet = setTimer.elapsedSeconds() / gangstersEnumerated;

    // show status
    fprintf(stderr, "  %5.2f/%5.2f s/g", secondsPerGangsterGroup, secondsPerGangsterSet);

    std::FILE* file = nullptr;
    for (int groupIndex = 0; groupIndex < box2GroupSize_; ++groupIndex)
    {
      int gangOffset = box2GroupStartOffset_ + groupIndex;
      int gangIndex = gangSet_->startIndex + gangOffset;
      uint64_t count = box2GroupCount_(groupIndex);

      if (counts_[gangOffset] == 0)
      {
        counts_[gangOffset] = count;

        if (!file)
          file = fopen(countFile_.c_str(), "a");
        if (file)
        {
          CountPacket cp;
          cp.gangIndex = gangIndex;
          cp.count = count;
          cp.thread = groupIndex;
          cp.time = secondsPerGangsterGroup;
          cp.write(file);
        }
        else
          throw std::runtime_error("Can't write count file");
      }
      else if (counts_[gangOffset] != count)
      {
        printf("\n%d.%d (%d) expected %s got %s\n",
               gangSetIndex_, gangOffset, gangIndex,
               commas(counts_[gangOffset]).c_str(), commas(count).c_str());
        throw std::runtime_error("Count verification failure");
      }
    }
    if (file)
      fclose(file);

    // Update box2GroupStartIndex_
    if (countBackwards_)
      --box2GroupStartOffset_;
    else
      box2GroupStartOffset_ += box2GroupSize_;
  }
}

// ***************************************************
// *                                                 *
// *  Read and Translate Pettersen's band gangsters  *
// *                                                 *
// ***************************************************
//
// While I don't know Pettersen's method for finding the gangsters and their
// properties, I do have his results. Pettersen has different codes, canonical forms,
// class representatives, and oddly different GangSets as described above. But since
// math is math, the 2006 and 2022 gangsters must be equivalent, meaning a one-to-one
// correspondance with the same gangMembers and bandCounts. The order and box codes
// will be different. The following code allows Pettersen gangsters to be read,
// translated to formats used here, and compared.
// 
// Pettersen finds (by a method I don't know) and uses aut_cnt (automorphism count)
// where I use gangMembers. They are reciprocals. Their product is always 24!^4 * 3! =
// 1990656.

// Here are the Pettersen gangster fields. The names come from his documentation of
// the binary format files. I did not look at the source code.
struct PGang
{
  uint32_t index;     // Pettersen index
  uint32_t boxes[3];  // Pettersen box codes
  uint32_t aut_cnt;   // automorphism count
  uint32_t cnt;       // same as my bandCount
};

static uint64_t getBytes(std::ifstream& in, int n)
{
  uint64_t x = 0;
  char b;
  for (int i = 0; i < n; ++i)
  {
    in.get(b);
    x = (x << 8) | (uint8_t)b;
  }
  return x;
}

static void readPGang(std::vector<PGang>& gang)
{
  gang.clear();
  std::ifstream in("gangsters.res", std::ios::in | std::ios::binary);
  while (!in.eof())
  {
    PGang pg;

    pg.index = (uint32_t)getBytes(in, 3);
    if (in.eof())
      break;
    for (int i = 0; i < 3; ++i)
      pg.boxes[i] = (uint32_t)getBytes(in, 2);
    pg.aut_cnt = (uint32_t)getBytes(in, 3);
    pg.cnt = (uint32_t)getBytes(in, 3);

    gang.push_back(pg);
  }
  in.close();
}

static void writePGang(std::vector<PGang>& gang, const char* file)
{
  std::ofstream out(file);
  for (const PGang& pg : gang)
  {
    out << strFormat("%8u, %4u.%4u.%4u, %8u, %8u\n",
                     pg.index, pg.boxes[0], pg.boxes[1], pg.boxes[2], pg.aut_cnt, pg.cnt);
  }
  out.close();
}

// Find and print information equivalent to the GangSets
static void analyzePGang(std::vector<PGang>& gang)
{
  int numGangsters = 0;
  uint64_t sumaut = 0, sumcnt = 0;	// init to keep GCC quiet
  uint64_t allaut = 0, allcnt = 0;
  int box0 = -1;
  //printf("Node  Gangsters     Members     Fill%%\n");
  for (const PGang& pg : gang)
  {
    int b0 = pg.boxes[0];
    if (b0 != box0)
    {
      if (box0 >= 0)
        printf("%4d  %8d  %14llu  %14llu\n", box0, numGangsters, sumaut, sumcnt);
      sumaut = sumcnt = numGangsters = 0;
      box0 = b0;
    }
    ++numGangsters;
    sumcnt += pg.cnt;
    sumaut += pg.aut_cnt;
    allcnt += pg.cnt;
    allaut += pg.aut_cnt;
  }
  printf("%4d  %8d  %14llu  %14llu\n", box0, numGangsters, sumaut, sumcnt);

  Bignum aut = allaut, cnt = allcnt;
  printf("%llu gangsters\naut_cnt = %s = %s\ncnt = %s = %s\n",
         gang.size(), aut.decimal(true).c_str(), aut.primeFact().c_str(),
         cnt.decimal(true).c_str(), cnt.primeFact().c_str());
}

// **********************************************
// *                                            *
// *  Read and Translate Pettersen Grid Counts  *
// *                                            *
// **********************************************

// The present grid counting method considers each gangster in turn and obtains the
// complete count for that gangster. Each gangster is porcessed autonomously,
// independent of every other.
//
// While I don't know Pettersen's method for grid counting, I can see from the
// result files main*.res that a gangster's grid count is spread out over
// up to 45 partial counts in separate files. This class finds Pettersen partial
// grid counts, collates and sums them, and writes a std::vector<Gangster>, which
// can then be used like any other (make human-readable file, comparison with
// present method). Various errors are detected (throw std::runtime_error as
// usual), which would indicate either bugs or my misunderstanding of the result
// data.
//
// Pettersen's method is more sophisticated and efficient than mine. I had the enormous
// benefit of 64-bit multicore CPUs, Pettersen had to squeeze into 32-bit single-core
// machines. My total running time is about 40% of Pettersen, all due to using every
// parallel thread of 2- and 4-core hyperthreaded 64-bit machines, and about 1.4 GB of
// lookup tables. This does make the present version simpler, always a good thing.
//
// I'm not sure which of the main*.res files are the official run and which are
// partial duplicates from test runs and whatever else was going on. So I just take
// them all, and throw if any duplicate has a mismatched count. There are hundreds
// of duplicates and no mismatches.
//
// My grid counts are always twice Pettersen's. Pettersen must have found a two-way
// symmetry that I still don't understand. The important question is whether that
// symmetry is just an abritraty scale factor, or that Pettersen has found a way to do
// half the work. With computation time measured in weeks, 2x is gold.
class PCountFinder
{
public:
  PCountFinder();

  void find(std::vector<Gangster>& gangsters);

private:
  uint64_t counts_[45][GangNum];

  static int baseIndex_(int top, int bot);
};

PCountFinder::PCountFinder()
{
  for (int k = 0; k < 45; ++k)
    for (int g = 0; g < GangNum; ++g)
      counts_[k][g] = 0;
}

int PCountFinder::baseIndex_(int top, int bot)
{
  if (top == bot)
    return top + 36;
  else if (top < bot)
    return bot * (bot - 1) / 2 + top;
  else
    return top * (top - 1) / 2 + bot;
}

void PCountFinder::find(std::vector<Gangster>& gangsters)
{
  const std::filesystem::path path{ "C:/Users/bsilv/Documents/VStudio Projects/bignum2/Sudoku4x3/4x3/final" };
  int files = 0;

  for (auto const& file : std::filesystem::directory_iterator{ path })
  {
    std::string filename = file.path().filename().u8string();
    if (file.path().extension() == ".res" && filename.substr(0, 4) == "main")
    {
      std::ifstream in(file.path(), std::ios_base::binary);
      if (!in.is_open())
      {
        printf("Can't open %s\n", filename.c_str());
        continue;
      }

      int duplicates = 0;
      int mismatches = 0;
      int k = -1;
      ++files;

      while (true)
      {
        int reclen = (int)getBytes(in, 1);
        if (in.eof())
          break;

        int code = (int)getBytes(in, 1);
        int bytesRead = 1;

        switch (code)
        {
        case 0:
        {
          int top = (int)getBytes(in, 1);
          int bot = (int)getBytes(in, 1);
          k = baseIndex_(top, bot);
          bytesRead += 2;
          break;
        }

        case 1:
          k = -1;
          break;

        case 2:
          break;

        case 3:
        {
          if (k < 0)
          {
            printf("End record with no start in %s\n", filename.c_str());
            goto skip;
          }

          uint64_t g = (int)getBytes(in, 3);
          if (g >= GangNum)
          {
            printf("Bad gangster index in %s\n", filename.c_str());
            goto skip;
          }

          /*uint64_t hits =*/ getBytes(in, 2);
          uint64_t count = getBytes(in, 8);
          bytesRead += 13;

          if (counts_[k][g] == 0)
            counts_[k][g] = count;
          else if (counts_[k][g] == count)
            ++duplicates;
          else
            ++mismatches;

          break;
        }

        default:
          printf("%s bad format, skipping\n", filename.c_str());
          goto skip;
        }

        for (int i = 0; i < reclen - bytesRead; ++i)
          getBytes(in, 1);
      }

      if (duplicates > 0)
        printf("%d duplicates in %s\n", duplicates, filename.c_str());

      if (mismatches > 0)
        printf("%d mismatches in %s\n", mismatches, filename.c_str());

    skip:;
      in.close();
    }
  }

  printf("%d Pettersen count files read\n", files);

  std::vector<PGang> pGang;
  readPGang(pGang);

  constexpr uint32_t CanonPerm = factorial<uint32_t>(3) * ipower<uint32_t>(factorial<uint32_t>(4), 4);

  gangsters.resize(GangNum);
  int box1Codes[9] = { 0, 1, 9, 36, 44, 66, 604, 614, 716 };
  for (uint32_t gi = 0; gi < GangNum; ++gi)
  {
    int i;
    for (i = 0; i < 9; ++i)
      if (pGang[gi].boxes[0] == box1Codes[i])
        break;
    if (i == 9)
      throw std::runtime_error(strFormat("Box 1 code %u not found", pGang[gi].boxes[0]));

    Gangster& g = gangsters[gi];
    g.nodeIndex = i;
    g.gangIndex = gi;
    g.bandCount = pGang[gi].cnt;
    g.gangMembers = CanonPerm / pGang[gi].aut_cnt;
  }

  for (int g = 0; g < GangNum; ++g)
  {
    uint64_t sum = 0;
    int numK = 0;
    for (int k = 0; k < 45; ++k)
      if (counts_[k][g] > 0)
      {
        ++numK;
        sum += counts_[k][g];
      }

    gangsters[g].gridCount = sum << 1;
  }
}

// ***************************************
// *                                     *
// *  Kilfoil-Silver-Pettersen Estimate  *
// *                                     *
// ***************************************

void kspEstimate()
{
  // Band 0 count
  uint64_t band0Count = 0;
  for (int i1 = 0; i1 < RowCompatible::Count - 1; ++i1)
  {
    RowSets box1 = RowCompatible::canon[i1];
    for (int i2 = i1 + 1; i2 < RowCompatible::Count; ++i2)
      band0Count += (int)(RowCompatible::canon[i2] & box1).isCompatible();
  }

  Bignum band0All = Bignum(6).power(12) * band0Count * (2 * factorial<int>(12));
  printf("\nBand 0 count, %llu classes:\n%s, %e, %s\n",
         band0Count, band0All.decimal(true).c_str(), band0All.makeDouble(), band0All.primeFact().c_str());

  // Stack 0 count
  Bignum stack0All = Bignum(24).power(6) * ColCompatible::Count * factorial<int>(12);
  printf("\nStack 0 count:\n%s, %e, %s\n",
         stack0All.decimal(true).c_str(), stack0All.makeDouble(), stack0All.primeFact().c_str());

  double b43 = band0All.makeDouble();
  double b34 = stack0All.makeDouble();
  double ksp = std::pow(b43, 3) * std::pow(b34, 4) / std::pow(factorial<double>(12), 12);
  printf("\nKSP estimate for 3x4 = %e\n", ksp);
}

// ***********************************
// *                                 *
// *  Make Commandline Usage String  *
// *                                 *
// ***********************************

static const char* commandline()
{
  return
    "sudoku3x4 switches\n\n"
    "Switches are a single case-sensitive character followed by switch-specific\n"
    "information. No spaces appear in any switch\n\n"
    "  b      compute band counts (can also be read from file with r and verified)\n"
    "  c<w>   print grid count progress report and, if grid counting done, the final result.\n"
    "         If combined with P, Pettersen and Silver gangsters are compared and reported.\n"
    "         If <w> is 'w' write human-readable copy to sGang.txt\n"
    "  e      compute and print the KSP estimate\n"
    "  f<baseFile> Set base filename for count files used by c and g. Count files will be\n"
    "              <basefile>_g<set><dir>.txt. If f is not given, <baseFile> is 'gridCount'\n"
    "  g<set><dir> begin grid counting\n"
    "         <set>      GridSet 0 - 8, or ! for all sets\n"
    "         <dir>      optional '-' for count backwards\n"
    "  G<blocks>,<threads> enable GPU, force box2 group counting\n"
    "         Can specify <blocks>,<threads>, just <blocks>, or neither, default is emperical best\n"
    "  k<setOffset> If g is given, begin grid counting at <setOffset> within the GridSet,\n"
    "         and work forward or backward from there\n"
    "  m      compute gangMembers (can also be read from file with r and verified)\n"
    "  N      Don't use CPU threads for grid counting. If GPU counting is also not selected, counting\n"
    "         is skipped and only the setup functions are run for testing and timing\n"
    "  p<w>   read Pettersen gangsters, translate, and print report. If <w> is 'w' write\n"
    "         human-readable copy to pettersenGang.txt\n"
    "  P<w>   read Pettersen gangsters and grid counts, translate. If <w> is 'w' write human-\n"
    "         readable copy to pGang.txt.\n"
    "  q      Quit after processing switches\n"
    "  r<gangFile> Read gangster file. If b or m switches given, verify, otherwise load from\n"
    "         file. Load/verify handled independently for b amd m. <gangFile> optional, default\n"
    "         is bandGang.txt.\n"
    "  s<options> Show (print) stuff. <options> in any combination and order\n"
    "         c show gangster count summaries\n"
    "         n show node data\n"
    "         t show execution times\n"
    "         * show everything\n"
    "  t<n>   Set to use <n> threads, or <n> = 0 for all virtual processors. Default is 1.\n"
    "  v      If g also given, count in verify mode\n"
    "  2<use> <use> omoitted or + to use box2 group counting mode , <use> - or switch omitted\n"
    "         to use original counting mode\n"
    "  w<gangFile> Write gangster file. <gangFile> optional, default is bandGang.txt.\n"
    "  !      Print GPU properties\n"
    "  ?      Print this\n"
    ;
}

// ******************
// *                *
// *  Main Program  *
// *                *
// ******************

int main(int argc, char* argv[])
{
  constexpr const char* defaultGangFilename = "bandGang.txt";

  // Scan flags
  bool showNodes = false, showTimers = false, showGangCounts = false;
  bool dontRun = false, ksp = false, noCount = false;
  bool computeMemberCounts = false, computeBandCounts = false;
  bool verifyMode = false, countBackwards = false, gpuEnable = false;
  bool countAll = false, writeAll = false, box2GroupMode = false;

  const char* readFilename = nullptr;
  const char* writeFilename = nullptr;
  const char* countFilenameBase = "gridCount";

  int numThreads = 1, gridSet = -1, countStartOffset = 0;
  int gpuThreads = 32, gpuBlocks = 88;

  std::vector<Gangster> pGang;

  try
  {
    for (int i = 1; i < argc; ++i)
    {
      switch (*argv[i])
      {
      case 'b':
        computeBandCounts = true;
        break;

      case 'c':
        countAll = true;
        writeAll = argv[i][1] == 'w';
        break;

      case 'e':
        // KSP estimate
        ksp = true;
        break;

      case 'f':
        countFilenameBase = argv[i] + 1;
        if (!*countFilenameBase)
        {
          fprintf(stderr, "Bad base filename");
          dontRun = true;
        }
        break;

      case 'g':
        if (argv[i][1] == '!')
          gridSet = 9;
        else
        {
          gridSet = argv[i][1] - '0';
          if (gridSet < 0 || gridSet > 8)
          {
            fprintf(stderr, "Bad grid set\n");
            dontRun = true;
            break;
          }
        }

        countBackwards = argv[i][2] == '-';
        break;

      case 'G':
#ifdef JETSON
        gpuEnable = true;
        box2GroupMode = true;
        sscanf(argv[i] + 1, "%d,%d", &gpuBlocks, &gpuThreads);
        if (1 <= gpuBlocks && gpuBlocks <= 256 && 1 <= gpuThreads && gpuThreads <= 1024)
        {
          gpuGrid(gpuBlocks, gpuThreads);
          printf("GPU using %d blocks, %d threads\n", gpuBlocks, gpuThreads);
        }
        else
        {
          fprintf(stderr, "Bad GPU block,thread count\n");
          dontRun = true;
        }
#else
        printf("No GPU\n");
#endif
        break;

      case 'k':
        if (sscanf(argv[i], "k%d", &countStartOffset) != 1)
        {
          printf("Bad count skip value %s\n", argv[i] + 1);
          dontRun = true;
        }
        break;

      case 'm':
        computeMemberCounts = true;
        break;

      case 'N':
        noCount = true;
        break;

      case 'p':
      {
        // Read KFP's gangsters
        std::vector<PGang> gang;
        readPGang(gang);
        if (argv[i][1] == 'w')
          writePGang(gang, "pettersenGang.txt");
        analyzePGang(gang);
        break;
      }

      case 'P':
      {
        auto pcf = std::make_unique<PCountFinder>();
        pcf->find(pGang);
        std::sort(pGang.begin(), pGang.end());
        if (argv[i][1] == 'w')
        {
          Gangster::writeAll(pGang, "pGang.txt");
          Gangster::duplicates(pGang);
        }
        break;
      }

      case 'q':
        dontRun = true;
        break;

      case 'r':
        // read filename
        readFilename = argv[i] + 1;
        if (*readFilename == 0)
          readFilename = defaultGangFilename;
        break;

      case 's':
        // Show flags
        for (int p = 1; argv[i][p]; ++p)
          switch (argv[i][p])
          {
          case 'c':
            showGangCounts = true;
            break;

          case 'n':
            showNodes = true;
            break;
          case 't':
            showTimers = true;
            break;
          case '*':
            showNodes = true;
            showTimers = true;
            showGangCounts = true;
            break;
          default:
            fprintf(stderr, "%c unknown display code\n", argv[i][p]);
            dontRun = true;
          }
        break;

      case 't':
      {
        // Set number of threads
        if (sscanf(argv[i], "t%d", &numThreads) == 1 && numThreads >= 0 && numThreads <= 32)
        {
          if (numThreads == 0)
            numThreads = std::thread::hardware_concurrency();
        }
        else
        {
          fprintf(stderr, "Bad thread count %s\n", argv[i] + 1);
          dontRun = true;
        }
        break;
      }

      case 'v':
        verifyMode = true;
        break;

      case 'w':
        // write filename
        writeFilename = argv[i] + 1;
        if (*writeFilename == 0)
          writeFilename = defaultGangFilename;
        break;

      case '2':
        switch (argv[i][1])
        {
        case '+':
        case 0:
          box2GroupMode = true;
          break;
        case '-':
          box2GroupMode = false;
          break;
        default:
          fprintf(stderr, "%c unknown box2 group mode\n", argv[i][1]);
          dontRun = true;
        }
        break;

      case '!':
#ifdef JETSON
        printDeviceProperties();
#else
        printf("No GPU\n");
#endif
        break;

      case '?':
        std::cout << commandline();
        dontRun = true;
        break;

      default:
        fprintf(stderr, "%s unknown flag\n", argv[i]);
        dontRun = true;
      }
    }

    if (dontRun)
      return 1;

    if (gpuEnable && !box2GroupMode)
    {
      printf("Can't reject box2 group counting mode with GPU\n");
      return 1;
    }

    printf("CPU using %d thread%s\n", numThreads, numThreads == 1 ? "" : "s");

    if (showTimers)
      ProfileTree::clear();
    ProfileTree::start("Sudoku3x4");

    RowCode::init();
    ColCode::init();

    RowTables::init();
    RowTables::verify();
    ColTables::init();
    ColTables::verify();

    RowCompatible::init();
    ColCompatible::init();
    RowCompatible::verify();
    ColCompatible::verify();

    if (ksp)
      kspEstimate();

    ColNode::init();
    ColNode::verify();
    if (showNodes)
      ColNode::print();

    // Construct band gangster object
    auto bGang = std::make_unique<BandGang>();
    bGang->verifyTables();

    // Find all gangsters and fill cache
    bGang->find(numThreads);

    // Count members of each gangster equivalence class, set counts in bGang->gangMembers()
    if (computeMemberCounts)
      bGang->countMembers(numThreads);

    // Count band configurations for each gangster
    if (computeBandCounts)
      bGang->countBands(numThreads);

    if (readFilename)
      bGang->readFile(readFilename, computeMemberCounts, computeBandCounts);

    if (!readFilename && (!computeMemberCounts || !computeBandCounts))
    {
      fprintf(stderr, "Not enough gangster data read or computed, quitting\n");
      return 1;
    }

    bGang->replaceCacheCodesWithBandCounts();

    bGang->computeGangSets();

    if (countAll)
    {
      std::vector<Gangster> gang;
      bGang->countAll(gang, countFilenameBase);
      if (writeAll)
      {
        Gangster::writeAll(gang, "sGang.txt");
        Gangster::duplicates(gang);
      }

      if (pGang.size() > 0)
      {
        std::sort(gang.begin(), gang.end());
        Gangster::compare(pGang, gang);
      }
    }

    const IList& gang = bGang->gangMembers();

    // Print GangSets
    if (showNodes)
    {
      printf("\nSet  Gangsters  StartIndex  Unique Codes    GCD\n");
      for (int si = 0; si < 9; ++si)
      {
        const GangSet& gs = bGang->gangSet(si);
        printf(" %d    %6d     %6d       %6d       %4d\n",
               si, gs.count, gs.startIndex, gs.uniqueBoxCodes, gs.gcdBandCounts);
      }
    }

    // Print totals
    if (showGangCounts)
    {
      uint64_t totalBandCounts = 0;
      Bignum totalConfig = 0;
      for (int gi = 0; gi < (int)gang.size(); ++gi)
      {
        totalBandCounts += bGang->bandCount(gi);
        totalConfig += Bignum(bGang->bandCount(gi)) * gang.count(gi);
      }
      printf("\nTotals:\n"
             "  Members        = %22s\n"
             "  Band Counts    = %22s\n"
             "  Configurations = %22s = %.6e = %s\n",
              commas(gang.totalCount()).c_str(), commas(totalBandCounts).c_str(),
              totalConfig.decimal(true).c_str(), totalConfig.makeDouble(), totalConfig.primeFact().c_str());
    }

    // Write band gangsters to file
    if (writeFilename)
      bGang->writeFile(writeFilename);

    // Count complete grid
    if (gridSet >= 0)
    {
      auto gridCounter = std::make_unique<GridCounter>(*bGang, gpuEnable);
      if (gridSet <= 8)
        gridCounter->count(gridSet, noCount ? 0 : numThreads, countFilenameBase, verifyMode,
                           countBackwards, countStartOffset, box2GroupMode);
      else
      {
        for (int set = 0; set < 9; ++set)
          gridCounter->count(set, noCount ? 0 : numThreads, countFilenameBase, verifyMode,
                             countBackwards, countStartOffset, box2GroupMode);

        std::vector<Gangster> gang;
        bGang->countAll(gang, countFilenameBase);
        if (writeAll)
          Gangster::writeAll(gang, "sGang.txt");
      }

#ifdef JETSON
      sudokudaEnd();
#endif
    }

    ProfileTree::stop();
    if (showTimers)
      std::cout << '\n' << ProfileTree::toString();
  }
  catch (const std::exception& ex)
  {
    fprintf(stderr, "Error %s\n", ex.what());
  }

  return 0;
}
