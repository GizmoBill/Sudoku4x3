// COPYRIGHT (C) 2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

#ifndef _symmetrytable_
#define _symmetrytable_

// ********************
// *                  *
// *  Symmetry Table  *
// *                  *
// ********************
//
// A 2D Dim * Dim table of type T, where the value at (x, y) and (y, x) are the same.
// The !Small version uses Dim * Dim * sizeof(T) bytes, storing the same value at (x, y)
// and (y, x). The Small version uses Dim * (Dim + 1) / 2 * sizeof(T) bytes, storing
// only one value for (x, y) and (y, x). The idea is that while it takes a little
// longer to compute the address of the value in the Small version, the use of less
// memory shoud improve data cache performance. Emperically, the !Small version is
// faster in this application.

template<typename T, uint32_t Dim, bool Small>
class SymTable
{
public:
  const T& get(uint32_t x, uint32_t y) const { return table_[x][y]; }

  // Set new value and return previous value
  T set(uint32_t x, uint32_t y, const T& value)
  {
    T v = table_[x][y];
    table_[x][y] = table_[y][x] = value;
    return v;
  }

  void setAll(const T& value)
  {
    for (uint32_t i = 0; i < Dim; ++i)
      for (uint32_t j = 0; j < Dim; ++j)
        table_[i][j] = value;
  }

  T* address(uint32_t x, uint32_t y) { return &table_[x][y]; }
  const T* address(uint32_t x, uint32_t y) const { return &table_[x][y]; }

private:
  T table_[Dim][Dim];
};

// Specialization for Small
template<typename T, uint32_t Dim>
class SymTable<T, Dim, true>
{
public:
  const T& get(uint32_t x, uint32_t y) const { return *address(x, y); }

  // Set new value and return previous value
  T set(uint32_t x, uint32_t y, const T& value)
  {
    T& vRef = *address(x, y);
    T v = vRef;
    vRef = value;
    return v;
  }

  void setAll(const T& value)
  {
    for (int i = 0; i < size_; ++i)
      table_[i] = value;
  }

  T* address(uint32_t x, uint32_t y)
  {
    uint32_t u = std::min(x, y);
    uint32_t v = std::max(x, y);
    return table_ + (((uint64_t)v * (v + 1)) >> 1) + u;
  }

private:
  static constexpr size_t size_ = (size_t)Dim * (Dim + 1) / 2;
  T table_[size_];
};

#endif
