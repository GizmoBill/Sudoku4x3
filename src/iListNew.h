// COPYRIGHT (C) 2005-2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

#ifndef __iList__
#define __iList__

// *****************************
// *                           *
// *  Simple List of Integers  *
// *                           *
// *****************************
//
// IList is primarily intended for computing a histogram of a sequence of integer codes, specifically
// a list of each distinct code in the sequence and a corresponding count of the number of occurrences
// of the code. For this use, each code in the sequence is passed to the enter() function. The class
// also provides high-performance searching and sorting of the list of codes, and is used for
// purposes beyond computing histograms.

#include "general.h"
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

template<class T, class U = uint32_t>
class NumList
{
public:
  struct Elem
  {
    T code;
    U count;

    Elem() {}
    Elem(T code) : code(code) {}
    Elem(T code, U count) : code(code), count(count) {}

    bool operator==(const Elem& e) const { return code == e.code; }
    bool operator< (const Elem& e) const { return code <  e.code; }
  };

  NumList& operator=(const NumList&) = default;
  NumList(const NumList&) = default;

  NumList(size_t expected = 0);
  // Constructor optionally allocates elements. Needs to be optional so client can make array of
  // ILists if desired.

  size_t size() const { return list_.size();}
  // Number of elements currently in the list

  size_t expected() const { return expected_;}
  void expected(size_t n);
  // Get/set number of elements expected to be added to list.

  bool overflow() const { return size() > expected();}
  // True if more codes have been entered or appended than expected

  bool autoSort() const { return autoSort_;}
  void autoSort(bool x) { autoSort_ = x;}
  // Get/set autoSort mode. When autoSort is true, the list will automatically be sorted
  // as soon as it fills up, which will speed up subsequent calls to lookup() and enter().
  // Be aware that all indicies may change when the list is sorted. By default autoSort
  // is false.

  T operator[] (size_t i) const { return list_[i].code;}
  // Return ith code.

  U count(size_t i) const { return list_[i].count;}
  void count(size_t i, U n) { list_[i].count = n;}
  void addCount(size_t i, U n) { list_[i].count += n;}
  // Get/set ith count

  uint64_t totalCount() const;

  void clear() { list_.clear(); sorted_ = true; }
  // Remove all elements from list

  bool lookup(T code, size_t& index) const;
  // If found, set index corresponding to code, return true. Othereise leave index
  // unchanged, return false. Fast binary search if list is sorted.

  size_t enter(T code, U count = 1);
  // Add count to running sum corresponding to code, creating new entry for code if necessary.
  // Returns index of list element, -1 if a new element needs to be created but there is no room.
  // Faster if list is sorted.

  size_t append(T code, U count);
  // Add new list element, returns index. Client to insure that new element does
  // not already exist.

  void sort(bool force = false);
  // Sort list for fast lookups. List may be sorted at any time; adding new elements may cause the
  // list to become unsorted. Fast n*log(n) method. Does nothing if already sorted, unless force.

  bool isSorted() const { return sorted_; }

  bool keepSorted() const { return keepSorted_; }

  void keepSorted(bool);

  void eliminateDuplicates();
  // Duplicates are removed, and counts are aggregated in the one representative element 

  auto begin() const { return list_.begin(); }
  auto end  () const { return list_.end  (); }

  bool verify(const char* id = 0) const;
  // Verify that size() == expected(). If not, print a message that includes the optional id.

  void print(const char* id = 0, const char* format = "%4d:%6d [%4d]", int maxToPrint = 300) const;
  // Print the contents of the list.

private:
  std::vector<Elem> list_;

  size_t expected_;
  bool autoSort_;
  bool sorted_;
  bool keepSorted_;
};

typedef NumList<int32_t> IList;

template<class T, class U>
NumList<T, U>::NumList(size_t expected)
: autoSort_(false), sorted_(true)
{
  this->expected(expected);
}

template<class T, class U>
void NumList<T, U>::expected(size_t n)
{
  expected_ = n;
  list_.reserve(n);
}

template<class T, class U>
uint64_t NumList<T, U>::totalCount() const
{
  uint64_t n = 0;
  for (const Elem& e : list_)
    n += e.count;
  return n;
}

template<class T, class U>
void NumList<T, U>::sort(bool force)
{
  if (!isSorted() || force)
  {
    std::sort(list_.begin(), list_.end());
    sorted_ = true;
  }
}

template<class T, class U>
bool NumList<T, U>::lookup(T code, size_t& index) const
{
  if (!sorted_ || size() < 16)
  {
    // linear search for non-sorted or small lists
    auto p = std::find(list_.begin(), list_.end(), Elem(code));
    if (p != list_.end())
    {
      index = p - list_.begin();
      return true;
    }
  }
  else
  {
    // binary search
    auto p = std::lower_bound(list_.begin(), list_.end(), Elem(code));
    if (p != list_.end() && p->code == code)
    {
      index = p - list_.begin();
      return true;
    }
  }

  return false;
}

template<class T, class U>
size_t NumList<T, U>::append(T code, U count)
{
  if (!keepSorted())
  {
    size_t i = size();
    list_.push_back(Elem(code, count));

    if (sorted_ && i > 0 && list_[i] < list_[i - 1])
      sorted_ = false;

    if (!sorted_ && autoSort_ && size() == expected())
    {
      sort();
      lookup(code, i);
    }

    return i;
  }
  else
  {
    if (!isSorted())
      throw std::runtime_error("Unsorted NumList but keepSorted is true");

    auto p = std::lower_bound(list_.begin(), list_.end(), Elem(code));
    p = list_.insert(p, Elem(code, count));
    return p - list_.begin();
  }
}

template<class T, class U>
size_t NumList<T, U>::enter(T code, U count)
{
  size_t i;
  if (lookup(code, i))
    list_[i].count += count;
  else
    i = append(code, count);
  return i;
}

template<class T, class U>
void NumList<T, U>::eliminateDuplicates()
{
  sort();
  NumList concise;
  T code = list_[0].code;
  U n = 0;
  for (const Elem& e : list_)
    if (e.code != code)
    {
      concise.append(code, n);
      code = e.code;
      n = e.count;
    }
    else
      n += e.count;
}

template<class T, class U>
void NumList<T, U>::keepSorted(bool b)
{
  if (b)
    sort();
  keepSorted_ = b;
}

template<class T, class U>
bool NumList<T, U>::verify(const char* id) const
{
  if (size() != expected())
  {
    if (id)
      printf("%s ", id);
    printf("expected %llu, got %llu\n", expected(), size());
    return true;
  }
  return false;
}

template<class T, class U>
void NumList<T, U>::print(const char* id, const char* format, int maxToPrint) const
{
  uint32_t total = 0;
  for (const Elem& e : list_)
    total += e.count;
  if (id)
    printf("\n%llu distinct elements of %s, total = %d:\n", size(), id, total);

  if (size() > 0)
  {
    int numToPrint = std::min((int)size(), maxToPrint);
    bool nl = format[strlen(format) - 1] == '\n';
    int cols = 1;
    if (!nl)
      cols = std::min((numToPrint + 15) / 16, 6);
    int rows = (numToPrint + cols - 1) / cols;
    for (int i = 0; i < rows; ++i)
    {
      for (int j = i; j < numToPrint; j += rows)
      {
        printf(format, j, list_[j].code, list_[j].count);
        printf("  ");
      }
      if (!nl)
        printf("\n");
    }

    if (size() != (size_t)numToPrint)
      printf("%llu more ...\n", size() - numToPrint);
  }
}

// ***********
// *         *
// *  T Set  *
// *         *
// ***********
//
// A set of objects of type T that are ordered by operator<, and that therefore can be sorted
// and subject to binary search. The objects may be expensive to copy, and so an array of
// pointers to the objects is maintained internally for fast sorting.
//
// T must support assignment, operator==, operator<.

template<class T>
class TSet
{
  T*   set_;
  T**  list_;
  int  allocated_;
  int  size_;
  bool overflow_;
  bool autoSort_;
  bool sorted_;
  int  sortTrigger_;
  int  sortTrigCount_;

  void sort(T**, int, T**);

  TSet& operator=(const TSet&);
  TSet(const TSet&);
  // no copies yet

public:

  TSet(int expected = 0);
  // Constructor optionally allocates elements. Needs to be optional so client can make array of
  // TSet if desired.

  ~TSet() { alloc(0);}

  int size() const { return size_;}
  // Number of elements currently in the set

  int alloc() const { return allocated_;}
  void alloc(int expected);
  // Get/set max number of elements the set can hold. Setting the value clears the set.
  // Note that 0 is a valid value.

  bool overflow() const { return overflow_;}
  // True if more codes have been entered or appended than will fit

  bool autoSort() const { return autoSort_;}
  void autoSort(bool x) { autoSort_ = x;}
  // Get/set autoSort mode. When autoSort is true, the set will automatically be sorted
  // as soon as it fills up, which will speed up subsequent calls to lookup() and enter().
  // By default autoSort is false.

  int  sortTrigger() const { return sortTrigger_;}
  void sortTrigger(int x) { sortTrigger_ = x; sortTrigCount_ = 0;}
  // Get/set sortTrigger value. List is automatically sorted when sortTrigger lookups
  // have been done on an unsorted, non-tiny list with no intervening additions to
  // the list. If sortTrigger is 0 (default), do nothing.

  const T& operator[] (int i) const { BC(i, size_); return set_[i];}
  // Return ith element.

  const T& get(int s) const { BC(s, size_); return *list_[s];}

  int index(int s) const { BC(s, size_); return list_[s] - set_;}

  void clear() {size_ = 0; overflow_ = false; sorted_ = false; sortTrigCount_ = 0;}
  // Remove all elements from set

  int lookup(const T& code) const;
  // Return index corresponding to code, or -1 if not found. Fast binary search if set is
  // sorted.

  int enter(const T& code);
  // Return index corresponding to code, creating new entry for code if necessary.
  // Returns -1 if a new element needs to be created but there is no room.
  // Faster if list is sorted.

  int append(const T& code);
  // Add new set element, returns index or -1 if no room. Client to insure that new element does
  // not already exist.

  void sort();
  // Sort set for fast lookups. Set may be sorted at any time; adding new elements causes the
  // set to become unsorted. Fast n*log(n) method.

  bool verify(const char* id = 0) const;
  // Verify that size() == alloc(). If not, print a message that includes the optional id.

  //void print(const char* id = 0) const;
  // Print the contents of the set.
};

#if false
// ***********
// *         *
// *  T Map  *
// *         *
// ***********

template<class D, class R>
class TMap : public TSet<D>
{
  R* map_;

public:
  TMap(int expected = 0);
 ~TMap();

  void alloc(int expected);
  int alloc() const { return TSet<D>::alloc();}

  int append(const D&, const R&);

  const R& range(int i) const { BC(i, size_); return map_[i];}
        R& range(int i)       { BC(i, size_); return map_[i];}

  const R& map(const D& d) const { int i = lookup(d); assert(i >= 0); return map_[i];}
        R& map(const D& d)       { int i = lookup(d); assert(i >= 0); return map_[i];}
};

template<class D, class R>
class THist : public TMap<D,R>
{
public:
  THist(int expected);

  int enter(const D&, const R&);
};

// ***********
// *         *
// *  T Set  *
// *         *
// ***********

template<class T>
TSet<T>::TSet(int expected)
: set_(0), list_(0), autoSort_(false), sortTrigger_(0)
{
  alloc(expected);
}

template<class T>
void TSet<T>::alloc(int expected)
{
  clear();
  delete[] set_;
  delete[] list_;
  allocated_ = expected;
  if (allocated_)
  {
    set_ = new T[allocated_];
    list_ = new T*[allocated_];
  }
  else
  {
    set_ = 0;
    list_ = 0;
  }
}

template<class T>
int TSet<T>::lookup(const T& code) const
{
  enum {TinySize = 16};
  if (!sorted_ && size_ >= TinySize && ++((TSet<T>*)this)->sortTrigCount_ > sortTrigger_)
    ((TSet<T>*)this)->sort();

  if (!sorted_ || size_ < TinySize)
  {
    // linear search for non-sorted or small lists
    for (int i = 0; i < size_; ++i)
      if (set_[i] == code)
	return i;
    return -1;
  }
  else
  {
    // binary search
    int n = size_, i = 0;
    while (n)
    {
      int k = n >> 1, j = i + k;
      if (*list_[j] < code)
      {
	i = j + 1;
	n -= k + 1;
      }
      else
	if (*list_[j] == code)
	  return list_[j] - set_;
	else
	  n = k;
    }
    return -1;
  }
}

template<class T>
int TSet<T>::append(const T& code)
{
  if (size_ < allocated_)
  {
    int i = size_++;
    set_[i] = code;
    list_[i] = set_ + i;
    sorted_ = false;
    sortTrigCount_ = 0;
    if (autoSort_ && size_ == allocated_)
      sort();
    return i;
  }

  overflow_ = true;
  return -1;
}

template<class T>
int TSet<T>::enter(const T& code)
{
  int i = lookup(code);
  if (i < 0)
    i = append(code);
  return i;
}

template<class T>
bool TSet<T>::verify(const char* id) const
{
  bool err = overflow_ || size_ < allocated_;
  if (err)
  {
    if (id)
      printf("%s ", id);
    printf("expected %d, got %s%d\n", allocated_, overflow_ ? ">" : "", size_);
  }
  return err;
}


template<class T>
void TSet<T>::sort(T** e, int n, T** temp)
{
  if (n == 1)
    return;
  if (n == 2)
  {
    if (*e[1] < *e[0])
    {
      *temp = e[0];
      e[0] = e[1];
      e[1] = *temp;
    }
    return;
  }

  // recursive n*log(n) sort
  int h = n >> 1;
  sort(e, h, temp);
  sort(e + h, n - h, temp);

  // merge
  int j;
  for (j = 0; j < h; ++j)
    temp[j] = e[j];
  int i = 0, m = 0;
  while (i < h && j < n)
    if (*temp[i] < *e[j])
      e[m++] = temp[i++];
    else
      e[m++] = e[j++];
  while (i < h)
    e[m++] = temp[i++];
}

template<class T>
void TSet<T>::sort()
{
  if (!sorted_ && size_ > 1)
  {
    T** temp = new T*[size_ >> 1];
    sort(list_, size_, temp);
    sorted_ = true;
    delete[] temp;
  }
}

template<class D, class R>
TMap<D,R>::TMap(int expected)
: TSet<D>(expected), map_(0)
{
  alloc(expected);
}

template<class D, class R>
TMap<D,R>::~TMap()
{
  alloc(0);
}

template<class D, class R>
void TMap<D,R>::alloc(int expected)
{
  TSet<D>::alloc(expected);
  delete [] map_;
  if (expected)
    map_ = new R[expected];
  else
    map_ = 0;
}

template<class D, class R>
int TMap<D,R>::append(const D& d, const R& r)
{
  int i = TSet<D>::append(d);
  if (i >= 0)
    map_[i] = r;
  return i;
}

template<class D, class R>
THist<D,R>::THist(int expected)
: TMap<D,R>(expected)
{
}

template<class D, class R>
int THist<D,R>::enter(const D& d, const R& r)
{
  int i = TSet<D>::lookup(d);
  if (i < 0)
    return append(d, r);
  R& x = range(i);
  x += r;
}
#endif

#endif
