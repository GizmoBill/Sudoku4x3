// COPYRIGHT (C) 2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

// ********************************************************
// *                                                      *
// *  Infrastructure for Using Multiple Parallel Threads  *
// *                                                      *
// ********************************************************

// A Rope is a set of one or more intertwined parallel execution threads that cooperate
// to compute a result. ("intertwined" is a rope metaphore, not a technical term). The
// computation is partitioned into N autonomous, independent steps, each given an iteration
// index from 0 to N-1.
//
// The client of a Rope provides a class as template parameter T, an object of type T,
// a member function pointer to an iteration function, N, and the number of threads to
// use C. The Rope then creates C-1 threads to add to the client's thread. The Rope
// calls the iteration function repeatedly for each thread, giving each call a unique
// iteration index, until all N steps are complete. The call order is unspecified and
// will vary from run to run. The iteration function must manage race conditions.
// Throws in the iteration function are caught and rethrown to the client thread.

#ifndef __rope__
#define __rope__

#include <thread>
#include <exception>
#include <vector>
#include <atomic>

template<class T>
class Rope
{
public:
  Rope(T* object, void (T::*memberFunction)(int, int))
    : object_(object), memberFunction_(memberFunction) {}

  void run(int iterations, int threadCount = 0);

private:
  T* object_;
  void (T::*memberFunction_)(int, int);

  std::atomic<int> index_;
  std::exception_ptr caughtException;
  void loop_(int iterations, int threadIndex);
};

template<class T>
void Rope<T>::run(int iterations, int threadCount)
{
  if (!(0 <= threadCount && threadCount <= 32))
    throw std::runtime_error("Bad threadCount");

  if (threadCount == 0)
    threadCount = std::thread::hardware_concurrency();

  if (threadCount == 1)
    for (int i = 0; i < iterations; ++i)
      (object_->*memberFunction_)(i, 0);
  else
  {
    std::vector<std::thread> threads_(threadCount - 1);
    index_ = 0;
    caughtException = std::exception_ptr();
    for (int i = 1; i < threadCount; ++i)
      threads_[i - 1] = std::thread(&Rope::loop_, this, iterations, i);
    loop_(iterations, 0);

    for (int i = 1; i < threadCount; ++i)
      threads_[i - 1].join();
    if (caughtException)
      std::rethrow_exception(caughtException);
  }
}

template<class T>
void Rope<T>::loop_(int iterations, int threadIndex)
{
  try
  {
    while (true)
    {
      int i = index_++;
      if (i < iterations)
        (object_->*memberFunction_)(i, threadIndex);
      else
        break;
    }
  }
  catch (...)
  {
    caughtException = std::current_exception();
  }
}

#endif
