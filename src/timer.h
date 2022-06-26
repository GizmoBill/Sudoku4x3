// COPYRIGHT (C) 2005-2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

#include <algorithm>
#include <chrono>

namespace BillGeneral {

// ***********
// *         *
// *  Timer  *
// *         *
// ***********

// A simple timer class using the portable high res clock in chrono.
class Timer
{
public:
  using Clock = std::chrono::high_resolution_clock;
  static constexpr double clockPeriod =
    (double)Clock::duration::period::num / Clock::duration::period::den;

  Timer(bool running = false)
  {
    if (running)
      start();
  }

  bool running() const { return running_; }

  void start()
  {
    if (running())
      stop();
    t0_ = Clock::now();
    running_ = true;
  }

  double stop();

  void clear();

  double elapsedSeconds() const;

  int iterations() const
  {
    return iterations_;
  }

  double minSeconds() const { return min_.count() * clockPeriod; }
  double maxSeconds() const { return max_.count() * clockPeriod; }

  double meanSeconds() const;

private:
  bool running_ = false;
  Clock::time_point t0_;
  int iterations_ = 0;
  Clock::duration elapsed_ = Clock::duration::zero();
  Clock::duration min_ = Clock::duration::zero();
  Clock::duration max_ = Clock::duration::zero();
};

}
