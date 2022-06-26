// COPYRIGHT (C) 2005-2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

#include "timer.h"

namespace BillGeneral {

// ***********
// *         *
// *  Timer  *
// *         *
// ***********

// A simple timer class using the portable high res clock in chrono.

double Timer::stop()
{
  if (running())
  {
    Clock::duration t = Clock::now() - t0_;
    running_ = false;
    elapsed_ += t;
    if (iterations_ == 0)
      min_ = max_ = t;
    else
    {
      min_ = std::min(min_, t);
      max_ = std::max(max_, t);
    }
    ++iterations_;
  }
  return elapsedSeconds();
}

void Timer::clear()
{
  running_ = false;
  iterations_ = 0;
  elapsed_ = Clock::duration::zero();
}

double Timer::elapsedSeconds() const
{
  Clock::duration t = elapsed_;
  if (running())
    t += Clock::now() - t0_;
  return (double)t.count() * clockPeriod;
}


double Timer::meanSeconds() const
{
  if (iterations() == 0)
    return 0;
  return elapsedSeconds() / iterations();
}

}
