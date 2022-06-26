// COPYRIGHT (C) 2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

// **********************************************************
// *                                                        *
// *  Profile Code Using Simple Global Execution Time Tree  *
// *                                                        *
// **********************************************************

#ifndef _profile_
#define _profile_

#include <chrono>
#include <vector>

namespace BillGeneral
{
  class ProfileTree
  {
    using Timespan  = std::chrono::high_resolution_clock::duration;
    using Timepoint = std::chrono::high_resolution_clock::time_point;

    class TreeNode
    {
    public:
      TreeNode(int parent, const char* tag);

      int index() { return (int)(this - tree_.data()); }

      int beginIteration(const char* tag);

      int  endIteration(Timespan nanosec, uint64_t iterations)
      {
        elapsedSec_ += nanosec.count() * 1.0e-9;
        iterations_ += iterations;
        return parent_;
      }

      bool isRoot() const { return parent_ < 0; }

      double childrenSec() const;
      double overheadSec() const { return elapsedSec_ - childrenSec(); }

      std::string toString(int indent) const;

    private:
      const char* tag_;
      double elapsedSec_;
      uint64_t iterations_;

      int parent_;
      int siblings_;
      int children_;
    };

    static int current_;
  
    static std::vector<Timepoint> timeStack_;

    static std::vector<TreeNode> tree_;

    static bool active_;

  public:
    // Clear the tree and make it active. Initial state is inactive.
    static void clear();

    // Get/set tree activation state. State won't change if any timers are running.
    // When the tree is inactive, calls to start and stop do nothing and have low
    // overhead.
    static void isActive(bool newActiveState) { if (!current_) active_ = newActiveState; }
    static bool isActive() { return active_; }

    // Start and stop a timer. The tag is identified by its machine address, not its contents,
    // and must have static or heap lifetime. Literal strings are typically used. These are
    // ignored if the tree is inactive.
    static void start(const char* tag);
    static void stop(uint64_t iterations = 1);

    // Nice formatted print of the tree
    static std::string toString();
  };

}

#endif
