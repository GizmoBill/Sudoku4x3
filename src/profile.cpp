// COPYRIGHT (C) 2022 BILL SILVER OF NOBLEBORO ME. I GRANT YOU A NONEXCLUSIVE
// LICENSE TO DO AS YOU PLEASE WITH THE FOLLOWING SOURCE CODE AT YOUR OWN RISK,
// AS LONG AS YOU INCLUDE THIS COPYRIGHT NOTICE IN COPIES OR DERIVED WORKS MADE
// FROM PORTIONS OR THE ENTIRETY OF THE CODE.

// **********************************************************
// *                                                        *
// *  Profile Code Using Simple Global Execution Time Tree  *
// *                                                        *
// **********************************************************

#include "profile.h"
#include "general.h"
#include <stdexcept>

namespace BillGeneral
{
  int ProfileTree::current_ = -1;

  std::vector<ProfileTree::Timepoint> ProfileTree::timeStack_;

  std::vector<ProfileTree::TreeNode> ProfileTree::tree_;

  bool ProfileTree::active_ = false;

  ProfileTree::TreeNode::TreeNode(int parent, const char* tag)
    : tag_(tag), elapsedSec_(0), iterations_(0),
	  parent_(parent), siblings_(-1), children_(-1)

  {
  }

  int ProfileTree::TreeNode::beginIteration(const char* tag)
  {
    int lastKid = -1;
    for (int kid = children_; kid >= 0; kid = tree_[kid].siblings_)
    {
      if (tree_[kid].tag_ == tag)
        return kid;
      lastKid = kid;
    }

    if (lastKid >= 0)
      tree_[lastKid].siblings_ = (int)tree_.size();
    else
      children_ = (int)tree_.size();

    tree_.push_back(TreeNode(index(), tag));

    return tree_.back().index();
  }

  double ProfileTree::TreeNode::childrenSec() const
  {
    double s = 0;
    for (int kid = children_; kid >= 0; kid = tree_[kid].siblings_)
      s += tree_[kid].elapsedSec_/* + tree_[kid].childrenSec()*/;
    return s;
  }

  std::string ProfileTree::TreeNode::toString(int indent) const
  {
    constexpr int tagWidth = 32;

    std::string s = strFormat("%*s%*s", indent, "", indent - tagWidth, tag_);
    if (iterations_ > 1)
      s += strFormat("%9llu * %13.4f -> ", iterations_, elapsedSec_ / iterations_ * 1.0e6);
    else
      s += "                             ";
    s += strFormat("%7.3f\n", elapsedSec_);

    for (int kid = children_; kid >= 0; kid = tree_[kid].siblings_)
      s += tree_[kid].toString(indent + 2);

    if (children_ >= 0)
      s += strFormat("%*s%*s                             %7.3f\n",
                     indent + 2, "", indent + 2 - tagWidth, "Overhead", overheadSec());

    return s;
  }

  void ProfileTree::clear()
  {
    current_ = -1;
    timeStack_.clear();
    tree_.clear();
    active_ = true;
  }

  void ProfileTree::start(const char* tag)
  {
    if (!isActive())
      return;

    timeStack_.push_back(std::chrono::high_resolution_clock::now());

    if (current_ >= 0)
      current_ = tree_[current_].beginIteration(tag);
    else
    {
      tree_.push_back(TreeNode(-1, tag));
      current_ = tree_.back().index();
    }
  }

  void ProfileTree::stop(uint64_t iterations)
  {
    if (!isActive())
      return;

    if (current_ >= 0)
    {
      current_ = tree_[current_].endIteration(std::chrono::high_resolution_clock::now() - timeStack_.back(),
                                              iterations);
      timeStack_.pop_back();
    }
    else
      throw std::runtime_error("Profile tree stop with no start");
  }

  std::string ProfileTree::toString()
  {
    std::string s = "Profile tree:\n";
    for (const TreeNode& node : tree_)
      if (node.isRoot())
        s += node.toString(0);
    return s;
  }

}
