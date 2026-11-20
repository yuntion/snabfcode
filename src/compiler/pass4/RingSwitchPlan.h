#pragma once
#include "LayoutContract.h"
namespace snabf::pass4 {
enum class Action { Down, Up, SplitRepack, Align };
struct RingSwitchTransitionSpec {
  Action kind;
  int64_t parallelFactor;
  bool emittable;
  double cost;
  std::string reason;
};
RingSwitchTransitionSpec transition(Action, const SlotLayoutDesc &,
                                    const SlotLayoutDesc &, int64_t beta,
                                    double price);
// Same units on both sides: ring-NTT limb work. Prices are not measured time.
double limbWork(int64_t ring, int64_t level);
double conversionWork(int64_t sourceRing, int64_t level, double price);
} // namespace snabf::pass4
