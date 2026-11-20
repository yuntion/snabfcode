#pragma once
#include "GadgetNoiseContract.h"
namespace snabf::pass3 {
struct Violation {
  std::string reason;
  int level = -1;
  double deficit = 0;
};
Violation replay(const Graph &, const Options &, Solution &);
// Construct the same system used to encode polynomial coefficients: inputs,
// post-mask scales, each multiplication/rescale and both sides of every join.
std::vector<double> replayScales(const Graph &, const Solution &);
} // namespace snabf::pass3
