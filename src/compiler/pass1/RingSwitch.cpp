#include "RingSwitch.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace snabf {
namespace {
bool power(std::int64_t n) {
  return n > 0 && std::has_single_bit(std::uint64_t(n));
}
double price(double x) {
  if (!std::isfinite(x) || x < 0)
    throw std::invalid_argument("invalid bootstrap cost/variance");
  return x;
}
} // namespace
BootDecision evaluateRingSwitch(const BootContract &c) {
  if (!power(c.n) || !power(c.target) || c.target < 4 || c.target >= c.n ||
      c.n > (1LL << 30))
    throw std::invalid_argument("invalid target subring");
  BootDecision d;
  auto reject = [&](const char *why) {
    d.reason = why;
    return d;
  };
  if (c.level < 1 || c.level > c.maxLevel)
    return reject("no_input_level_for_switch");
  if (c.minSmallLevel < 0)
    throw std::invalid_argument("negative required small-ring level");
  if (c.level - 1 < c.minSmallLevel)
    return reject("insufficient_downstream_level");
  const auto small = c.target / 2;
  if (!c.layoutAdmitted || !power(c.period) || c.period > c.n / 2 ||
      c.live < 1 || c.live > c.period || c.beta < 1 ||
      c.beta > c.n / c.target || c.live > small * c.beta)
    return reject("incompatible_packing");
  if (c.layout == "trace-periodic") {
    if (c.beta != 1 || small % c.period != 0)
      return reject("incompatible_packing");
  } else if (c.layout == "split-repack") {
    // Disjoint bands of width small; mask each before tracing, compensate
    // the normalized trace by period/small, and mask the embeds on return.
    if (c.period <= small || c.beta != (c.live + small - 1) / small ||
        !(price(c.prepare) > 0))
      return reject("incompatible_packing");
  } else
    return reject("incompatible_packing");
  if (c.weights.empty())
    throw std::invalid_argument("empty switching gadget");
  double squares = 0;
  for (auto w : c.weights)
    squares += price(w) * w;
  d.rsVariance = price(c.n * price(c.keyVariance) * squares / 12);
  d.outputVariance = price(price(c.incomingVariance) + *d.rsVariance);
  if (c.varianceCeiling && *d.outputVariance > price(*c.varianceCeiling))
    return reject("downstream_budget");
  double prev = -1, previousCost = -1;
  for (auto [cap, cost] : c.downstreamBins) {
    if (price(cap) <= prev || price(cost) < previousCost)
      throw std::invalid_argument("invalid variance bins");
    prev = cap;
    previousCost = cost;
    if (!d.downstreamPenalty && *d.rsVariance <= cap)
      d.downstreamPenalty = cost;
  }
  if (!d.downstreamPenalty)
    return reject("downstream_budget");
  auto boot = [&](auto n, auto level) {
    auto it = c.bootCosts.find({n, level});
    if (it == c.bootCosts.end())
      throw std::invalid_argument(
          "missing level-specific bootstrap calibration");
    return price(it->second);
  };
  d.savings = boot(c.n, c.level) - c.beta * boot(c.target, c.level - 1);
  d.penalty = price(price(c.down) + c.beta * price(c.up) + price(c.prepare) +
                    *d.downstreamPenalty);
  d.margin = *d.savings - *d.penalty;
  if (!std::isfinite(*d.margin))
    throw std::invalid_argument("non-finite bootstrap margin");
  d.enabled = *d.margin > 0;
  d.reason = d.enabled        ? "profitable"
             : *d.margin == 0 ? "tie"
                              : "negative_gain";
  return d;
}
} // namespace snabf
