#include "RouteSelection.h"
#include "pass1/RingSwitch.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

using namespace snabf;
namespace {
void gate(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
void tower() {
  // Independent finite-group oracle over every subset of Z_8. In
  // particular the empty transform and the identity have different supports.
  for (unsigned mask = 0; mask < 256; ++mask) {
    std::vector<int64_t> ds;
    std::set<unsigned> H{0};
    for (unsigned d = 0; d < 8; ++d)
      if (mask >> d & 1) {
        ds.push_back(d);
        ds.push_back(int(d) - 8);
        auto old = H;
        for (auto h : old)
          for (unsigned k = 0; k < 8; ++k)
            H.insert((h + k * d) % 8);
      }
    unsigned seen = 0;
    for (const auto &r : executableRoutes(4, ds)) {
      for (auto s : r.splits) {
        gate(s.rung == seen++ && s.stride * s.order == 8 &&
                 s.order * s.index == H.size(),
             "missing/wrong tower rung");
        std::set<unsigned> support;
        std::set<uint64_t> keys;
        for (auto [d, b, g] : r.terms) {
          gate((b + g) % 8 == d && b % s.stride == 0 && g < s.stride,
               "invalid coset");
          for (auto step : {b, g})
            if (step) {
              uint64_t k = 1;
              for (unsigned j = 0; j < step; ++j)
                k = k * 5 % 32;
              keys.insert(k);
            }
          support.insert(d);
        }
        std::set<unsigned> partition;
        for (auto &[g, terms] : r.cosets)
          for (auto d : terms)
            gate(d % s.stride == g && partition.insert(d).second,
                 "duplicate/wrong coset member");
        gate(partition == support &&
                 std::equal(keys.begin(), keys.end(), r.galoisKeys.begin(),
                            r.galoisKeys.end()),
             "wrong keys/partition");
        for (unsigned d = 0; d < 8; ++d)
          gate(bool(mask >> d & 1) == bool(support.count(d)),
               "support changed");
      }
    }
    gate((1u << (seen - 1)) == H.size(), "incomplete tower");
  }
}
void keys() {
  LinearRegion a{"QP-A", 5, 8, executableRoutes(14, {0, 7, 9})};
  LinearRegion b{"QP-A", 1.875, 3, executableRoutes(14, {0, 6, 8})};
  KeySizes sizes;
  for (auto r : {a, b})
    for (auto &c : r.candidates)
      for (auto g : c.galoisKeys)
        sizes[{r.keyScope, g}] = 10486624;
  double eta = 1. / 1048576.;
  auto best = selectSubgroupRoutes({a, b}, sizes, eta),
       local = selectSubgroupRoutes({a}, sizes, eta);
  gate(best.selected[0] != local.selected[0] &&
           best.referenceCounts.size() == 3,
       "expected cross-layer feedback");
  gate(best.charges[0].amortizedBytes + best.charges[1].amortizedBytes ==
           double(best.keyBytes),
       "shares do not sum to union");
  double brute = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < a.candidates.size(); ++i)
    for (size_t j = 0; j < b.candidates.size(); ++j) {
      auto &x = a.candidates[i];
      auto &y = b.candidates[j];
      std::set<uint64_t> u(x.galoisKeys.begin(), x.galoisKeys.end());
      u.insert(y.galoisKeys.begin(), y.galoisKeys.end());
      brute = std::min(
          brute, 5 * x.babyRotations.size() + 8 * x.giantRotations.size() +
                     1.875 * y.babyRotations.size() +
                     3 * y.giantRotations.size() + eta * u.size() * 10486624.);
    }
  gate(best.objective == brute, "MILP differs from exhaustive physical union");
  a.densePricing = true;
  a.candidates = executableRoutes(14, {0, 7, 9}, false);
  auto p = selectSubgroupRoutes({a}, sizes, eta);
  auto split = a.candidates[p.selected[0]].splits[0];
  gate(p.charges[0].hoistedWork == 5 * (split.order - 1) &&
           p.charges[0].giantWork == 8 * (split.index - 1),
       "literal dense cost differs from Eq.2");
}
void boot() {
  BootContract a{32,
                 16,
                 3,
                 4,
                 8,
                 8,
                 1,
                 "trace-periodic",
                 true,
                 0.375,
                 10,
                 20,
                 0,
                 {1, 2},
                 {{5, 5}, {20, 15}},
                 {{{32, 3}, 80}, {{16, 2}, 40}}};
  auto test = [](auto c, std::string why) {
    auto d = evaluateRingSwitch(c);
    gate(d.reason == why && d.enabled == (why == "profitable"),
         "wrong ring-switch outcome");
    return d;
  };
  auto yes = test(a, "profitable");
  gate(yes.margin == 5 && yes.rsVariance == 5 && yes.downstreamPenalty == 5,
       "wrong Eq.3 factors");
  auto c = a;
  c.down = 15;
  test(c, "tie");
  c.down = 16;
  test(c, "negative_gain");
  c = a;
  c.keyVariance = 0.75;
  auto nextBin = test(c, "negative_gain");
  gate(nextBin.rsVariance == 10 && nextBin.downstreamPenalty == 15 &&
           nextBin.margin == -5,
       "wrong downstream variance bin or margin");
  c.keyVariance = 2;
  test(c, "downstream_budget");
  c = a;
  c.level = 0;
  test(c, "no_input_level_for_switch");
  c = a;
  c.minSmallLevel = 3;
  test(c, "insufficient_downstream_level");
  c = a;
  c.layoutAdmitted = false;
  test(c, "incompatible_packing");
  c = a;
  c.live = 9;
  test(c, "incompatible_packing");
  c = a;
  c.beta = 3;
  test(c, "incompatible_packing");
  c = a;
  c.period = 16;
  test(c, "incompatible_packing");
  c = a;
  c.incomingVariance = 2;
  c.varianceCeiling = 6;
  test(c, "downstream_budget");
  c.varianceCeiling = 7;
  test(c, "profitable");
  c = a;
  c.period = 16;
  c.live = 16;
  c.beta = 2;
  c.layout = "split-repack";
  c.prepare = 4;
  gate(test(c, "negative_gain").margin == -59,
       "missing split-repack charge for an unprofitable proposal");
  c.bootCosts[{32, 3}] = 150;
  gate(test(c, "profitable").margin == 11,
       "lost multiplicity/preparation charge");
  c.prepare = 0;
  test(c, "incompatible_packing");
}
} // namespace
int main() {
  try {
    tower();
    keys();
    boot();
    std::cout << "256 subgroup-support oracles, physical union, dense formula "
                 "and ring-switch boundary cases passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
