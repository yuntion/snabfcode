#include "SubgroupTower.h"
#include <algorithm>
#include <bit>
#include <numeric>
#include <set>
#include <stdexcept>

namespace snabf {
static unsigned groupOrder(unsigned logN) {
  if (logN < 2 || logN > 30)
    throw std::invalid_argument("unsupported ring dimension");
  return 1u << (logN - 1);
}
std::uint64_t rotationElement(unsigned d, unsigned logN) {
  const std::uint64_t modulus = 4ULL * groupOrder(logN);
  std::uint64_t key = 1, base = 5;
  for (; d; d >>= 1, base = base * base % modulus)
    if (d & 1)
      key = key * base % modulus;
  return key;
}
unsigned rotationDepth(const SubgroupRoute &r) {
  unsigned depth = 0;
  for (auto t : r.terms)
    depth = std::max(depth, unsigned(t.baby != 0) + unsigned(t.giant != 0));
  return depth;
}
std::vector<SubgroupRoute>
enumerateSubgroupRoutes(unsigned logN,
                        const std::vector<std::int64_t> &offsets) {
  const unsigned group = groupOrder(logN);
  unsigned generator = group;
  std::set<unsigned> support;
  for (auto x : offsets) {
    auto dx = x % std::int64_t(group);
    unsigned d = unsigned(dx < 0 ? dx + group : dx);
    support.insert(d);
    generator = std::gcd(generator, d);
  }
  const unsigned order = group / generator, rank = std::countr_zero(order);
  std::vector<SubgroupRoute> routes;
  for (unsigned s = 0; s <= rank; ++s) {
    unsigned stride = group >> s;
    SubgroupRoute r{s,  generator, stride, {},
                    {}, {},        {},     {{s, stride, 1u << s, order >> s}},
                    {}};
    std::set<unsigned> babies, giants;
    std::set<std::uint64_t> keys;
    for (auto d : support) {
      unsigned g = d % stride, b = d - g;
      r.terms.push_back({d, b, g});
      r.cosets[g].push_back(d);
      if (b)
        babies.insert(b);
      if (g)
        giants.insert(g);
    }
    for (auto b : babies)
      keys.insert(rotationElement(b, logN));
    for (auto g : giants)
      keys.insert(rotationElement(g, logN));
    r.babyRotations.assign(babies.begin(), babies.end());
    r.giantRotations.assign(giants.begin(), giants.end());
    r.galoisKeys.assign(keys.begin(), keys.end());
    routes.push_back(std::move(r));
  }
  return routes;
}
std::vector<SubgroupRoute>
executableRoutes(unsigned logN, const std::vector<std::int64_t> &offsets,
                 bool foldEquivalent) {
  std::vector<SubgroupRoute> out;
  for (auto &r : enumerateSubgroupRoutes(logN, offsets)) {
    // Baby offsets grow monotonically as the stride halves; equal programs
    // form contiguous runs. Equal key sets alone do not imply equal work.
    if (foldEquivalent && !out.empty() && out.back().terms == r.terms)
      out.back().splits.push_back(r.splits.front());
    else
      out.push_back(std::move(r));
  }
  return out;
}
} // namespace snabf
