#include "CostProfile.h"

namespace snabf::pass3 {
const MeasuredBasis &measuredBasis(const CostProfile &p, int n,
                                   const Solution &s) {
  for (auto &b : p.bases())
    if (b.n == n && b.q == s.q && b.p == s.p)
      return b;
  throw MissingCost("unmeasured ordered Q/P or ring");
}

std::vector<PriceGroup> priceGroups(const Graph &g, const MeasuredBasis &b) {
  std::vector<std::vector<int>> members;
  std::map<std::pair<int, int>, size_t> hoists;
  for (int i = 0; i < int(g.edges.size()); ++i) {
    const auto &e = g.edges[i];
    if (e.op == Op::Down)
      throw MissingCost("unmeasured downconv at " + e.id);
    if (!keyOp(e.op))
      continue;
    if ((e.ring ? e.ring : g.n) != b.n)
      throw MissingCost("ring at " + e.id);
    if (e.op != Op::Baby) {
      members.push_back({i});
      continue;
    }
    if (e.routeSite < 0 || e.inputs.size() != 1)
      throw MissingCost("unbound hoist at " + e.id);
    auto [it, first] =
        hoists.try_emplace({e.routeSite, e.inputs.front()}, members.size());
    if (first)
      members.emplace_back();
    members[it->second].push_back(i);
  }
  std::vector<PriceGroup> groups;
  for (auto &group : members) {
    const auto &first = g.edges[group.front()];
    CostCall key{first.level, first.op, {}};
    for (int i : group) {
      const auto &e = g.edges[i];
      if (e.level != key.ell)
        throw MissingCost("inconsistent hoist level at " + e.id);
      if (e.op != Op::Relin)
        key.shifts.push_back(e.shift);
    }
    auto it = b.calls.find(key);
    if (it == b.calls.end() ||
        !it->second.contains(std::vector<int>(group.size(), 0)))
      throw MissingCost("native " + first.id + " (" + name(key.op) +
                        ", L=" + std::to_string(key.ell) + ")");
    groups.push_back({std::move(group), &it->second});
  }
  return groups;
}

std::vector<SelectedPrice> replayPrices(const Graph &g, const CostProfile &p,
                                        const Solution &s) {
  std::vector<SelectedPrice> result;
  for (auto &group : priceGroups(g, measuredBasis(p, g.n, s))) {
    std::vector<int> ks;
    for (int i : group.edges)
      ks.push_back(s.k.at(i));
    auto it = group.prices->find(ks);
    if (it == group.prices->end())
      throw MissingCost("selected k pattern at " +
                        g.edges[group.edges.front()].id);
    result.push_back({std::move(group.edges), it->second});
  }
  return result;
}
} // namespace snabf::pass3
