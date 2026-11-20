#include "DynamicRingPlanner.h"
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
namespace snabf::pass4 {
Decision search(std::optional<Evaluation> base,
                const std::vector<Proposal> &proposals, unsigned limit,
                const std::function<Evaluation(const Proposal &)> &eval) {
  if (!limit || limit > 16)
    throw std::invalid_argument("invalid_iteration_limit");
  Decision d{0, std::numeric_limits<double>::infinity(), false, {}, {}};
  if (base && base->checked && std::isfinite(base->cost) && base->cost >= 0)
    d.cost = base->cost;
  std::map<std::pair<int64_t, std::string>, Evaluation> cache;
  for (auto &p : proposals) {
    auto key = std::make_pair(p.target, p.fingerprint);
    auto it = cache.find(key);
    bool hit = it != cache.end();
    if (!hit) {
      if (cache.size() >= limit) {
        d.limit = true;
        break;
      }
      it = cache.emplace(key, eval(p)).first;
    }
    auto r = it->second;
    if (r.checked && (!std::isfinite(r.cost) || r.cost < 0))
      throw std::invalid_argument("invalid_evaluated_cost");
    if (r.checked && r.cost < d.cost) {
      d.cost = r.cost;
      d.target = p.target;
      d.winner = p.fingerprint;
    } else if (r.checked)
      r.reason = "no_net_benefit";
    d.trace.push_back({p, r, hit});
  }
  if (!std::isfinite(d.cost))
    throw std::runtime_error("no_verified_schedule");
  return d;
}
} // namespace snabf::pass4
