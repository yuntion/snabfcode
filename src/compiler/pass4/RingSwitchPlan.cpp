#include "RingSwitchPlan.h"
#include <algorithm>
#include <cmath>
namespace snabf::pass4 {
std::optional<CutLevels> commonCutLevels(const std::vector<CutLevels> &values) {
  if (values.empty()) return std::nullopt;
  CutLevels window = values.front();
  for (auto v : values) {
    if (v.required < 0 || v.current < 0) return std::nullopt;
    window.required = std::max(window.required, v.required);
    window.current = std::min(window.current, v.current);
  }
  if (window.required > window.current) return std::nullopt;
  return window;
}
double limbWork(int64_t n, int64_t l) {
  return n * std::log2(double(n)) * (1. + l);
}
double conversionWork(int64_t n, int64_t l, double price) {
  return price * limbWork(n, l) * (l + 2);
}
RingSwitchTransitionSpec transition(Action k, const SlotLayoutDesc &a,
                                    const SlotLayoutDesc &b, int64_t beta,
                                    double price) {
  auto fail = [&](std::string r) {
    return RingSwitchTransitionSpec{k, beta, false, 0, std::move(r)};
  };
  if (!std::isfinite(price) || price < 0)
    return fail("invalid_conversion_cost");
  if (k == Action::SplitRepack || beta > 1)
    return fail("split_repack_not_emittable");
  if (beta != 1)
    return fail("invalid_parallel_factor");
  if ((k == Action::Down && a.ring <= b.ring) ||
      (k == Action::Up && a.ring >= b.ring) ||
      (k == Action::Align && a.ring != b.ring))
    return fail("ring_direction");
  if (auto c = compare(a, b); c.status == Compatibility::Incompatible)
    return fail(c.reason);
  if (k != Action::Align && a.packing == Packing::PackedMulti)
    return fail("split_repack_not_emittable");
  if (k == Action::Up)
    return fail("ring_up_not_emittable");
  if (k == Action::Down && a.packing != Packing::PeriodicSingle)
    return fail("layout_transform_not_emittable");
  return {k,
          beta,
          true,
          k == Action::Align
              ? 0
              : conversionWork(std::max(a.ring, b.ring), a.level, price),
          {}};
}
} // namespace snabf::pass4
