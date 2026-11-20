#include "LayoutContract.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <set>
namespace snabf::pass4 {
const char *name(Packing p) {
  switch (p) {
  case Packing::PeriodicSingle:
    return "periodic_single";
  case Packing::SingleStrided:
    return "single_strided";
  case Packing::PackedMulti:
    return "packed_multi";
  }
  return "unknown";
}
std::string validate(const SlotLayoutDesc &x) {
  if (x.ring < 4 || x.ring > (1LL << 30) ||
      !std::has_single_bit(uint64_t(x.ring)) || x.slots != x.ring / 2 ||
      x.width <= 0 || x.width > x.slots)
    return "slot_capacity_exceeded";
  if (x.period <= 0 || x.period > x.slots || x.slots % x.period)
    return "period_mismatch";
  if (x.stride <= 0 || x.stride > x.period || x.period % x.stride)
    return "stride_unaligned";
  if (x.level < 0)
    return "level_insufficient";
  if (!std::isfinite(x.logScale) || x.logScale <= 0)
    return "scale_mismatch";
  if (x.offsets.empty() || x.active.empty() ||
      (x.packing != Packing::PackedMulti && x.offsets.size() != 1))
    return "invalid_lane_map";
  std::set<int64_t> used, lanes;
  for (auto lane : x.active) {
    if (lane < 0 || size_t(lane) >= x.offsets.size() ||
        !lanes.insert(lane).second)
      return "invalid_lane_map";
    auto start = x.offsets[lane];
    if (start < 0 || start >= x.period ||
        x.width - 1 > (x.period - 1 - start) / x.stride)
      return "slot_capacity_exceeded";
    for (int64_t j = 0; j < x.width; ++j)
      if (!used.insert(start + j * x.stride).second)
        return "lane_overlap";
  }
  if (!std::is_sorted(x.mask.begin(), x.mask.end()) ||
      std::adjacent_find(x.mask.begin(), x.mask.end()) != x.mask.end())
    return "mask_mismatch";
  for (auto i : x.mask)
    if (!used.contains(i))
      return "mask_mismatch";
  return {};
}
LayoutContract compare(const SlotLayoutDesc &a, const SlotLayoutDesc &b) {
  auto bad = [&](std::string r) {
    return LayoutContract{a, b, Compatibility::Incompatible, std::move(r)};
  };
  for (auto *x : {&a, &b})
    if (auto r = validate(*x); !r.empty())
      return bad(r);
  if (a.level != b.level)
    return bad("level_insufficient");
  if (a.logScale != b.logScale)
    return bad("scale_mismatch");
  if (a.period != b.period)
    return bad("period_mismatch");
  if (a.stride != b.stride)
    return bad("stride_unaligned");
  if (a.width != b.width || a.packing != b.packing || a.offsets != b.offsets ||
      a.active != b.active)
    return bad("lane_map_mismatch");
  if (a.mask != b.mask)
    return bad("mask_mismatch");
  return {a, b, a == b ? Compatibility::Exact : Compatibility::Adaptable, {}};
}
} // namespace snabf::pass4
