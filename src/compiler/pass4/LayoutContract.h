#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace snabf::pass4 {
enum class Packing { PeriodicSingle, SingleStrided, PackedMulti };
struct SlotLayoutDesc {
  int64_t ring, slots, width, stride, period, level;
  double logScale;
  Packing packing;
  // Within one encoded period: each active lane covers width logical values.
  std::vector<int64_t> offsets;
  std::vector<int64_t> active;
  // Padding or an application mask; empty = all mapped values are live.
  std::vector<int64_t> mask;
  bool operator==(const SlotLayoutDesc &) const = default;
};
enum class Compatibility { Exact, Adaptable, Incompatible };
struct LayoutContract {
  SlotLayoutDesc provided, required;
  Compatibility status;
  std::string reason;
};
std::string validate(const SlotLayoutDesc &);
LayoutContract compare(const SlotLayoutDesc &, const SlotLayoutDesc &);
const char *name(Packing);
} // namespace snabf::pass4
