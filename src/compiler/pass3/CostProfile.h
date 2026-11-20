#pragma once
#include "GadgetNoiseContract.h"
#include <compare>
#include <map>
#include <stdexcept>
#include <string_view>

namespace snabf::pass3 {
struct CostCall {
  int ell;
  Op op;
  std::vector<int> shifts;
  auto operator<=>(const CostCall &) const = default;
};
using PatternPrices =
    std::map<std::vector<int>, double>; // absolute group/site microseconds
struct MeasuredBasis {
  int n;
  std::vector<int64_t> q, p;
  std::map<CostCall, PatternPrices> calls;
};
class CostProfile {
public:
  static std::shared_ptr<const CostProfile>
  loadFile(std::string_view path, std::string_view expectedHardwareID);
  static std::shared_ptr<const CostProfile>
  parse(std::string_view json, std::string_view expectedHardwareID);
  const auto &bases() const { return bases_; }
  const auto &digest() const { return digest_; }
  const auto &hardwareID() const { return hardware_; }
  const auto &json() const { return json_; }

private:
  CostProfile() = default;
  std::vector<MeasuredBasis> bases_;
  std::string digest_, hardware_, json_;
};

class MissingCost : public std::runtime_error {
public:
  explicit MissingCost(const std::string &what)
      : std::runtime_error("cost_profile_missing_site: " + what) {}
};
struct PriceGroup {
  std::vector<int> edges;
  const PatternPrices *prices;
};
const MeasuredBasis &measuredBasis(const CostProfile &, int n,
                                   const Solution &);
std::vector<PriceGroup> priceGroups(const Graph &, const MeasuredBasis &);
struct SelectedPrice {
  std::vector<int> edges;
  double us;
};
// Reconstruct independently from the graph, physical basis and chosen Ks.
std::vector<SelectedPrice> replayPrices(const Graph &, const CostProfile &,
                                        const Solution &);
} // namespace snabf::pass3
