#include "Orchestrator.h"

#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace snabf {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

bool nonnegative(double value) {
  return std::isfinite(value) && value >= 0;
}

bool validBound(std::optional<double> value) {
  return !value || nonnegative(*value);
}

}  // namespace

void SNABFOrchestrator::addProfile(std::string id, Profile profile) {
  require(!id.empty() && !profile.keyId.empty() && !profile.layoutId.empty(),
          "profile, key, and layout IDs are required");
  require(profile.logN >= 2 && profile.logN <= 30 && !profile.q.empty(),
          "invalid ring dimension or empty Q chain");
  const auto modulus = std::uint64_t{2} << profile.logN;
  std::set<std::uint64_t> seen;
  for (const auto* basis : {&profile.q, &profile.p}) {
    for (auto limb : *basis) {
      require(limb > modulus && limb % modulus == 1,
              "modulus is not NTT-compatible");
      require(seen.insert(limb).second, "duplicate Q/P modulus");
    }
  }
  require(profiles_.emplace(std::move(id), std::move(profile)).second,
          "duplicate profile ID");
}

void SNABFOrchestrator::recordValue(std::string id, BudgetState state) {
  const auto found = profiles_.find(state.profileId);
  require(!id.empty() && found != profiles_.end(),
          "a value must reference a registered profile");
  require(state.level < found->second.q.size(), "invalid active RNS level");
  require(std::isfinite(state.log2Scale) &&
              std::isfinite(state.idealRange.lower) &&
              std::isfinite(state.idealRange.upper) &&
              state.idealRange.lower <= state.idealRange.upper,
          "invalid log scale or ideal range");
  require(nonnegative(state.approximationBound) &&
              validBound(state.canonicalBias) &&
              validBound(state.canonicalVarianceProxy) &&
              validBound(state.estimatedSecurityBits) &&
              validBound(state.validityFailure) &&
              (!state.validityFailure || *state.validityFailure < 1),
          "invalid budget bound");
  require(states_.emplace(std::move(id), std::move(state)).second,
          "duplicate value ID");
}

const Profile& SNABFOrchestrator::profile(const std::string& id) const {
  return profiles_.at(id);
}

const BudgetState& SNABFOrchestrator::state(const std::string& value) const {
  return states_.at(value);
}

void SNABFOrchestrator::recordLinearSite(std::string id, PackedLinearBudget state) {
  const auto it = profiles_.find(state.profileId);
  require(!id.empty() && it != profiles_.end(), "linear site needs a registered profile");
  const auto& p = it->second;
  require(state.level < p.q.size() && state.logSlots < p.logN &&
              state.maskScale == p.q[state.level], "invalid HEIR linear site budget");
  require(linearSites_.emplace(std::move(id), std::move(state)).second, "duplicate linear site");
}

const PackedLinearBudget& SNABFOrchestrator::linearSite(const std::string& id) const {
  return linearSites_.at(id);
}

}  // namespace snabf
