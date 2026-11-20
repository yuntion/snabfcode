#ifndef SNABF_ORCHESTRATOR_H
#define SNABF_ORCHESTRATOR_H

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace snabf {

struct Interval {
  double lower;
  double upper;
};

struct Profile {
  std::string keyId;
  std::string layoutId;
  unsigned logN;
  std::vector<std::uint64_t> q;
  std::vector<std::uint64_t> p;
};

struct BudgetState {
  std::string profileId;
  unsigned level;  // Zero-based: active Q contains level + 1 primes.
  double log2Scale;
  Interval idealRange;
  double approximationBound;  // Decoded, deterministic approximation only.
  std::optional<double> canonicalBias;
  std::optional<double> canonicalVarianceProxy;
  std::optional<double> validityFailure;
  std::optional<double> estimatedSecurityBits;
};

// Available before polynomial/error analysis. maskScale is the plaintext
// diagonal encoding scale, not a guessed ciphertext scale or noise bound.
struct PackedLinearBudget {
  std::string profileId;
  unsigned level;
  unsigned logSlots;
  std::uint64_t maskScale;
};

class SNABFOrchestrator {
 public:
  void addProfile(std::string id, Profile profile);
  void recordValue(std::string id, BudgetState state);
  void recordLinearSite(std::string id, PackedLinearBudget state);
  const Profile& profile(const std::string& id) const;
  const BudgetState& state(const std::string& value) const;
  const PackedLinearBudget& linearSite(const std::string& id) const;

 private:
  std::map<std::string, Profile> profiles_;
  std::map<std::string, BudgetState> states_;
  std::map<std::string, PackedLinearBudget> linearSites_;
};

}  // namespace snabf

#endif
