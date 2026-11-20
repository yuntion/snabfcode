#pragma once

#include <gmpxx.h>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace snabf::pass2 {

// Exact signed coefficients matter: rounding towards either infinity can
// understate |a+b| after a later subtraction. A scale or a probability is
// also a rational until the transcendental tail calculation.
using Exact = mpq_class;
Exact exactBinary(double finiteValue);
double upperDouble(const Exact &);
double lowerDouble(const Exact &);
Exact sqrtUpper(const Exact &);
Exact logUpper(const Exact &);
// Lattigo v6.2.0 non-modular rlwe.Scale: 128-bit big.Float, nearest-even.
Exact roundCKKSScale(const Exact &);

enum class NoiseSourceKind { FreshEncrypt, FreshEncode, ProvenIndependent };
enum class DependentKind { RescaleRound, GadgetTrunc, SharedEvkSwitch,
                           DownConvRound, BootstrapNoise, UnknownDependent };

struct LinearTerm {
  Exact signed_coeff, base_var_proxy;
  NoiseSourceKind kind;
  std::string formula_ref;
  // Empty => Holder singleton. Same nonempty family means a trusted proof
  // of JOINT independence on G, not pairwise independence or distinct IDs.
  std::string joint_proof;
};

struct DependentNoiseTerm {
  std::string site_id, shared_key_id;
  DependentKind kind;
  Exact sigma_proxy;
  std::string formula_ref;
};

using EventKey = std::pair<std::string, std::string>; // occurrence, reason

class AffineErrorForm {
public:
  void addBias(const Exact &bound);
  // A source is the SAME component of the SAME random draw. Even a group
  // previously summarized by its independent variance needs a stable ID:
  // an unlabelled "independent_var" scalar is not closed under fan-out.
  void addLinear(const std::string &source, const LinearTerm &);
  void addDependent(const DependentNoiseTerm &);
  void addEvent(const EventKey &, const Exact &failureBound);
  void markMissing(const std::string &site);
  void addScaled(const AffineErrorForm &, const Exact &signedGain);
  // For non-affine/full-output contracts only: retain proof obligations,
  // but do not pretend that the old component is the new random variable.
  void inheritObligations(const AffineErrorForm &);

  Exact varianceUpper() const;
  Exact deltaValid() const;
  double computeTotalVarianceProxy() const { return upperDouble(varianceUpper()); }
  double computeDeltaValidSum() const { return upperDouble(deltaValid()); }
  const auto &bias() const { return bias_; }
  const auto &linearTerms() const { return linear_; }
  const auto &dependentTerms() const { return dependent_; }
  const auto &events() const { return events_; }
  const auto &missingContracts() const { return missing_; }

private:
  Exact bias_ = 0;
  std::map<std::string, LinearTerm> linear_;
  std::map<std::string, DependentNoiseTerm> dependent_;
  std::map<EventKey, Exact> events_;
  std::set<std::string> missing_;
};

} // namespace snabf::pass2
