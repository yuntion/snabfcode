#pragma once
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>
namespace snabf::pass4 {
struct Proposal {
  std::string fingerprint;
  int64_t target;
};
struct Evaluation {
  bool checked;
  double cost;
  std::string reason;
};
struct Round {
  Proposal proposal;
  Evaluation result;
  bool cached;
};
struct Decision {
  int64_t target;
  double cost;
  bool limit;
  std::vector<Round> trace;
  std::string winner;
};
// Callers give this search a verified fallback, or none. The closure owns the
// actual SSA, MILP and estimator checks; an unchecked state cannot become best.
Decision search(std::optional<Evaluation> baseline,
                const std::vector<Proposal> &, unsigned maxIters,
                const std::function<Evaluation(const Proposal &)> &, bool requireDown = false);
// Source-profile evaluations on one fixed proposal are distinct from the
// existing (N, level) enumeration. The screened initial profile counts as one.
struct CutFeedbackVisits {
  static constexpr unsigned limit = 4;
  std::string enter(const std::string &fingerprint);
  bool full() const { return seen.size() >= limit; }
private:
  std::set<std::string> seen;
};
} // namespace snabf::pass4
