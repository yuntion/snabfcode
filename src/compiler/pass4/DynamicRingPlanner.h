#pragma once
#include <functional>
#include <optional>
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
                const std::function<Evaluation(const Proposal &)> &);
} // namespace snabf::pass4
