#include "RouteSelection.h"

#include <cmath>
#include <set>
#include <stdexcept>

#include "Highs.h"

namespace snabf {
namespace {

void check(HighsStatus status) {
  if (status != HighsStatus::kOk) {
    throw std::runtime_error("HiGHS rejected a subgroup routing constraint");
  }
}

bool isCost(double cost) { return std::isfinite(cost) && cost >= 0; }

HighsInt addBinary(Highs& solver, double cost) {
  if (!isCost(cost)) throw std::invalid_argument("invalid route cost");
  const auto column = solver.getNumCol();
  check(solver.addVar(0, 1));
  check(solver.changeColCost(column, cost));
  check(solver.changeColIntegrality(column, HighsVarType::kInteger));
  return column;
}

}  // namespace

RoutingPlan selectSubgroupRoutes(
    const std::vector<LinearRegion>& regions,
    const std::map<ScopedGaloisKey, std::uint64_t>& keyBytes,
    double costPerKeyByte, const std::vector<RouteDepthCut>& cuts) {
  RoutingPlan plan;
  if (!isCost(costPerKeyByte)) throw std::invalid_argument("invalid key cost");
  if (regions.empty()) return plan;

  Highs solver;
  check(solver.setOptionValue("output_flag", false));
  // HiGHS shares its scheduler across instances in one snabf-opt process.
  // Use the same setting as the Pass 3 MILP during mixed-ring replanning.
  check(solver.setOptionValue("threads", 1));
  check(solver.setOptionValue("mip_rel_gap", 0.0));
  check(solver.setOptionValue("mip_abs_gap", 0.0));
  std::vector<std::vector<HighsInt>> routeColumns;
  std::map<ScopedGaloisKey, HighsInt> keyColumns;
  for (const auto& region : regions) {
    if (region.keyScope.empty() || region.candidates.empty() ||
        !isCost(region.hoistedStepCost) || !isCost(region.giantStepCost)) {
      throw std::invalid_argument("invalid linear region inputs");
    }
    auto& choices = routeColumns.emplace_back();
    for (const auto& route : region.candidates) {
      const auto charge = operationCharge(region, route);
      const auto choice = addBinary(solver, charge.hoistedWork + charge.giantWork);
      choices.push_back(choice);
      std::set<std::uint64_t> unique;
      for (const auto galois : route.galoisKeys) {
        if (!unique.insert(galois).second || galois == 1) {
          throw std::invalid_argument("duplicate or identity Galois key");
        }
        const ScopedGaloisKey key{region.keyScope, galois};
        auto it = keyColumns.find(key);
        if (it == keyColumns.end()) {
          if (!keyBytes.contains(key) || keyBytes.at(key) == 0) {
            throw std::invalid_argument("missing key size for routing candidate");
          }
          it = keyColumns.emplace(key, addBinary(solver, costPerKeyByte * keyBytes.at(key))).first;
        }
        const HighsInt columns[] = {choice, it->second};
        const double weights[] = {1.0, -1.0};
        check(solver.addRow(-kHighsInf, 0, 2, columns, weights));
      }
    }
    const std::vector<double> weights(choices.size(), 1.0);
    check(solver.addRow(1, 1, static_cast<HighsInt>(choices.size()),
                        choices.data(), weights.data()));
  }

  for (const auto& cut : cuts) {
    std::vector<HighsInt> cols;
    std::vector<double> coeffs;
    for (auto i : cut.sites) for (std::size_t j=0; j<regions.at(i).candidates.size(); ++j) {
      cols.push_back(routeColumns.at(i).at(j));
      coeffs.push_back(rotationDepth(regions.at(i).candidates.at(j)));
    }
    check(solver.addRow(-kHighsInf,cut.limit,cols.size(),cols.data(),coeffs.data()));
  }
  check(solver.run());
  if (solver.getModelStatus() != HighsModelStatus::kOptimal) {
    throw std::runtime_error("no optimal subgroup routing solution");
  }
  const auto& values = solver.getSolution().col_value;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto& choices = routeColumns[i];
    const auto start = plan.selected.size();
    for (std::size_t j = 0; j < choices.size(); ++j) {
      if (values.at(choices[j]) <= 0.5) continue;
      plan.selected.push_back(j);
    }
    if (plan.selected.size() != start + 1) {
      throw std::runtime_error("invalid one-hot subgroup selection");
    }
  }

  return scoreSelection(regions, keyBytes, costPerKeyByte, plan.selected);
}

}  // namespace snabf
