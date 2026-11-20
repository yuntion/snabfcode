#include "Orchestrator.h"
#include "RouteSelection.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename F>
void expectRejected(F operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("accepted invalid budget state or profile");
}

void testBudget() {
  snabf::SNABFOrchestrator planner;
  const snabf::Profile profile{"key0", "cyclic8", 4, {97, 193}, {257}};
  const snabf::BudgetState input{"source", 1, 30, {-1, 1}, 0, {}, {}, {}, {}};
  planner.addProfile("source", profile);
  planner.recordValue("arg0", input);
  check(planner.state("arg0").level == 1 && planner.profile("source").q.size() == 2,
        "incorrect level accounting");
  check(!planner.state("arg0").estimatedSecurityBits, "unknown is not 128 bits");
  planner.recordLinearSite("apply0", {"source", 1, 3, 193});
  check(planner.linearSite("apply0").maskScale == 193, "lost HEIR mask scale");
  expectRejected([&] { planner.recordLinearSite("bad_level", {"source", 2, 3, 193}); });
  expectRejected([&] { planner.recordLinearSite("bad_slots", {"source", 1, 4, 193}); });
  expectRejected([&] { planner.recordLinearSite("bad_scale", {"source", 1, 3, 97}); });
  expectRejected([&] { planner.recordValue("arg0", input); });
  auto invalid = input;
  invalid.level = 2;
  expectRejected([&] { planner.recordValue("bad", invalid); });
  invalid = input;
  invalid.canonicalVarianceProxy = -1;
  expectRejected([&] { planner.recordValue("bad", invalid); });
  auto overlapping = profile;
  overlapping.p = {97};
  expectRejected([&] { planner.addProfile("bad", overlapping); });
}

void testRoutes() {
  constexpr unsigned slots = 16;
  const auto routes = snabf::enumerateSubgroupRoutes(5, {0, 2, -2, 8, 18});
  check(routes.size() == 4 && routes[1].subgroupStride == 2,
        "wrong minimal subgroup/order for sparse offsets");
  check(routes[2].babyStride == 4 && routes[2].terms.size() == 4,
        "wrong intermediate subgroup or normalization");
  check(routes[2].babyRotations == std::vector<unsigned>{8, 12} &&
            routes[2].giantRotations == std::vector<unsigned>{2} &&
            routes[2].galoisKeys == std::vector<std::uint64_t>{17, 25, 33},
        "wrong executed rotations or key union");

  // Nonconstant diagonals expose the sign of the required inverse rotation.
  for (const auto& route : routes) {
    for (unsigned i = 0; i < slots; ++i) {
      double direct = 0;
      double routed = 0;
      for (const auto [d, b, g] : route.terms) {
        const auto diagonal = [d](unsigned j) { return 2 * d + 3 * j + 1; };
        const auto x = [](unsigned j) { return 0.125 * j - 1; };
        direct += diagonal(i) * x((i + d) % slots);
        const auto rotatedIndex = (i + g) % slots;
        routed += diagonal((rotatedIndex + slots - g) % slots) *
                  x((rotatedIndex + b) % slots);
      }
      check(std::abs(direct - routed) < 1e-12, "invalid coset factorization");
    }
  }
  const auto identity = snabf::enumerateSubgroupRoutes(5, {0, 16, -16});
  check(identity.size() == 1 && identity[0].galoisKeys.empty(),
        "identity rotation should require no key");
}

void testSharedKeySelection() {
  constexpr double eta = 0.02;  // A 100-byte test cost entry contributes 2.
  snabf::LinearRegion a{"shared", 2, 1,
                        snabf::enumerateSubgroupRoutes(5, {0, 1, 2, 3, 4})};
  snabf::LinearRegion b{"shared", 2, 1,
                        snabf::enumerateSubgroupRoutes(5, {0, 3})};
  std::map<snabf::ScopedGaloisKey, std::uint64_t> keyCosts;
  for (const auto* region : {&a, &b}) {
    for (const auto& route : region->candidates) {
      for (const auto g : route.galoisKeys) {
        keyCosts[{"shared", g}] = 100;
        keyCosts[{"other", g}] = 100;
      }
    }
  }
  const auto local = snabf::selectSubgroupRoutes({a}, keyCosts, eta);
  check(local.keyBytes == 300 && local.objective == 11, "wrong isolated route");
  const auto joint = snabf::selectSubgroupRoutes({a, b}, keyCosts, eta);
  check(a.candidates.at(joint.selected.at(0)).galoisKeys.size() == 4 &&
            joint.keyBytes == 400 && joint.referenceCounts.at({"shared", 61}) == 2 &&
            joint.objective == 13,
        "global key reuse should change the first region's chosen route");
  b.keyScope = "other";
  const auto unshared = snabf::selectSubgroupRoutes({a, b}, keyCosts, eta);
  check(unshared.keyBytes == 400 && unshared.objective == 14 &&
            a.candidates.at(unshared.selected.at(0)).galoisKeys.size() == 3,
        "key sharing across profiles must not change the chosen route");
}

}  // namespace

int main() {
  try {
    testBudget();
    testRoutes();
    testSharedKeySelection();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
