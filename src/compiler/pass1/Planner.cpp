// HEIR region descriptors in JSON -> exact graph and independent tower plans.
#include "RouteSelection.h"
#include "Orchestrator.h"
#include "RingSwitch.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using namespace snabf;
using boost::property_tree::ptree;

template <class T> std::vector<T> list(const ptree& node, const char* name) {
  std::vector<T> result;
  for (const auto& entry : node.get_child(name)) result.push_back(entry.second.get_value<T>());
  return result;
}

template <class Range, class Emit> void array(const Range& range, Emit emit) {
  std::cout << '[';
  bool first = true;
  for (const auto& value : range) {
    if (!first) std::cout << ',';
    first = false;
    emit(value);
  }
  std::cout << ']';
}
template <class Range> void numbers(const Range& r) {
  array(r, [](auto value) { std::cout << value; });
}

void emitRoute(const SubgroupRoute& r) {
  std::cout << "{\"s\":" << r.towerIndex << ",\"subgroup_stride\":" << r.subgroupStride
            << ",\"baby_stride\":" << r.babyStride << ",\"terms\":";
  array(r.terms, [](auto t) { numbers(std::array{t.diagonal, t.baby, t.giant}); });
  std::cout << ",\"baby\":"; numbers(r.babyRotations);
  std::cout << ",\"giant\":"; numbers(r.giantRotations);
  std::cout << ",\"galois\":"; numbers(r.galoisKeys);
  std::cout << '}';
}

void emitPlan(const RoutingPlan& p, const std::vector<std::size_t>& selected) {
  std::cout << "{\"objective\":" << p.objective << ",\"key_bytes\":" << p.keyBytes
            << ",\"selected\":";
  numbers(selected);
  std::cout << ",\"keys\":";
  array(p.referenceCounts, [](const auto& x) {
    std::cout << "{\"scope\":" << std::quoted(x.first.first) << ",\"galois\":" << x.first.second
              << ",\"refs\":" << x.second << '}';
  });
  std::cout << '}';
}

void run() {
  ptree input;
  boost::property_tree::read_json(std::cin, input);
  const auto& p = input.get_child("profile");
  auto scope = input.get<std::string>("scope");  // SHA-256 of the concrete key family.
  if (scope.size() != 64 || scope.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument("expected a concrete key-scope digest");
  SNABFOrchestrator graph;
  graph.addProfile(scope, {scope, "heir-packed", p.get<unsigned>("logN"),
                           list<std::uint64_t>(p, "Q"), list<std::uint64_t>(p, "P")});
  const auto bytes = input.get<std::uint64_t>("bytes_per_key");
  const auto& costs = input.get_child("costs");
  const double eta = costs.get<double>("key_units_per_mib") / (1 << 20);
  std::vector<LinearRegion> regions;
  std::map<ScopedGaloisKey, std::uint64_t> sizes;
  for (const auto& entry : input.get_child("regions")) {
    const auto& v = entry.second;
    const auto id = v.get<std::string>("apply_id");
    graph.recordLinearSite(id, {scope, v.get<unsigned>("level"), v.get<unsigned>("log_slots"),
                                v.get<std::uint64_t>("mask_scale")});
    const auto& state = graph.linearSite(id);
    const auto& profile = graph.profile(state.profileId);
    auto level = state.level, logSlots = state.logSlots;
    auto offsets = list<std::int64_t>(v, "diagonals");
    if (offsets.empty()) throw std::invalid_argument("empty transform");
    for (auto d : offsets) if (d < 0 || d >= (1u << logSlots)) throw std::invalid_argument("non-canonical packed diagonal");
    auto& r = regions.emplace_back(LinearRegion{scope,
        (level+1) * costs.get<double>("hoisted_per_active_limb"),
        (level+1) * costs.get<double>("giant_per_active_limb"), {}});
    for (auto& candidate : executableRoutes(profile.logN, offsets)) {
      for (auto g : candidate.galoisKeys) sizes[{scope,g}] = bytes;
      r.candidates.push_back(std::move(candidate));
    }
  }
  const auto joint = selectSubgroupRoutes(regions, sizes, eta);
  std::vector<std::size_t> independent;
  auto fixed = regions;
  for (std::size_t i=0; i<regions.size(); ++i) {
    auto local = selectSubgroupRoutes({regions[i]}, sizes, eta);
    independent.push_back(local.selected.at(0));
    fixed[i].candidates = {regions[i].candidates.at(independent.back())};
  }
  // Rescore independent decisions on the SAME physical union as the joint arm.
  auto scored = selectSubgroupRoutes(fixed, sizes, eta);
  std::cout << std::setprecision(17) << "{\"candidates\":";
  array(regions, [](const auto& r) { array(r.candidates, emitRoute); });
  std::cout << ",\"independent\":"; emitPlan(scored, independent);
  std::cout << ",\"joint\":"; emitPlan(joint, joint.selected);
  std::cout << "}\n";
}
// Read-only historical JSON adapter; the decision is the native Eq. (3).
void bootReport() {
  ptree s; boost::property_tree::read_json(std::cin,s); BootContract c{};
  c.n=s.get<int64_t>("ring_n"); c.target=s.get<int64_t>("target_n"); c.level=s.get<int64_t>("level");
  c.maxLevel=s.get<int64_t>("max_level"); c.live=s.get<int64_t>("live_slots"); c.beta=s.get<int64_t>("small_ciphertexts");
  c.period=s.get<int64_t>("slot_period",0); c.layout=s.get<std::string>("layout_contract",""); c.layoutAdmitted=s.get<bool>("layout_verified");
  c.prepare=s.get<double>("layout_cost",0); c.keyVariance=s.get<double>("key_variance"); c.down=s.get<double>("downswitch_cost");
  c.up=s.get<double>("upswitch_cost"); c.weights=list<double>(s,"gadget_weights"); c.minSmallLevel=s.get<int64_t>("min_small_level",0);
  c.incomingVariance=s.get<double>("incoming_variance",0); if(auto v=s.get_optional<double>("variance_ceiling"))c.varianceCeiling=*v;
  for(auto& [_,row]:s.get_child("downstream_bins")) c.downstreamBins.emplace_back(row.get<double>("variance_cap"),row.get<double>("extra_cost"));
  for(auto& [key,row]:s.get_child("boot_costs")){
    auto i=key.find('@'); if(i==std::string::npos)throw std::invalid_argument("invalid ring@level");
    if(!c.bootCosts.emplace(std::make_pair(std::stoll(key.substr(0,i)),std::stoll(key.substr(i+1))),row.get_value<double>()).second)
      throw std::invalid_argument("duplicate bootstrap calibration");
  }
  auto d=evaluateRingSwitch(c);std::cout<<std::setprecision(17)<<"{\"enabled\":"<<std::boolalpha<<d.enabled<<",\"reason\":"<<std::quoted(d.reason)<<",\"beta_pack\":"<<c.beta;
  auto write=[](const char* key,std::optional<double> v){if(v)std::cout<<','<<std::quoted(key)<<':'<<*v;};
  write("rs_variance",d.rsVariance); write("downstream_penalty",d.downstreamPenalty);write("margin",d.margin);
  write("savings",d.savings);write("penalty",d.penalty);std::cout<<"}\n";
}
}  // namespace

int main(int argc,char** argv) {
  try { if(argc==1)run();else if(argc==2 && std::string(argv[1])=="--bootstrap")bootReport();else throw std::invalid_argument("unexpected arguments"); }
  catch (const std::exception& e) { std::cerr << "pass1: " << e.what() << '\n'; return 1; }
}
