#include "pass3/ConcreteVerifier.h"
#include "pass3/CostProfile.h"
#include "pass3/ParameterMILP.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include <iostream>

using namespace snabf::pass3;
using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;
namespace {
void yes(bool b, const char *why) {
  if (!b)
    throw std::runtime_error(why);
}
template <class F> void rejects(F f, const char *reason) {
  try {
    f();
  } catch (const std::exception &e) {
    if (std::string(e.what()).find(reason) != std::string::npos)
      return;
    throw;
  }
  throw std::runtime_error(std::string("unexpected pass: ") + reason);
}
// Independent fixture writer. The data below are controlled *test prices*,
// not machine observations. An optional second mode consumes a real Go file.
std::string canonical(const Value &v) {
  auto str = [](std::string s) {
    return "s" + std::to_string(s.size()) + ":" + s;
  };
  if (auto o = v.getAsObject()) {
    std::map<std::string, std::string> m;
    for (auto &[k, x] : *o)
      m.emplace(k.str(), canonical(x));
    std::string s = "{";
    for (auto &[k, x] : m)
      s += str(k) + x;
    return s + "}";
  }
  if (auto a = v.getAsArray()) {
    std::string s = "[";
    for (auto &x : *a)
      s += canonical(x);
    return s + "]";
  }
  if (auto s = v.getAsString())
    return str(s->str());
  return "i" + std::to_string(*v.getAsInteger()) + ";";
}
std::string hash(const Value &v) {
  auto s = canonical(v);
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(s)), true);
}
std::string seal(Object o) {
  o["profile_sha256"] = hash(Value(Object(o)));
  return llvm::formatv("{0}", Value(std::move(o))).str();
}
Object hw() {
  return Object{{"os", "linux"},
                {"arch", "fixture"},
                {"cpu_model", "unit-test"},
                {"microcode", "test"},
                {"governor", "test"},
                {"affinity", "0"},
                {"id_scope", "machine-id"},
                {"machine_sha256", std::string(64, '0')}};
}
std::string hardware() { return hash(Value(hw())); }
template <class T> Array arr(const std::vector<T> &x) {
  Array a;
  for (auto i : x)
    a.push_back(int64_t(i));
  return a;
}
Value row(std::string op, std::vector<int> shifts, std::vector<int> pattern,
           int us) {
  std::string price = std::to_string(us) + ".00000";
  Array obs;
  for (int k : pattern)
    if (k)
      obs.push_back(Object{{"truncated_limbs_k", k},
                           {"full_blocks_skipped", k / 2},
                           {"measured_single_limb_ntts_saved", -3}});
  return Object{{"shape", 0},
                {"op", op},
                {"ell", 2},
                {"shifts", arr(shifts)},
                {"k", arr(pattern)},
                {"block_rows", Array{2, 1}},
                {"samples_us", Array{price, price}},
                {"holdout_us", Array{"400.00000"}},
                {"median_us", price},
                {"iqr_us", "0.00000"},
                {"observed_non_native", std::move(obs)}};
}
Object fixture(int n, const Solution &s, Array sites) {
  return Object{
      {"schema", "snabf-gadget-sites-v1"},
      {"abi", "direct-go-expanded-pure-rns-ntt-v1"},
      {"units", "us"},
      {"quantiles", "type7-from-raw-us"},
      {"created_utc", "2026-11-20T00:00:00Z"},
      {"bench_rev", std::string(40, 'c')},
      {"hardware", hw()},
      {"hardware_id", hardware()},
      {"toolchain",
       Object{{"go", "synthetic-fixture"},
              {"lattigo", "v6.2.0"},
              {"gomaxprocs", "1"},
              {"gc", "forced-before-each-call; disabled-inside-clock"},
              {"binary_sha256", std::string(64, 'a')},
              {"build", "synthetic-unit-test"}}},
      {"samples", 2},
      {"holdout", 1},
      {"warmup", 1},
      {"q_p_shape", Array{Object{{"n", n}, {"q", arr(s.q)}, {"p", arr(s.p)}}}},
      {"sites", std::move(sites)}};
}
Graph graph(int n) {
  Graph g{n,
          3,
          3,
          0,
          {{"x", Op::Source, 2, {}, {}, .125},
           {"mask", Op::Mask, 2, {0}, {.5}, .0625, 2},
           {"giant", Op::Giant, 2, {1}, {1}, .0625},
           {"drop", Op::Rescale, 1, {2}, {1}, .0625}}};
  g.edges[2].shift = 1;
  return g;
}
Options options() {
  Options o;
  o.alpha = 2;
  o.minScale = o.maxScale = 36;
  o.maxK = 2;
  return o;
}

void modelTest() {
  auto g = graph(1024);
  auto o = options();
  auto old = optimize(g, o);
  auto exact = old;
  exact.k[2] = 0;
  yes(old.k[2] == 1 && replay(g, o, exact).reason.empty(),
      "both k=0,1 must be feasible before pricing");
  auto native = row("giant", {1}, {0}, 100), cheap = row("giant", {1}, {1}, 20),
       dear = row("giant", {1}, {1}, 140), noise = row("giant", {1}, {2}, 1);
  auto profile = [&](Array a) {
    return CostProfile::parse(seal(fixture(g.n, old, std::move(a))),
                              hardware());
  };
  for (bool lose : {true, false}) {
    o.costProfile = profile(Array{native, lose ? dear : cheap, noise});
    auto src = optimize(g, o);
    auto boundary = optimizeAtBoundary(g, o, old);
    auto k = lose ? 0 : 1;
    auto us = lose ? 100. : 20.;
    for (auto s : {src, boundary}) {
      yes(s.q == old.q && s.p == old.p && s.k[2] == k && s.objective == us,
          "wrong measured decision or mixed units");
      yes(replay(g, o, s).reason.empty(), "noise replay");
      auto charge = replayPrices(g, *o.costProfile, s);
      yes(charge.size() == 1 && charge[0].edges == std::vector<int>{2} &&
              charge[0].us == us,
          "charge replay");
    }
  }
  o.costProfile = profile(Array{native, noise});
  auto s = optimize(g, o);
  yes(s.k[2] == 0 && s.objective == 100 &&
          optimizeAtBoundary(g, o, old).k[2] == 0,
      "missing k=1 became free");
  s.k[2] = 1;
  rejects([&] { replayPrices(g, *o.costProfile, s); },
          "cost_profile_missing_site");
  s.k[2] = 2;
  yes(!replay(g, o, s).reason.empty(), "cheap k=2 relaxed error constraints");
  auto bad = old;
  std::swap(bad.q[1], bad.q[2]);
  rejects([&] { optimizeAtBoundary(g, o, bad); }, "cost_profile_missing_site");
  auto other = g;
  other.edges[2].shift = 2;
  rejects([&] { optimize(other, o); }, "cost_profile_missing_site");
  rejects([&] { optimizeAtBoundary(other, o, old); },
          "cost_profile_missing_site");
  other = g;
  other.edges[2].op = Op::Down;
  rejects([&] { optimizeAtBoundary(other, o, old); }, "unmeasured downconv");
  rejects([&] { profile(Array{cheap}); }, "cost_profile_missing_site");
  // Two consumers, one emitted call. The measured mixed pattern can win;
  // the tempting but unmeasured Cartesian product [1,1] cannot be selected.
  auto group = g;
  group.edges.pop_back();
  group.edges[2].op = Op::Baby;
  group.edges[2].routeSite = 7;
  group.edges.push_back(group.edges[2]);
  group.edges[3].shift = 2;
  group.edges[3].id = "baby2";
  group.edges.push_back({"merge", Op::Add, 2, {2, 3}, {1, 1}, .125});
  group.edges.push_back({"rescale", Op::Rescale, 1, {4}, {1}, .125});
  group.sink = 5;
  o.costProfile = profile(Array{row("baby", {1, 2}, {0, 0}, 130),
                                row("baby", {1, 2}, {1, 0}, 60),
                                row("baby", {1, 2}, {0, 1}, 80)});
  for (auto a : {optimize(group, o), optimizeAtBoundary(group, o, old)}) {
    yes(a.k[2] == 1 && a.k[3] == 0 && a.objective == 60,
        "hoist overcharged or free combination");
    yes(replayPrices(group, *o.costProfile, a).size() == 1,
        "two charges for one hoist");
  }
  group.edges[3].routeSite = 8;
  rejects([&] { optimize(group, o); }, "cost_profile_missing_site");

  auto valid = fixture(g.n, old, Array{native, cheap});
  auto load = [&](Object v) {
    return CostProfile::parse(seal(std::move(v)), hardware());
  };
  auto invalid = valid;
  invalid["units"] = "ns";
  rejects([&] { load(invalid); }, "schema, units or ABI");
  invalid = valid;
  invalid.erase("samples");
  rejects([&] { load(invalid); }, "samples");
  invalid = valid;
  invalid["schema"] = "v2";
  rejects([&] { load(invalid); }, "schema");
  auto edit = [&](const char *key, Value value) {
    auto v = valid;
    (*(*v.getArray("sites"))[0].getAsObject())[key] = std::move(value);
    return v;
  };
  rejects([&] { load(edit("median_us", "0.00100")); }, "quantiles");
  rejects(
      [&] {
        load(edit("samples_us", Array{"-1.00000", "0.00000"}));
      },
      "unsigned microseconds");
  rejects(
      [&] {
        load(edit("samples_us", Array{"NaN", "0.00000"}));
      },
      "microsecond decimal");
  rejects([&] { load(edit("holdout_us", Array{})); }, "sample count");
  rejects(
      [&] {
        load(edit("block_rows", Array{1, 2}));
      },
      "original block rows");
  rejects([&] { load(edit("shape", 17)); }, "integer out of range");
  rejects([&] { CostProfile::parse(seal(valid), std::string(64, '0')); },
          "hardware_id");
  auto raw = seal(valid);
  auto where = raw.find("synthetic-fixture");
  yes(where != raw.npos, "fixture string");
  raw[where] = 'X';
  rejects([&] { CostProfile::parse(raw, hardware()); }, "profile_sha256");
  raw = seal(valid);
  raw.insert(1, "\"units\":\"us\",");
  rejects([&] { CostProfile::parse(raw, hardware()); }, "duplicate");
}
void goSmoke(const char *path, const char *id) {
  auto o = options();
  o.costProfile = CostProfile::loadFile(path, id);
  auto &b = o.costProfile->bases().at(0);
  auto g = graph(b.n);
  auto s = optimize(g, o);
  auto bound = optimizeAtBoundary(g, o, s);
  for (auto x : {s, bound}) {
    auto r = replayPrices(g, *o.costProfile, x);
    yes(r.size() == 1 && r[0].us == x.objective && x.q == b.q && x.p == b.p &&
            replay(g, o, x).reason.empty(),
        "real Go whole-site price not replayed");
  }
  std::cout << "Go profile=" << o.costProfile->digest() << " N=" << g.n
            << " giant k=" << s.k[2] << " objective_us=" << s.objective << "\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    modelTest();
    if (argc == 3)
      goSmoke(argv[1], argv[2]);
    else
      yes(argc == 1,
          "usage: Pass3CalibrationTest [default-go-profile hardware-id]");
    std::cout << "Pass3CalibrationTest passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
