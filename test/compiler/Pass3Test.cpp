#include "pass3/ConcreteVerifier.h"
#include "pass3/ParameterMILP.h"
#include "pass3/PrimeInstantiator.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
using namespace snabf::pass3;
void yes(bool b, const char *msg) {
  if (!b)
    throw std::runtime_error(msg);
}
int main() {
  yes(!prime(341550071728321ULL) && prime(65537) && !prime(1), "primality");
  std::set<int64_t> used;
  auto v = nearestPrime(1024, 34, used);
  auto w = nearestPrime(1024, 34, used);
  yes(v != w && prime(w) && w % 2048 == 1, "disjoint NTT primes");
  int64_t next = v + 2048;
  while (!prime(next))
    next += 2048;
  Graph g{1024, 1, 0, 0, {{"input", Op::Source, 0, {}, {}, .1}}};
  Options o;
  o.minScale = o.maxScale = 32;
  o.guardBits = std::log2(double(v) + (next - v) * .20) - 32;
  o.minBase = 25;
  auto s = optimize(g, o);
  yes(!s.cuts.empty() && s.q[0] > v,
      "concrete sub-bit miss did not cut and re-solve");
  std::cout << "first prime=" << v << " refined=" << s.q[0] << " "
            << s.cuts.front() << '\n';
  auto bad = s;
  bad.q = {v};
  bad.p = {nearestPrime(1024, 40, used)};
  yes(replay(g, o, bad).level == 0, "sub-bit precision regression");
  bad = s;
  bad.k[0] = 1;
  yes(!replay(g, o, bad).reason.empty(), "invalid non-key site");
  bad = s;
  bad.p = s.q;
  yes(!replay(g, o, bad).reason.empty(), "Q/P collision");
  bad = s;
  bad.q[0] += 2;
  yes(!replay(g, o, bad).reason.empty(), "invalid NTT prime");
  bad = s;
  bad.scale = 80;
  o.maxScale = 52;
  yes(!replay(g, o, bad).reason.empty(), "unrepresentable output scale");
  Graph t{1024,
          3,
          3,
          0,
          {{"x", Op::Source, 2, {}, {}, .125},
           {"mask", Op::Mask, 2, {0}, {.5}, .0625, 2},
           {"rotation", Op::Giant, 2, {1}, {1}, .0625},
           {"rescale", Op::Rescale, 1, {2}, {1}, .0625}}};
  o.minScale = o.maxScale = 36;
  o.guardBits = 1;
  o.minBase = 24;
  o.maxK = 2;
  auto loose = optimize(t, o);
  yes(loose.k[2] == 1, "expected a profitable feasible k=1");
  auto paired = loose;
  paired.k[2] = 0;
  yes(replay(t, o, paired).reason.empty(), "matched exact not feasible");
  auto invalid = loose;
  invalid.k[2] = 2;
  yes(!replay(t, o, invalid).reason.empty(),
      "full omitted product not charged");
  o.maxK = 0;
  auto zero = optimize(t, o);
  yes(zero.k[2] == 0, "exact ablation");

  for (bool hetero : {false,true}) for (int alpha : {1,2}) {
    auto a=o; a.alpha=alpha; a.heterogeneous=hetero; a.maxK=2;
    auto v=optimize(t,a);
    yes(v.p.size()==size_t(alpha) && replay(t,a,v).reason.empty(), "block solver concrete replay");
    yes(v.auxBits.size()==v.p.size(), "missing continuous special limbs");
    if (hetero && alpha==2) {
      auto tighter=a; tighter.primeWidthMin=28.; tighter.primeWidthMax=55.; tighter.auxMarginBits=5;
      yes(replay(t,tighter,v).blockFirst>=0, "stronger auxiliary margin not replayed");
      auto w=optimize(t,tighter);
      yes(replay(t,tighter,w).reason.empty() && std::all_of(w.auxBits.begin(),w.auxBits.end(),[](double x){return x>=28 && x<=55;}), "custom P bounds not respected");
    }
    if (hetero) {
      auto anchored = a; anchored.heirScaleAnchor = true;
      auto w = optimize(t, anchored);
      yes(replay(t, anchored, w).reason.empty(), "HEIR-emittable search failed");
      std::set<int64_t> seen(w.q.begin(), w.q.end());
      seen.insert(w.p.begin(), w.p.end());
      w.q.back() = nearestPrime(t.n, w.scale - 2., seen);
      yes(replay(t, anchored, w).reason == "rescale prime outside HEIR integer scale anchor", "stale HEIR scale bin passed");
    }
    std::cout << "alpha=" << alpha << " heterogeneous=" << hetero << " chosen k=" << v.k[2]
              << " physical_log_scale=" << v.physical.back() << '\n';
    yes(optimizeAtBoundary(t,a,v).k[2]==v.k[2], "block selection changed on same physical graph");
    if (alpha!=2) continue;
    // Zeroing just q0 leaves the entire first CRT digit active for ordinary EVK noise.
    auto err=[&](int k){return switchKeyError(v.q,v.p,2,k,t.n);};
    yes(err(0)==err(1) && err(1)>err(2), "partial block incorrectly priced as an absent key digit");
    long double P=specialModulus(v.p);
    yes(std::abs(err(2)/(10.L*t.n*t.n*(v.q[2]+1.L)/P)-1)<1e-12, "wrong P denominator");
    auto corrupt=v; std::set<int64_t> occupied(v.q.begin(),v.q.end());
    corrupt.p={nearestPrime(t.n,24,occupied),nearestPrime(t.n,24,occupied)};
    yes(replay(t,a,corrupt).blockFirst==0, "concrete block capacity not replayed");
    auto wrong=t;
    wrong.edges.push_back({"other",Op::Coefficient,2,{0},{1},.125});
    wrong.edges.push_back({"unaligned",Op::Add,2,{1,4},{1,1},.25});
    wrong.sink=5; corrupt=v; corrupt.k.resize(6);
    bool rejected=false;
    try { rejected=!replay(wrong,a,corrupt).reason.empty(); }
    catch (const std::runtime_error &e) { rejected=std::string(e.what()).find("unaligned physical add")!=std::string::npos; }
    yes(rejected, "stale join physical scale passed");
  }
}
