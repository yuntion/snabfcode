#include "pass3/ConcreteVerifier.h"
#include "pass3/ParameterMILP.h"
#include "pass3/PrimeInstantiator.h"
#include <cmath>
#include <iostream>
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
}
