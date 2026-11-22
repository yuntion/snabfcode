#include "Pass3SlotBounds.h"
#include <array>
#include <limits>
#include <map>
#include <stdexcept>

using snabf::ir::Pass3SlotBounds;
using snabf::ir::ContractError;
using namespace snabf::pass3;

namespace {
void expect(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
template <class F> void mustReject(F f) {
  try { f(); } catch (const ContractError &) { return; }
  throw std::runtime_error("missing co-slot proof accepted");
}

struct Probe {
  size_t period;
  Pass3SlotBounds bounds;
  std::vector<Edge> edges;
  explicit Probe(size_t n) : period(n), bounds(n) {}
  int emit(Op op, std::vector<int> args, Pass3SlotBounds::Operands spec = {},
           double weight = 1, double source = 0) {
    int i = int(edges.size());
    Edge e{"test", op, 0, args, std::vector<double>(args.size(), weight), source};
    e.magnitude = bounds.append(e, spec);
    edges.push_back(std::move(e));
    return i;
  }
  int mask(int a, const std::vector<double> &v, int giant = 0) {
    double gain = 0;
    for (double w : v) gain = std::max(gain, std::abs(w));
    return emit(Op::Mask, {a}, {.shift = giant, .row = v}, gain);
  }
  double peak(int a) const { return edges.at(a).magnitude; }
  // A following valid diagonal that keeps only one slot observes the actual
  // parent, not just the scalar peak written to the edge record.
  double slot(int a, size_t i) {
    std::vector<double> probe(period); probe.at(i) = 1;
    return peak(mask(a, probe));
  }
};

void counterexample() {
  Probe p(64);
  int x = p.emit(Op::Source, {}, {}, 1, .5);
  std::vector<int> terms;
  double sumMaxima = 0;
  // 33 disjoint maxima suffice for a row L1 of one and a diagonal-max sum >32.
  for (int d = 0; d < 33; ++d) {
    std::vector<double> diagonal(64);
    diagonal[d] = d == 32 ? -775./1024 : 1.;
    int baby = d ? p.emit(Op::Baby, {x}, {.shift = d}) : x;
    int term = p.mask(baby, diagonal);
    sumMaxima += p.edges[term].gains[0];
    terms.push_back(term);
  }
  int y = p.emit(Op::Add, terms);
  expect(sumMaxima == 32.7568359375 && .5 * sumMaxima == 16.37841796875, "old guard fixture");
  expect(p.peak(y) == .5 && p.peak(y)/p.peak(x) == 1., "co-slot maximum of row sums");
  for (int i = 0; i < 64; ++i)
    expect(p.slot(y,i) == (i < 32 ? .5 : i == 32 ? .5*775/1024 : 0.), "disjoint support lost");
}

void bsgs() {
  constexpr int n = 8;
  Probe p(n);
  int x = p.emit(Op::Source, {}, {}, 1, 1);
  std::vector<double> input{1, 4, 0, 2, 7, 1, 3, 6};
  int a = p.mask(x, input);
  int baby = p.emit(Op::Baby, {a}, {.shift = 4});
  // Physical group 16, encoded period 8, tower stride 4. Two different
  // nonzero giants, a common baby, wraparound, signed and zero weights.
  std::map<int, std::vector<double>> rows{
    {0, {0,2,1,0,1,0,0,1}}, {1, {1,0,0,-3,0,1,2,0}}, {3, {0,1,0,2,1,0,0,4}},
    {4, {2,0,0,1,1,0,2,0}}, {5, {0,-1,8,0,0,2,0,1}}, {7, {3,0,1,1,0,0,2,0}}};
  std::map<int, std::vector<int>> groups;
  std::array<double,n> logical{};
  for (const auto &[d, row] : rows) {
    int g = d % 4, b = d - g;
    p.bounds.checkTerm(d,d,b,g,16);
    groups[g].push_back(p.mask(b ? baby : a, row, g));
    for (int i = 0; i < n; ++i) logical[i] += std::abs(row[i]) * input[(i+d)%n];
  }
  std::vector<int> rotated;
  for (auto &[g, terms] : groups) {
    int partial = p.emit(Op::Add, terms);
    for (int i = 0; i < n; ++i) {
      double expected = 0;
      for (const auto &[d, row] : rows) if (d%4 == g)
        expected += std::abs(row[(i+n-g)%n]) * input[(i+d-g)%n];
      expect(p.slot(partial,i) == expected, "incorrect pre-giant coset bounds");
    }
    rotated.push_back(g ? p.emit(Op::Giant, {partial}, {.shift = g}) : partial);
  }
  int y = p.emit(Op::Rescale, {p.emit(Op::Add, rotated)});
  int product = p.emit(Op::Multiply, {a, y});
  int residual = p.emit(Op::Add, {a, y});
  for (int i = 0; i < n; ++i) {
    expect(p.slot(y,i) == logical[i], "BSGS differs from unsplit row-L1 oracle");
    expect(p.slot(product,i) == input[i]*logical[i], "multiply ignored a parent slot");
    expect(p.slot(residual,i) == input[i]+logical[i], "residual ignored a parent slot");
  }
  // All signs on the admissible input box: the oracle is an enclosure even
  // when a different sign vector maximizes each *signed* row.
  for (int signs = 0; signs < 1<<n; ++signs) for (int i = 0; i < n; ++i) {
    double actual = 0;
    for (const auto &[d,row] : rows) {
      int j = (i+d)%n; actual += row[i]*input[j]*((signs>>j)&1 ? -1 : 1);
    }
    expect(std::abs(actual) <= logical[i], "signed row escapes bound");
  }
  mustReject([&]{ p.bounds.checkTerm(5,7,4,3,16); }); // wrong table row
  mustReject([&]{ p.bounds.checkTerm(5,5,12,1,16); }); // matches mod 8, NOT mod group
  mustReject([&]{ p.mask(x,{1.,2.}); });
  mustReject([&]{ p.emit(Op::Down,{a}); });
}

void roundingAndAffineParents() {
  Probe p(4);
  double u = std::nextafter(1.,2.);
  int a = p.emit(Op::Source, {}, {}, 1, u);
  int sq = p.emit(Op::Multiply, {a,a});
  expect(p.peak(sq) == std::nextafter(std::nextafter(u,2.),2.), "product rounded inward");
  int constant = p.emit(Op::AddPlain, {a}, {.absConstant = std::ldexp(1.,-54)});
  expect(p.peak(constant) == std::nextafter(u,2.), "sum rounded inward");
  int denorm = p.emit(Op::Source, {}, {}, 1, std::numeric_limits<double>::denorm_min());
  expect(p.peak(p.emit(Op::Coefficient,{denorm},{},.5)) == p.peak(denorm), "underflow rounded inward");

  // A logical width of 2 over 4 physical lanes. A cap on this SSA result
  // must be intersected per lane, neither broadcast by its max nor applied
  // to a predecessor. The second replica is observed by a following mask.
  int final = p.emit(Op::Relin, {sq});
  p.bounds.finish(final, final, p.edges[final], {.75, .25});
  for (int i = 0; i < 4; ++i) expect(p.slot(final,i) == (i%2 ? .25 : .75), "affine parent map");
  expect(p.slot(sq,1) > 1., "affine cap leaked into an intermediate");
}
} // namespace

int main() { counterexample(); bsgs(); roundingAndAffineParents(); }
