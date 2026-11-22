#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snabf::pass3 {
class CostProfile;
enum class Op {
  Source,
  Drop,
  Add,
  Multiply,
  Mask,
  Coefficient,
  AddPlain,
  Rescale,
  Baby,
  Giant,
  Relin, Down
};
const char *name(Op);
struct Edge {
  std::string id;
  Op op;
  int level;
  std::vector<int> inputs;
  std::vector<double>
      gains;          // decoded absolute input-error gains on the ideal domain
  double magnitude;   // maximum absolute ideal slot value
  int maskLevel = -1; // Mask encodes at q[maskLevel]; Coefficient is adjustable
  int poly = -1, node = -1;
  int ring = 0; // 0 means the source profile; Down's switching arithmetic is in the source ring.
  int shift = 0, routeSite = -1; // Actual emitted rotation and hoist grouping.
};
struct PolyNode {
  std::string op;
  std::vector<int> args;
  int depth;
};
struct Poly {
  int level, output;
  std::vector<PolyNode> nodes;
};
struct Graph {
  int n = 0, levels = 0, sink = -1;
  double approximation = 0;
  std::vector<Edge> edges;
  std::vector<Poly> polys;
};
struct Options {
  int minScale = 36, maxScale = 46, maxK = 1;
  double tolerance = 1. / 4096, securityCap = 350, minBase = 24, guardBits = 1;
  // -1 preserves frozen one-P receipts; 0 searches {1,2}; otherwise pin |P|.
  int alpha = -1;
  bool heterogeneous = false;
  // The pinned HEIR materializer rounds dropped moduli to integer log scales.
  // Keep every drop in the default-scale bin when specializing a HEIR graph.
  bool heirScaleAnchor = false;
  // Continuous prime widths (q0 has its own minBase). Margin is also
  // enforced as an exact integer inequality on each instantiated block.
  double primeWidthMin = 24., primeWidthMax = 59.;
  int auxMarginBits = 2;
  std::shared_ptr<const CostProfile> costProfile; // null: frozen legacy proxy
  bool coSlotMagnitude = false; // IR extraction policy; absent in frozen plans.
  bool customLimits() const { return primeWidthMin != 24. || primeWidthMax != 59. || auxMarginBits != 2; }
  bool blockSearch() const { return alpha != -1 || heterogeneous || customLimits(); }
  bool admits(int count) const {
    return alpha == 0 ? (count == 1 || count == 2) : count == (alpha < 0 ? 1 : alpha);
  }
};
bool keyOp(Op);
// Fixed EVK blocks, not a new partition of the surviving suffix.
int activeDigitBlocks(int level, int k, int alpha);
long double specialModulus(const std::vector<int64_t> &p);
long double switchKeyError(const std::vector<int64_t> &q,
                           const std::vector<int64_t> &p, int level, int k, int n);
double secretBound(int n);
double
residualFactor(Op, int n); // canonical residual bound per unit omitted product
double rounding(int n);
struct Solution {
  int scale = 0;
  std::vector<double> bits, scales;
  double pBits = 0, objective = 0;
  std::vector<double> auxBits;
  std::vector<int> k;
  std::vector<int64_t> q, p;
  std::vector<double> physical, errors;
  std::vector<std::string> cuts;
};
void validate(const Graph &, const Options &);
int scaleClass(const Graph &, int);
} // namespace snabf::pass3
