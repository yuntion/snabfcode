#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace snabf::pass3 {
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
};
bool keyOp(Op);
// Fixed EVK blocks, not a new partition of the surviving suffix.
int activeDigitBlocks(int level, int k, int alpha);
double secretBound(int n);
double
residualFactor(Op, int n); // canonical residual bound per unit omitted product
double rounding(int n);
struct Solution {
  int scale = 0;
  std::vector<double> bits, scales;
  double pBits = 0, objective = 0;
  std::vector<int> k;
  std::vector<int64_t> q, p;
  std::vector<double> physical, errors;
  std::vector<std::string> cuts;
};
void validate(const Graph &, const Options &);
int scaleClass(const Graph &, int);
} // namespace snabf::pass3
