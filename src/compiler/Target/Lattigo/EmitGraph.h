#pragma once
#include "Pipeline.h"
#include <map>
#include <set>

namespace snabf::ir::lattigo_target::detail {
enum class Op { Source, Drop, Mul, Mask, Coefficient, AddPlain, Add, Rescale,
                Relin, Baby, Giant, Down };
struct Edge {
  Op op;
  std::string id;
  int64_t n, level, k;
  double scale;
  ArrayRef<int64_t> inputs;
  ArrayRef<double> values;
  int64_t shift = 0, site = -1;
};
struct EmitGraph {
  explicit EmitGraph(DictionaryAttr);
  struct Operand { DictionaryAttr value, provided, required; std::string input; ArrayAttr consumers; };
  DictionaryAttr program, transition, switchingDomain;
  std::vector<Operand> operands;
  int64_t source, target, width, slots, sink;
  std::vector<Edge> edges;
  std::map<int64_t, DictionaryAttr> families;
  std::set<int64_t> relin;
  std::map<std::pair<int64_t,int64_t>, std::vector<size_t>> hoists;
};
inline int64_t num(DictionaryAttr d, StringRef k) { return field<IntegerAttr>(d,k).getInt(); }
inline double real(DictionaryAttr d, StringRef k) { return field<FloatAttr>(d,k).getValueAsDouble(); }
inline ArrayRef<int64_t> ints(DictionaryAttr d,StringRef k) { return field<DenseI64ArrayAttr>(d,k).asArrayRef(); }
inline StringRef str(DictionaryAttr d,StringRef k) { return field<StringAttr>(d,k).getValue(); }
} // namespace snabf::ir::lattigo_target::detail
