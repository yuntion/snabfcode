#pragma once
#include "IR.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/JSON.h"
#include <stdexcept>

namespace snabf::ir {
using namespace mlir;
struct ContractError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
inline void require(bool ok, const Twine &reason) {
  if (!ok) throw ContractError(reason.str());
}
inline StringRef name(Operation *o) { return o->getName().getStringRef(); }
inline DictionaryAttr props(Operation *o) {
  auto d = dyn_cast_or_null<DictionaryAttr>(o->getPropertiesAsAttribute());
  return d ? d : DictionaryAttr::get(o->getContext());
}
template <class T>
T field(DictionaryAttr d, StringRef k) {
  auto a = d.getAs<T>(k);
  require(bool(a), "missing or invalid field: " + k);
  return a;
}
template <class T>
T field(Operation *o, StringRef k) {
  auto a = o->getAttrOfType<T>(k);
  if (!a) a = props(o).getAs<T>(k);
  require(bool(a), name(o) + ": missing or invalid " + k);
  return a;
}
template <class V>
std::string spelling(V v) {
  std::string text;
  llvm::raw_string_ostream out(text);
  v.print(out);
  return text;
}
struct NativeCT {
  int64_t n, level, logScale;
  std::vector<int64_t> chain;
  std::string key;
};
NativeCT decodeCT(Type type);
struct Parameters {
  unsigned logN;
  std::vector<int64_t> q, p;
  int64_t logScale;
};
Parameters decodeParameters(Attribute a);
std::string weightId(DenseFPElementsAttr weights);
DenseFPElementsAttr preparedWeights(Value table, ArrayRef<int64_t> diagonals);
DenseFPElementsAttr plainWeights(Value table);
void importCKKS(ModuleOp m, int64_t bytes, double inputBound, double hoist,
                double giant, double eta);
void planModule(ModuleOp m, bool independent);
void publishPass1(ModuleOp m);
void verifyPlan(ModuleOp m);
void lowerCKKS(ModuleOp m);
llvm::json::Object bindBackend(ModuleOp m);
void registerPasses();
}  // namespace snabf::ir
