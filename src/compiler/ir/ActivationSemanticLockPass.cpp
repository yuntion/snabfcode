#include "NativeLinalgImportInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <cmath>

using namespace mlir;
namespace snabf::native {
namespace {
// Comparisons use the rounded *source* serialization, not a host libm value
// or a tolerance. Only the two original analytic expression trees are in v1.
enum Literal { One, Half, InvSqrt2 };
constexpr uint64_t bits32[] = {0x3f800000, 0x3f000000, 0x3f3504f3};
constexpr uint64_t bits64[] = {0x3ff0000000000000, 0x3fe0000000000000, 0x3fe6a09e667f3bcd};
constexpr const char *roles[] = {"one", "half", "inv_sqrt2"};

struct Matcher {
  Block &block;
  const ValueIDs &outerIDs;
  Value x;
  SmallVector<std::pair<Literal, Value>> literals;

  Operation *local(Value v, StringRef name) const {
    auto *op = v ? v.getDefiningOp() : nullptr;
    return op && op->getBlock() == &block && op->getName().getStringRef() == name ? op : nullptr;
  }
  bool literal(Value v, Literal role) const {
    auto c = v.getDefiningOp<arith::ConstantOp>();
    auto f = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
    if (!f || !outerIDs.contains(v) || f.getType() != x.getType()) return false;
    return f.getValue().bitcastToAPInt().getZExtValue() == (f.getType().isF32() ? bits32[role] : bits64[role]);
  }
  // Match a commutative binary op with one literal; preserve actual operand
  // order in the body witness. Do not commute a division or reassociate a mul.
  Value other(Value v, StringRef name, Literal role) {
    if (auto *op = local(v, name))
      for (unsigned i : {0u, 1u})
        if (literal(op->getOperand(i), role)) {
          literals.emplace_back(role, op->getOperand(i));
          return op->getOperand(1-i);
        }
    return {};
  }
  bool silu(Value result) {
    auto *p = local(result, "arith.mulf");
    if (!p) return false;
    for (unsigned i : {0u, 1u}) {
      if (p->getOperand(i) != x) continue;
      auto *div = local(p->getOperand(1-i), "arith.divf");
      if (!div || !literal(div->getOperand(0), One)) continue;
      literals.clear();
      literals.emplace_back(One, div->getOperand(0));
      auto *exp = local(other(div->getOperand(1), "arith.addf", One), "math.exp");
      auto *neg = exp ? local(exp->getOperand(0), "arith.negf") : nullptr;
      if (neg && neg->getOperand(0) == x) return true;
    }
    return false;
  }
  bool gelu(Value result) {
    auto *p = local(result, "arith.mulf");
    if (!p) return false;
    for (unsigned i : {0u, 1u}) {
      literals.clear();
      if (other(p->getOperand(i), "arith.mulf", Half) != x) continue;
      auto *erf = local(other(p->getOperand(1-i), "arith.addf", One), "math.erf");
      if (erf && other(erf->getOperand(0), "arith.mulf", InvSqrt2) == x) return true;
    }
    return false;
  }
};
} // namespace

LogicalResult lockActivation(const Source &s, linalg::GenericOp g, const ValueIDs &ids,
                             StringRef site, json::Object &lock) {
  if (failed(s.attrs(g, {"indexing_maps", "iterator_types", "operandSegmentSizes", "domain_lower", "domain_upper"})))
    return failure();
  auto maps = g.getIndexingMapsArray();
  auto iters = g.getIteratorTypesArray();
  if (g.getInputs().size() != 1 || g.getOutputs().size() != 1 || g->getNumResults() != 1 ||
      maps.size() != 2 || !llvm::all_of(maps, [](AffineMap m) { return m.getNumDims()==1 && m.getNumSymbols()==0 && m.isIdentity(); }) ||
      iters.size() != 1 || iters[0] != utils::IteratorType::parallel || !g.getRegion().hasOneBlock())
    return s.reject(g, "[SNABF_NATIVE_GENERIC] expected unary elementwise identity maps and one parallel iterator");
  auto lo = g->getAttrOfType<FloatAttr>("domain_lower"), hi = g->getAttrOfType<FloatAttr>("domain_upper");
  if (!lo || !hi || !lo.getType().isF64() || !hi.getType().isF64() || !lo.getValue().isFinite() ||
      !hi.getValue().isFinite() || lo.getValueAsDouble() > hi.getValueAsDouble())
    return s.reject(g, "[SNABF_NATIVE_GENERIC] expected declared ordered finite f64 activation bounds");
  auto &b = g.getRegion().front();
  if (b.getNumArguments()!=2 || b.empty() || !isa<linalg::YieldOp>(b.back()) || b.back().getNumOperands()!=1)
    return s.reject(g, "[SNABF_NATIVE_GENERIC] malformed scalar region");
  if (!b.getArgument(1).use_empty())
    return s.reject(b.getArgument(1).use_begin()->getOwner(), "[SNABF_NATIVE_GENERIC] activation reads outs block argument");
  if (failed(s.attrs(&b.back(), {}))) return failure();
  for (auto &op : b.without_terminator()) {
    if (isa<arith::MaximumFOp, arith::SelectOp>(op))
      return s.reject(&op, "[SNABF_RELU_NOT_RELEASED] maximum/select is outside C-1");
    if (!isa<arith::MulFOp, arith::AddFOp, arith::NegFOp, arith::DivFOp, math::ExpOp, math::ErfOp>(op))
      return s.reject(&op, "[SNABF_NATIVE_GENERIC] unsupported scalar operation " + op.getName().getStringRef());
    if (failed(s.strictFloatOp(&op))) return failure();
  }
  Value yielded = b.back().getOperand(0);
  Matcher m{b, ids, b.getArgument(0), {}};
  StringRef kind;
  if (m.silu(yielded)) kind = "silu";
  else if (m.gelu(yielded)) kind = "gelu_exact_erf";
  if (kind.empty() || std::distance(b.begin(), b.end()) != 6)
    return s.reject(g, "[SNABF_NATIVE_GENERIC] body is not the exact connected source-SSA SiLU or erf-GeLU expression");

  ValueIDs scoped(ids);
  scoped[b.getArgument(0)] = (site + ".arg0").str();
  json::Array body, literals;
  for (auto &op : b.without_terminator()) {
    auto id = (site + ".b" + llvm::Twine(body.size())).str();
    json::Array operands;
    for (auto v : op.getOperands()) operands.push_back(scoped.lookup(v));
    body.push_back(json::Object{{"id", id}, {"op", op.getName().getStringRef().str()},
                                {"inputs", std::move(operands)}, {"location", s.location(&op)}});
    scoped[op.getResult(0)] = std::move(id);
  }
  for (auto [role, v] : m.literals) {
    auto f = cast<FloatAttr>(v.getDefiningOp<arith::ConstantOp>().getValue());
    literals.push_back(json::Object{{"role", roles[role]}, {"id", ids.lookup(v)},
                                    {"type", typeText(v.getType())}, {"bits", floatBits(f.getValue())}});
  }
  lock = json::Object{{"site", site.str()}, {"kind", kind.str()}, {"x", ids.lookup(g.getInputs().front())},
                      {"result", site.str()}, {"region_x", scoped.lookup(m.x)}, {"region_result", scoped.lookup(yielded)},
                      {"body", std::move(body)}, {"literals", std::move(literals)}, {"location", s.location(g)},
                      {"declared_domain", json::Array{lo.getValueAsDouble(), hi.getValueAsDouble()}},
                      {"origin", json::Object{{"kind", "source"}, {"source_sha256", s.digest},
                                             {"func_symbol", g->getParentOfType<func::FuncOp>().getSymName().str()},
                                             {"call_path", json::Array{}}, {"iteration_path", json::Array{}}}}};
  return success();
}
} // namespace snabf::native
