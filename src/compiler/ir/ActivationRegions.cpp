#include "Pass2IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <cmath>

namespace snabf::ir {
namespace {
Operation *producer(Value v, StringRef op) {
  Operation *d = v.getDefiningOp();
  return d && name(d) == op ? d : nullptr;
}
bool one(Value v) {
  auto *c = producer(v, "arith.constant");
  return c && field<FloatAttr>(c, "value").getValue().isExactlyValue(1.0);
}
bool siluBody(Operation *g) {
  if (g->getNumRegions() != 1 || !g->getRegion(0).hasOneBlock() ||
      g->getNumOperands() != 2 || g->getNumResults() != 1) return false;
  auto &b = g->getRegion(0).front();
  if (b.getNumArguments() != 2 || b.getTerminator()->getNumOperands() != 1) return false;
  auto x = b.getArgument(0);
  auto *mul = producer(b.getTerminator()->getOperand(0), "arith.mulf");
  if (!mul) return false;
  for (unsigned i : {0u,1u}) {
    if (mul->getOperand(i) != x) continue;
    auto *div = producer(mul->getOperand(1-i), "arith.divf");
    if (!div || !one(div->getOperand(0))) continue;
    auto *add = producer(div->getOperand(1), "arith.addf");
    if (!add) continue;
    for (unsigned j : {0u,1u}) {
      if (!one(add->getOperand(j))) continue;
      auto *exp = producer(add->getOperand(1-j), "math.exp");
      if (exp) if (auto *neg = producer(exp->getOperand(0), "arith.negf"))
        if (neg->getOperand(0) == x) return true;
    }
  }
  return false;
}
// Strip only value-preserving paths, not arbitrary plaintext masks. The old
// HEIR schedule may place a multiplication by encoded ones on either branch.
Value unmask(Value v) {
  while (auto s = v.getDefiningOp<StepOp>()) {
    auto k = s.getKind();
    if (k == "ckks.mul_plain") {
      auto w = plainWeights(s->getOperand(1));
      if (!w.isSplat() || !w.getSplatValue<APFloat>().isExactlyValue(1.0)) break;
    } else if (k != "ckks.rescale" && k != "ckks.level_reduce" && k != "ckks.relinearize") break;
    v = s->getOperand(0);
  }
  return v;
}
}
void markActivationSources(ModuleOp m) {
  Builder b(m.getContext());
  SmallVector<Attribute> records;
  m.walk([&](Operation *op) {
    if (name(op) != "linalg.generic" || !siluBody(op)) return;
    auto maps = field<ArrayAttr>(op,"indexing_maps");
    require(maps.size() == 2 && llvm::all_of(maps, [](Attribute a) {
      return cast<AffineMapAttr>(a).getValue().isIdentity();
    }), "activation must be elementwise on the source packing");
    auto f = op->getParentOfType<func::FuncOp>();
    records.push_back(b.getDictionaryAttr({b.getNamedAttr("function", b.getStringAttr(f.getName())),
      b.getNamedAttr("semantics", b.getStringAttr("silu"))}));
  });
  require(!records.empty(), "no source x/(1+exp(-x)) activation found");
  m->setAttr("snabf.activation_sources", b.getArrayAttr(records));
}
// Product entry v1. Classify operations while their source locations and
// tensor semantics still exist. Frozen replay has its own entry contract.
LogicalResult verifyStaticSource(ModuleOp m) {
  auto reject = [](Operation *op, StringRef reason) -> LogicalResult {
    return op->emitError() << "[SNABF_UNSUPPORTED_OP] op=" << name(op)
                           << " reason=" << reason;
  };
  auto funcs = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(funcs) || !llvm::hasSingleElement(*m.getBody()))
    return reject(m, "expected_one_model_function");
  auto f = *funcs.begin();
  if (f.isExternal() || !f.getBody().hasOneBlock() || f.getNumArguments() != 1 ||
      f.getNumResults() != 1 || !f.getArgAttr(0, "secret.secret"))
    return reject(f, "expected_single_secret_input_single_result_straight_line");
  auto input = dyn_cast<RankedTensorType>(f.getArgument(0).getType());
  if (!input || !input.hasStaticShape() || input.getRank() != 1 ||
      input.getDimSize(0) <= 0 || !input.getElementType().isF32() ||
      f.getResultTypes()[0] != input)
    return reject(f, "expected_same_static_vector_shape");
  bool linear = false, activation = false;
  for (Operation &op : f.getBody().front()) {
    for (Type type : op.getResultTypes())
      if (auto t = dyn_cast<ShapedType>(type); t && !t.hasStaticShape())
        return reject(&op, "dynamic_tensor_shape");
    auto k = name(&op);
    if (k == "linalg.matvec") { linear = true; continue; }
    if (k == "linalg.generic") {
      if (siluBody(&op)) { activation = true; continue; }
      auto &r = op.getRegion(0);
      if (op.getNumOperands() == 3 && r.hasOneBlock() &&
          r.front().getNumArguments() == 3 && r.front().getTerminator()->getNumOperands() == 1) {
        auto *add = producer(r.front().getTerminator()->getOperand(0), "arith.addf");
        auto maps = op.getAttrOfType<ArrayAttr>("indexing_maps");
        if (add && maps && maps.size() == 3 && llvm::all_of(maps, [](Attribute a) {
              return cast<AffineMapAttr>(a).getValue().isIdentity();
            }) && ((add->getOperand(0) == r.front().getArgument(0) && add->getOperand(1) == r.front().getArgument(1)) ||
                  (add->getOperand(1) == r.front().getArgument(0) && add->getOperand(0) == r.front().getArgument(1)))) continue;
      }
      return reject(&op, "unrecognized_elementwise_region");
    }
    if (k != "arith.constant" && k != "linalg.fill" && k != "tensor.empty" &&
        k != "linalg.add" && k != "arith.addf" && k != "func.return")
      return reject(&op, "outside_static_linear_silu_add_subset");
  }
  if (!linear || !activation) return reject(f, "expected_matvec_and_silu");
  return success();
}
void liftActivationRegions(ModuleOp m) {
  auto records = field<ArrayAttr>(m, "snabf.activation_sources");
  IRRewriter rw(m.getContext());
  SmallVector<StepOp> joins;
  m.walk([&](StepOp step){ if(step.getKind()=="ckks.mul") joins.push_back(step); });
  size_t used = 0;
  for (StepOp join : joins) {
    for (unsigned i : {0u,1u}) {
      auto raw = unmask(join.getInputs()[i]).getDefiningOp<PolyEvalOp>();
      auto x = unmask(join.getInputs()[1-i]);
      if (!raw || raw->hasAttr("activation") || unmask(raw.getInput()) != x) continue;
      require(used < records.size(), "unmarked HEIR activation join");
      auto r = cast<DictionaryAttr>(records[used++]);
      auto function = join->getParentOfType<func::FuncOp>();
      require(field<StringAttr>(field<DictionaryAttr>(function,"heir.interface"),"func_name") == field<StringAttr>(r,"function"), "source/evaluator provenance mismatch");
      rw.setInsertionPoint(join);
      auto logical = PolyEvalOp::create(rw, join.getLoc(), cast<CtType>(x.getType()), x, rw.getArrayAttr({}));
      logical->setAttr("activation", field<StringAttr>(r,"semantics"));
      for (auto key : {"native","native_props","native_attrs"}) logical->setAttr(key, join->getAttr(key));
      rw.replaceOp(join, logical.getOut());
      break;
    }
  }
  require(used == records.size() && used > 0, "source SiLU does not match the HEIR sigmoid/bypass SSA join");
  SmallVector<Operation *> graph;
  m.walk([&](Operation *o){ if(isa<StepOp,PolyEvalOp,LintransOp,SinkOp>(o)) graph.push_back(o); });
  // The synthesized DAG normalizes its own scale. Reconstruct logical edges
  // before scheduling residual joins; old HEIR alignment belongs to the old DAG.
  for (auto *o : graph) {
    if (!(isa<LintransOp,SinkOp>(o) || (isa<StepOp>(o) && cast<StepOp>(o).getKind()=="ckks.add"))) continue;
    for (auto &operand : o->getOpOperands()) {
      if (!isa<CtType>(operand.get().getType())) continue;
      auto root = unmask(operand.get());
      if (root != operand.get()) {
        auto *d = root.getDefiningOp();
        require(d && isa<PolyEvalOp,LintransOp,SourceOp,StepOp>(d), "invalid activation/suffix edge");
        operand.set(root);
      }
    }
  }
  for (auto *o : llvm::reverse(graph))
    if ((isa<StepOp,PolyEvalOp>(o)) && o->use_empty()) rw.eraseOp(o);
  m.walk([&](PolyEvalOp op){ require(op->hasAttr("activation"), "unclaimed HEIR polynomial"); });
  m.walk([&](StepOp op){ require(op.getKind()=="ckks.add", "unsupported nonidentity arithmetic around a marked SiLU"); });
}
}
