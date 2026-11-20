#include "Pass2IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include <cmath>

namespace snabf::ir {
namespace {
Operation *producer(Value v, StringRef op) {
  Operation *d = v.getDefiningOp();
  return d && name(d) == op ? d : nullptr;
}
bool literal(Value v, double expected) {
  auto *c = producer(v, "arith.constant");
  if (!c) return false;
  auto value = c->getAttr("value");
  auto f = dyn_cast_or_null<FloatAttr>(value);
  if (auto dense = dyn_cast_or_null<DenseFPElementsAttr>(value); dense && dense.isSplat())
    f = dense.getSplatValue<FloatAttr>();
  return f && f.getValueAsDouble() == (f.getType().isF32() ? double(float(expected)) : expected);
}
bool one(Value v) { return literal(v, 1.0); }
constexpr double invRoot2 = 0x1.6a09e667f3bcdp-1, root2 = 0x1.6a09e667f3bcdp+0;

// Return the very same x on both branches. No tolerance on literals: accept
// only the correctly rounded constants of the source element type.
bool cdf(Value v, Value x) {
  auto *a = producer(v, "arith.addf");
  if (!a) return false;
  for (unsigned i : {0u, 1u}) {
    if (!one(a->getOperand(i))) continue;
    auto *erf = producer(a->getOperand(1-i), "math.erf");
    if (!erf) continue;
    auto z = erf->getOperand(0);
    if (auto *mul = producer(z, "arith.mulf"))
      for (unsigned j : {0u, 1u})
        if (mul->getOperand(j) == x && literal(mul->getOperand(1-j), invRoot2)) return true;
    if (auto *div = producer(z, "arith.divf"); div && div->getOperand(0) == x) {
      auto d = div->getOperand(1);
      if (literal(d, root2)) return true;
      if (auto *s = producer(d, "math.sqrt"); s && literal(s->getOperand(0), 2)) return true;
    }
  }
  return false;
}
Value geluInput(Value result) {
  auto *m = producer(result, "arith.mulf");
  if (!m) return {};
  // All associations and commutations of half * x * (1+erf(...)). Do not
  // flatten x itself (it may be another elementwise source expression).
  for (unsigned i : {0u,1u}) {
    auto *pair = producer(m->getOperand(i), "arith.mulf");
    if (!pair) continue;
    Value f[] = {pair->getOperand(0), pair->getOperand(1), m->getOperand(1-i)};
    for (int j=0; j<3; ++j) if (literal(f[j], 0.5))
      for (int k : {1,2}) if (cdf(f[(j+k)%3], f[(j+3-k)%3])) return f[(j+3-k)%3];
  }
  return {};
}

void expression(Value v, Value x, llvm::DenseSet<Operation *> &ops) {
  if (v == x) return;
  if (auto *d = v.getDefiningOp(); d && ops.insert(d).second)
    for (auto a : d->getOperands()) expression(a, x, ops);
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

struct Site { Operation *root; Value x; StringRef kind; };
std::optional<Site> sourceSite(Operation *op) {
  if (auto g = dyn_cast<linalg::GenericOp>(op)) {
    if (g->getNumOperands()!=2 || g->getNumResults()!=1 || !g.getRegion().hasOneBlock() ||
        g.getNumDpsInputs()!=1 || g->getOperand(0).getType()!=g->getResult(0).getType()) return {};
    auto &block = g.getRegion().front();
    if (block.getNumArguments()!=2 || block.getTerminator()->getNumOperands()!=1) return {};
    Value result = block.getTerminator()->getOperand(0), x = block.getArgument(0);
    StringRef kind;
    if (geluInput(result) == x) kind = "gelu";
    else if (siluBody(op)) kind = "silu";
    else return {};
    if (!llvm::all_of(g.getIndexingMapsArray(), [](AffineMap m){return m.isIdentity();}) ||
        !llvm::all_of(g.getIteratorTypesArray(), [](auto i){return i==utils::IteratorType::parallel;})) return {};
    llvm::DenseSet<Operation *> members;
    expression(result,x,members);
    for (auto &o : block.without_terminator()) if (!members.contains(&o)) return {};
    return Site{result.getDefiningOp(), x, kind};
  }
  if (op->getNumResults()==1 && isa<RankedTensorType>(op->getResult(0).getType()))
    if (Value x=geluInput(op->getResult(0))) return Site{op,x,"gelu"};
  return {};
}

// The recognized source contract is the analytic activation, not an
// arbitrary tagged body. Normalize only its association and sqrt spelling.
void canonicalGelu(IRRewriter &rw, Site s) {
  auto loc = s.root->getLoc(); auto t = s.x.getType();
  rw.setInsertionPoint(s.root);
  auto constant = [&](double val) -> Value {
    auto el = dyn_cast<ShapedType>(t);
    auto f = rw.getFloatAttr(el ? el.getElementType() : t, val);
    TypedAttr a = el ? cast<TypedAttr>(DenseElementsAttr::get(el, f)) : cast<TypedAttr>(f);
    return arith::ConstantOp::create(rw,loc,a);
  };
  auto z = arith::MulFOp::create(rw,loc,s.x,constant(invRoot2));
  auto erf = math::ErfOp::create(rw,loc,z);
  auto a = arith::AddFOp::create(rw,loc,erf,constant(1));
  auto half = arith::MulFOp::create(rw,loc,s.x,constant(0.5));
  auto result = arith::MulFOp::create(rw,loc,half,a);
  rw.replaceOp(s.root,result.getResult());
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

Value plainInput(Value v, StringRef kind, double val) {
  auto s = unmask(v).getDefiningOp<StepOp>();
  if (!s || s.getKind()!=kind) return {};
  auto w = plainWeights(s.getInputs()[1]);
  if (!w.isSplat() || w.getSplatValue<APFloat>().convertToDouble() !=
      (w.getElementType().isF32() ? double(float(val)) : val)) return {};
  return unmask(s.getInputs()[0]);
}

Value joinedInput(StepOp join, StringRef kind) {
  for (unsigned i : {0u,1u}) {
    Value x=unmask(join.getInputs()[i]), branch=unmask(join.getInputs()[1-i]);
    if (kind == "gelu") {
      if (auto half=plainInput(x,"ckks.mul_plain",0.5)) x=half;
      else if (auto half=plainInput(branch,"ckks.mul_plain",0.5)) branch=half;
      else continue;
      branch=plainInput(branch,"ckks.add_plain",1);
      if (!branch) continue;
    } else if (kind != "silu") continue;
    auto raw=branch.getDefiningOp<PolyEvalOp>();
    if (!raw || raw->hasAttr("activation")) continue;
    auto input = kind=="gelu" ? plainInput(raw.getInput(),"ckks.mul_plain",invRoot2) : unmask(raw.getInput());
    if (input == x) return x;
  }
  return {};
}
}
void markActivationSources(ModuleOp m) {
  Builder b(m.getContext());
  SmallVector<Attribute> records;
  SmallVector<Site> gelu;
  llvm::DenseSet<Operation *> tagged;
  m.walk([&](Operation *op) {
    auto site = sourceSite(op);
    if (!site) return;
    for (auto *owner : {op, site->root}) {
      if (auto attr = owner->getAttr("snabf.activation")) {
        require(attr == b.getStringAttr(site->kind), "source activation marker disagrees with expression");
        tagged.insert(owner);
      }
      if (auto attr = owner->getAttr("approximate"))
        require(attr == b.getStringAttr("none"), "approximate activation is not exact");
    }
    auto f = op->getParentOfType<func::FuncOp>();
    records.push_back(b.getDictionaryAttr({b.getNamedAttr("function", b.getStringAttr(f.getName())),
      b.getNamedAttr("semantics", b.getStringAttr(site->kind))}));
    if (site->kind == "gelu") gelu.push_back(*site);
  });
  m.walk([&](Operation *op){ require(!op->hasAttr("snabf.activation") || tagged.contains(op),
                                    "unproven source activation marker"); });
  require(!records.empty(), "no source exact SiLU or erf GeLU activation found");
  IRRewriter rw(m.getContext());
  // Reverse order keeps x valid when one tensor activation feeds another.
  for (auto site : llvm::reverse(gelu)) canonicalGelu(rw, site);
  if (!gelu.empty()) m->setAttr("snabf.protect_erf", b.getUnitAttr());
  else m->removeAttr("snabf.protect_erf");
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
  llvm::DenseSet<Operation *> tensorExpr;
  for (auto &op : f.getBody().front())
    if (auto s = sourceSite(&op); s && s->root == &op) {
      expression(op.getResult(0),s->x,tensorExpr); activation=true;
    }
  for (Operation &op : f.getBody().front()) {
    for (Type type : op.getResultTypes())
      if (auto t = dyn_cast<ShapedType>(type); t && !t.hasStaticShape())
        return reject(&op, "dynamic_tensor_shape");
    auto k = name(&op);
    if (tensorExpr.contains(&op)) continue;
    if (k == "linalg.matvec") { linear = true; continue; }
    if (k == "linalg.generic") {
      if (sourceSite(&op)) { activation = true; continue; }
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
      return reject(&op, "outside_static_linear_activation_add_subset");
  }
  if (!linear || !activation) return reject(f, "expected_matvec_and_known_activation");
  return success();
}
void liftActivationRegions(ModuleOp m) {
  auto records = field<ArrayAttr>(m, "snabf.activation_sources");
  IRRewriter rw(m.getContext());
  SmallVector<StepOp> joins;
  m.walk([&](StepOp step){ if(step.getKind()=="ckks.mul") joins.push_back(step); });
  size_t used = 0;
  for (StepOp join : joins) {
      require(used < records.size(), "unmarked HEIR activation join");
      auto r = cast<DictionaryAttr>(records[used++]);
      auto kind=field<StringAttr>(r,"semantics");
      auto x=joinedInput(join,kind.getValue());
      require(bool(x), "HEIR join, scalar constants or original input disagree with source " + kind.getValue());
      auto function = join->getParentOfType<func::FuncOp>();
      require(field<StringAttr>(field<DictionaryAttr>(function,"heir.interface"),"func_name") == field<StringAttr>(r,"function"), "source/evaluator provenance mismatch");
      rw.setInsertionPoint(join);
      auto logical = PolyEvalOp::create(rw, join.getLoc(), cast<CtType>(x.getType()), x, rw.getArrayAttr({}));
      logical->setAttr("activation", kind);
      for (auto key : {"native","native_props","native_attrs"}) logical->setAttr(key, join->getAttr(key));
      rw.replaceOp(join, logical.getOut());
  }
  require(used == records.size() && used > 0, "missing source-proven HEIR activation join");
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
  m.walk([&](StepOp op){ require(op.getKind()=="ckks.add", "unsupported nonidentity arithmetic around a marked activation"); });
}
}
