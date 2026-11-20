#include "Pass4IR.h"
#include "Pass2IR.h"
#include "Pass3IR.h"
#include "pass3/PrimeInstantiator.h"
#include "pass4/DynamicRingPlanner.h"
#include "pass4/RingSwitchPlan.h"
#include "pass4/SSACut.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
namespace snabf::ir {
namespace {
using namespace ::snabf::pass4;
int64_t num(DictionaryAttr d, StringRef k) {
  return field<IntegerAttr>(d, k).getInt();
}
double real(DictionaryAttr d, StringRef k) {
  return field<FloatAttr>(d, k).getValueAsDouble();
}
std::string hash(StringRef s) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(s)), true);
}
ProfileOp source(ModuleOp m) {
  ProfileOp p;
  m.walk([&](SourceOp s) {
    require(!p, "multiple_sources_unsupported");
    p = SymbolTable::lookupNearestSymbolFrom<ProfileOp>(
        s, s.getOut().getType().getProfile());
  });
  require(bool(p), "missing_source");
  return p;
}
LintransOp suffix(ModuleOp m) {
  LintransOp l;
  m.walk([&](LintransOp x) { l = x; });
  require(bool(l), "missing_terminal_lintrans");
  return l;
}
SlotLayoutDesc layout(ProfileOp p, int64_t n, int64_t l, double scale,
                      int64_t width) {
  auto slots = int64_t(p.getSlots());
  auto text = spelling(p.getLayout());
  auto expected = "#tensor_ext.original_type<originalType = tensor<" +
                  std::to_string(width) +
                  "xf32>, layout = #tensor_ext.layout<\"{ [i0] -> [ct, slot] : "
                  "ct = 0 and (-i0 + slot) mod " +
                  std::to_string(width) +
                  " = 0 and 0 <= i0 <= " + std::to_string(width - 1) +
                  " and 0 <= slot <= " + std::to_string(slots - 1) + " }\">>";
  require(text == expected, "unsupported_heir_packing_map");
  return {n,   n / 2, width, 1, slots, l, scale, Packing::PeriodicSingle,
          {0}, {0},   {}};
}
SlotLayoutAttr attr(Builder &b, const SlotLayoutDesc &x) {
  NamedAttrList a;
  auto put = [&](StringRef k, int64_t v) { a.set(k, b.getI64IntegerAttr(v)); };
  put("ring_dim", x.ring);
  put("num_slots", x.slots);
  put("logical_width", x.width);
  put("stride", x.stride);
  put("period", x.period);
  put("active_level", x.level);
  put("lane_count", x.offsets.size());
  a.set("packing_mode", b.getStringAttr(name(x.packing)));
  a.set("offsets", b.getDenseI64ArrayAttr(x.offsets));
  a.set("active_lanes", b.getDenseI64ArrayAttr(x.active));
  a.set("mask_signature", b.getDenseI64ArrayAttr(x.mask));
  a.set("log2_scale", b.getF64FloatAttr(x.logScale));
  return SlotLayoutAttr::get(b.getContext(), a.getDictionary(b.getContext()));
}
// A cut is a program point, not the operands of a chosen list of consumers.
// Enumerate *uses* in suffix order first, then group by SSA identity. E.g. x
// can be both the linear operand and a bypass: one value, two crossing uses.
struct Cut {
  DictionaryAttr record;
  std::string reason;
};
// HEIR may schedule an unused mask load after the cut. Its *handle* is a
// crossing block argument. Do not ignore the use or trust the type/name alone:
// prove that every caller passes a read-only table of literal-derived entries.
bool isUnusedPublicPreload(ModuleOp m, OpOperand &use) {
  Operation *load = use.getOwner();
  auto eval = load->getParentOfType<func::FuncOp>();
  if (name(load) != "preprocessing.load" || use.getOperandNumber() != 0 ||
      load->getNumOperands() != 1 || load->getNumResults() != 1 ||
      !load->getResult(0).use_empty() || !eval || eval.getNumArguments() != 2 ||
      use.get() != eval.getArgument(1)) return false;
  auto tag = load->getAttrOfType<IntegerAttr>("site_id");
  auto elem = load->getAttrOfType<TypeAttr>("element_type");
  if (!tag || tag.getInt() < 0 || !elem || elem.getValue() != load->getResult(0).getType()) return false;
  if (!llvm::all_of(use.get().getUses(), [](OpOperand &u) {
        return name(u.getOwner()) == "preprocessing.load" && u.getOperandNumber() == 0;
      })) return false;
  bool foundCaller = false, valid = true;
  m.walk([&](func::CallOp call) {
    if (call.getCallee() != eval.getSymName()) return;
    foundCaller = true;
    auto proved = [&]() {
      if (call.getNumOperands() != 2 || !call.getOperand(1).hasOneUse()) return false;
      auto make = call.getOperand(1).getDefiningOp<func::CallOp>();
      if (!make || make.getNumOperands() || make.getNumResults() != 1) return false;
      auto prep = m.lookupSymbol<func::FuncOp>(make.getCallee());
      if (!prep || !prep.getBody().hasOneBlock() || prep.getNumArguments()) return false;
      auto *ret = prep.getBody().front().getTerminator();
      if (!isa<func::ReturnOp>(ret) || ret->getNumOperands() != 1) return false;
      Value table = ret->getOperand(0);
      auto *empty = table.getDefiningOp();
      if (!empty || name(empty) != "preprocessing.empty" || empty->getNumOperands() ||
          empty->getNumResults() != 1 || table.getType() != use.get().getType()) return false;
      std::set<int64_t> sites;
      bool matched = false;
      for (OpOperand &u : table.getUses()) {
        Operation *store = u.getOwner();
        if (store == ret) continue;
        if (name(store) != "preprocessing.store" || store->getBlock() != ret->getBlock() ||
            store->getNumOperands() != 2 || store->getNumResults() || u.getOperandNumber() != 1) return false;
        auto id = store->getAttrOfType<IntegerAttr>("site_id");
        auto t = store->getAttrOfType<TypeAttr>("element_type");
        auto *p = store->getOperand(0).getDefiningOp();
        if (!id || id.getInt() < 0 || !sites.insert(id.getInt()).second || !t ||
            t.getValue() != store->getOperand(0).getType() || !p || p->getNumOperands() != 1 || p->getNumResults() != 1) return false;
        auto type = spelling(t.getValue());
        if (!((name(p) == "lwe.rlwe_encode" && StringRef(type).starts_with("!lwe.lwe_plaintext<")) ||
              (name(p) == "kernel.prepare_linear_transform" && StringRef(type).starts_with("!kernel.prepared_linear_transform<")))) return false;
        auto *literal = p->getOperand(0).getDefiningOp();
        if (!literal || name(literal) != "arith.constant" || literal->getNumOperands() ||
            !literal->getAttrOfType<DenseFPElementsAttr>("value")) return false;
        if (id == tag && t == elem) matched = true;
      }
      return matched;
    };
    valid &= proved();
  });
  return foundCaller && valid;
}
// A HEIR result wrapper may allocate the returned tensor before the cut. Its
// single element has no incoming contents when a sink overwrites it in full.
bool isOutputABIStorage(OpOperand &use, Operation *first) {
  Value v = use.get();
  auto *empty = v.getDefiningOp();
  auto *ins = use.getOwner();
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!empty || name(empty) != "tensor.empty" || empty->getNumOperands() ||
      empty->getNumResults() != 1 || !v.hasOneUse() || !type ||
      type.getRank() != 1 || type.getDimSize(0) != 1 ||
      name(ins) != "tensor.insert" || use.getOperandNumber() != 1 ||
      ins->getNumOperands() != 3 || ins->getNumResults() != 1 ||
      ins->getResult(0).getType() != type || !ins->getResult(0).hasOneUse())
    return false;
  auto sink = ins->getOperand(0).getDefiningOp<SinkOp>();
  auto *index = ins->getOperand(2).getDefiningOp();
  if (!sink || sink->getBlock() != first->getBlock() ||
      !first->isBeforeInBlock(sink) || ins->getOperand(0) != sink.getCipher() ||
      type.getElementType() != sink.getCipher().getType() ||
      !sink.getCipher().hasOneUse() || !index ||
      name(index) != "arith.constant" ||
      !isa<IndexType>(ins->getOperand(2).getType()))
    return false;
  auto zero = index->getAttrOfType<IntegerAttr>("value");
  auto *ret = ins->getBlock()->getTerminator();
  if (!zero || !zero.getValue().isZero() || !ret ||
      name(ret) != "func.return" || ret->getNumOperands() != 1 ||
      ret->getOperand(0) != ins->getResult(0))
    return false;
  (void)decodeCT(type.getElementType()); // verify the native ciphertext ABI,
                                         // not merely a tensor shape
  return true;
}
Cut liveCut(ModuleOp m, LintransOp lin, ConvertOp bridge,
            const ConcreteSSAContext &ctx) {
  Builder b(m.getContext());
  Block *block = lin->getBlock();
  require(block && block->getParent()->hasOneBlock(),
          "unsupported_multiblock_encrypted_cut");
  Operation *first = bridge ? bridge.getOperation() : lin.getOperation();
  if (bridge)
    require(bridge->getBlock() == block &&
                bridge->getNextNode() == lin.getOperation() &&
                lin.getInput() == bridge.getOut(),
            "stale_conversion_ssa");
  m.walk([&](Operation *o) {
    if (llvm::any_of(o->getOperandTypes(),
                     [](Type t) { return isa<CtType>(t); }) ||
        llvm::any_of(o->getResultTypes(),
                     [](Type t) { return isa<CtType>(t); }))
      require(o->getBlock() == block, "unsupported_cross_region_encrypted_cut");
  });
  auto crossing = enumerateCut(
      first, lin, [](Type t) { return isa<CtType>(t); },
      [](Operation *o) {
        auto add = dyn_cast<StepOp>(o);
        return add && add.getKind() == "ckks.add";
      });
  auto ordinal = [&](Operation *o) { return crossing.order.find(o)->second; };
  auto id = [&](Value v) {
    if (auto r = dyn_cast<OpResult>(v))
      return "op" + std::to_string(ordinal(r.getOwner())) + ".r" +
             std::to_string(r.getResultNumber());
    return "arg" + std::to_string(cast<BlockArgument>(v).getArgNumber());
  };
  using Uses = SmallVector<Attribute>;
  llvm::DenseMap<Value, unsigned> indices;
  SmallVector<std::pair<Value, Uses>> live;
  llvm::DenseSet<Value> abiStorage;
  bool residual = false;
  unsigned ctUses = 0, mainUses = 0;
  for (auto crossed : crossing.uses) {
    auto &use = *crossed.use;
    auto &op = *use.getOwner();
    Value v = use.get();
    Operation *def = v.getDefiningOp();
    auto [it, fresh] = indices.try_emplace(v, live.size());
    if (fresh)
      live.emplace_back(v, Uses{});
    bool ct = isa<CtType>(v.getType());
    // Only a verified prepared table or a literal is known public. In
    // particular a captured raw HEIR ciphertext is not a public SSA value.
    bool publicTable = isa<LintransOp>(&op) && use.getOperandNumber() == 1;
    bool publicPreload = !ct && isUnusedPublicPreload(m, use);
    bool outputStorage = !ct && isOutputABIStorage(use, first);
    if (outputStorage)
      abiStorage.insert(v);
    if (!ct)
      require(outputStorage || publicTable || publicPreload ||
                  (def && name(def) == "arith.constant" &&
                   (isa<IntegerType, IndexType, FloatType>(v.getType()) ||
                    (isa<ShapedType>(v.getType()) &&
                     isa<IntegerType, FloatType>(
                         cast<ShapedType>(v.getType()).getElementType())))),
              "unclassified_live_nonpublic_value");
    bool main = ct && &op == first && use.getOperandNumber() == 0;
    NamedAttrList u;
    u.set("user", b.getStringAttr(name(&op)));
    u.set("user_index", b.getI64IntegerAttr(ordinal(&op)));
    u.set("operand", b.getI64IntegerAttr(use.getOperandNumber()));
    if (publicPreload) u.set("literal_table_site", field<IntegerAttr>(&op, "site_id"));
    u.set("role", b.getStringAttr(outputStorage ? "output_abi_storage"
                                  : !ct         ? "public"
                                  : main        ? "main"
                                                : "unconverted"));
    if (ct) {
      auto p = SymbolTable::lookupNearestSymbolFrom<ProfileOp>(
          lin, cast<CtType>(v.getType()).getProfile());
      require(bool(p), "unbound_live_ciphertext");
      u.set("operand_key_domain", p.getKeyIdAttr());
    }
    if (auto step = dyn_cast<StepOp>(op))
      u.set("kind", step.getKindAttr());
    live[it->second].second.push_back(u.getDictionary(b.getContext()));
    if (ct) {
      ++ctUses;
      mainUses += main;
      residual |= !main && crossed.reachesResidualJoin;
    }
  }
  SmallVector<Attribute> encrypted, publicValues, outputStorage;
  for (auto &[v, uses] : live) {
    NamedAttrList e;
    e.set("id", b.getStringAttr(id(v)));
    e.set("uses", b.getArrayAttr(uses));
    e.set("producer",
          b.getStringAttr(v.getDefiningOp() ? name(v.getDefiningOp())
                                            : StringRef("block_argument")));
    if (auto t = dyn_cast<CtType>(v.getType())) {
      auto p =
          SymbolTable::lookupNearestSymbolFrom<ProfileOp>(lin, t.getProfile());
      auto edge = ctx.values.find(v);
      require(p && edge != ctx.values.end() &&
                  t.getN() == int64_t(p.getRing()) && t.getLevel() >= 0 &&
                  t.getLevel() < int64_t(p.getQ().size()),
              "unbound_live_ciphertext");
      double s = real(edge->second, "physical_log_scale");
      e.set("edge_id", field<StringAttr>(edge->second, "ID"));
      e.set("key_domain", p.getKeyIdAttr());
      e.set("profile", p.getSymNameAttr());
      e.set("ring_dim", p.getRingAttr());
      e.set("active_level", b.getI64IntegerAttr(t.getLevel()));
      e.set("q", b.getDenseI64ArrayAttr(p.getQ().take_front(t.getLevel() + 1)));
      e.set("p", p.getPAttr());
      e.set("physical_log_scale", b.getF64FloatAttr(s));
      e.set("ciphertext_basis", b.getStringAttr("Q"));
      e.set("layout", attr(b, layout(p, t.getN(), t.getLevel(), s,
                                     t.getLogical().getNumElements()))
                          .getFields());
      encrypted.push_back(e.getDictionary(b.getContext()));
    } else if (abiStorage.contains(v)) {
      e.set("type", b.getStringAttr(spelling(v.getType())));
      e.set("fully_overwritten", b.getBoolAttr(true));
      e.set("overwrite_index", b.getI64IntegerAttr(0));
      outputStorage.push_back(e.getDictionary(b.getContext()));
    } else {
      e.set("type", b.getStringAttr(spelling(v.getType())));
      publicValues.push_back(e.getDictionary(b.getContext()));
    }
  }
  NamedAttrList r;
  r.set("site", field<IntegerAttr>(lin, "site"));
  r.set("boundary_op", b.getStringAttr(name(first)));
  r.set("live_values", b.getArrayAttr(encrypted));
  r.set("public_values", b.getArrayAttr(publicValues));
  r.set("output_abi_storage", b.getArrayAttr(outputStorage));
  r.set("ciphertext_use_count", b.getI64IntegerAttr(ctUses));
  std::string reason;
  if (residual)
    reason = "residual_bypass_crosses_cut_unconverted";
  else if (mainUses != 1 || encrypted.size() != 1 || ctUses != 1)
    reason = "multiple_cut_edges_not_emittable";
  else if (!lin.getOut().hasOneUse() ||
           !isa<SinkOp>(*lin.getOut().getUsers().begin()))
    reason = "unsupported_post_lintrans_suffix";
  return {r.getDictionary(b.getContext()), reason};
}
struct Screen {
  double bits = 0;
  int64_t keyBytes, switchBytes;
  DictionaryAttr record;
  std::string reason;
};
struct InvalidEvidence : ContractError {
  using ContractError::ContractError;
};
struct RejectedProposal : std::runtime_error {
  using std::runtime_error::runtime_error;
};
void evidence(bool ok, const char *reason) {
  if (!ok)
    throw InvalidEvidence(reason);
}
constexpr StringLiteral policyName = "snabf.pass4_policy";
int64_t switchFloor(ModuleOp m) {
  int64_t floor = 128;
  if (auto a = m->getAttr("snabf.switching_key_security_bits")) {
    auto i = dyn_cast<IntegerAttr>(a);
    require(i && i.getType().isSignlessInteger(64) && i.getInt() >= 0,
            "invalid_switching_key_security_bits");
    floor = std::max(floor, i.getInt());
  }
  if (auto env = std::getenv("SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS")) {
    int64_t i;
    require(!StringRef(env).getAsInteger(10, i) && i >= 0,
            "invalid_switching_key_security_bits");
    floor = std::max(floor, i);
  }
  return floor;
}
llvm::json::Value readEvidence(ProfileOp p) {
  const char *file = std::getenv("SNABF_PASS4_ESTIMATES");
  evidence(file && *file, "missing_external_estimator_evidence");
  auto buf = llvm::MemoryBuffer::getFile(file);
  evidence(bool(buf), "missing_external_estimator_evidence");
  auto v = llvm::json::parse(buf.get()->getBuffer());
  evidence(bool(v) && v->getAsObject(), "invalid_external_estimator_evidence");
  auto &obj = *v->getAsObject();
  evidence(obj.getString("estimator") ==
                   "ae45d740ceb138af87a4f92edc592a0ab0d4f43f" &&
               obj.getString("model") ==
                   "rough-classical-unbounded-ternary-sigma3.2" &&
               obj.getInteger("source_ring") == int64_t(p.getRing()),
           "estimator_context_mismatch");
  if (obj.get("schema"))
    evidence(obj.getString("schema") == "snabf.security-domains.v2",
             "unsupported_security_schema");
  return std::move(*v);
}
bool sameBasis(const llvm::json::Array *a, ArrayRef<int64_t> b) {
  if (!a || a->size() != b.size())
    return false;
  for (size_t i = 0; i < b.size(); ++i)
    if ((*a)[i].getAsInteger() != b[i])
      return false;
  return true;
}
// Source Q and QP come from the already rechecked Pass 3 evidence. V2 also
// binds both to independent records in the external estimator cache. Old
// pinned archives have no separate source row; expose that provenance.
DictionaryAttr sourceScreenRecord(ModuleOp m, ProfileOp p,
                                  DictionaryAttr base) {
  verifySourceScreen(m, base);
  auto src = llvm::json::parse(
      field<StringAttr>(m, "snabf.security_screen").getValue());
  auto inst = src && src->getAsObject()
                  ? src->getAsObject()->getObject("instances")
                  : nullptr;
  evidence(inst, "missing_source_estimate");
  std::optional<llvm::json::Value> cache;
  const llvm::json::Object *other = nullptr;
  if (auto f = std::getenv("SNABF_PASS4_ESTIMATES"); f && *f) {
    cache = readEvidence(p);
    auto &obj = *cache->getAsObject();
    if (obj.get("schema")) {
      other = obj.getObject("source");
      evidence(other && sameBasis(other->getArray("q"), p.getQ()) &&
                   sameBasis(other->getArray("p"), p.getP()),
               "stale_external_source_basis");
    }
  }
  Builder b(m.getContext());
  NamedAttrList d;
  d.set("source", b.getStringAttr(other ? "pass3_and_external_v2"
                                        : "pass3_concrete_screen"));
  d.set("required_bits", b.getI64IntegerAttr(128));
  double worst = INFINITY;
  for (auto k : {"Q", "QP"}) {
    auto row = inst->getObject(k);
    auto attacks = row ? row->getObject("attacks_log2_rop") : nullptr;
    evidence(attacks && row->getString("modulus"), "missing_source_estimate");
    auto external = other ? other->getObject(k) : nullptr;
    if (other)
      evidence(external &&
                   external->getString("modulus") ==
                       row->getString("modulus") &&
                   external->getInteger("n") == int64_t(p.getRing()),
               "stale_external_source_basis");
    NamedAttrList a;
    a.set("modulus", b.getStringAttr(*row->getString("modulus")));
    a.set("n", p.getRingAttr());
    for (auto attack : {"usvp", "dual_hybrid"}) {
      auto x = attacks->getNumber(attack);
      evidence(x && std::isfinite(*x) && *x >= 128, "invalid_source_estimate");
      if (external) {
        auto y = external->getNumber(attack);
        evidence(y && std::isfinite(*y) && *y >= 128 &&
                     std::abs(*x - *y) < 1e-8,
                 "stale_external_source_estimate");
        *x = std::min(*x, *y);
      }
      worst = std::min(worst, *x);
      a.set(attack, b.getF64FloatAttr(*x));
    }
    d.set(k, a.getDictionary(b.getContext()));
  }
  d.set("min_bits", b.getF64FloatAttr(worst));
  return d.getDictionary(b.getContext());
}
Screen screen(ProfileOp p, int64_t n, int l, int64_t switchBits,
              bool estimates = true) {
  Builder b(p.getContext());
  evidence(l >= 0 && l < int64_t(p.getQ().size()),
           "invalid_security_instance_level");
  auto data = readEvidence(p);
  auto &obj = *data.getAsObject();
  auto tab = obj.getObject("profiles");
  auto row = tab ? tab->getObject(std::to_string(n) + ":" + std::to_string(l))
                 : nullptr;
  evidence(row && sameBasis(row->getArray("q"), p.getQ().take_front(l + 1)) &&
               sameBasis(row->getArray("p"), p.getP()),
           "missing_concrete_security_instance");
  auto bytes = row->getInteger("key_bytes"),
       sw = row->getInteger("switch_bytes");
  evidence(bytes && *bytes > 0 && sw && *sw > 0, "missing_physical_key_size");
  evidence(row->getInteger("key_level_q") == l &&
               row->getInteger("key_level_p") == int64_t(p.getP().size()) - 1,
           "stale_conversion_key_modulus");
  if (obj.get("schema")) {
    auto swBasis = row->getObject("switch_basis");
    evidence(
        swBasis &&
            sameBasis(swBasis->getArray("q"), p.getQ().take_front(l + 1)) &&
            sameBasis(swBasis->getArray("p"), p.getP()) &&
            swBasis->getInteger("physical_ring") == int64_t(p.getRing()) &&
            swBasis->getInteger("embedded_secret_ring") == n,
        "stale_conversion_key_modulus");
  }
  Screen out{0, *bytes, *sw, {}, {}};
  if (!estimates)
    return out;
  boost::multiprecision::cpp_int Q = 1, QP;
  for (auto x : p.getQ().take_front(l + 1))
    Q *= x;
  QP = Q;
  for (auto x : p.getP())
    QP *= x;
  NamedAttrList checks;
  checks.set("estimator", b.getStringAttr(*obj.getString("estimator")));
  checks.set("model", b.getStringAttr(*obj.getString("model")));
  checks.set("switching_key_security_bits", b.getI64IntegerAttr(switchBits));
  checks.set("target_security_bits", b.getI64IntegerAttr(128));
  double ordinary = INFINITY, switching = INFINITY;
  bool completeOrdinary = true, completeSwitching = true;
  for (StringRef role :
       {"Q", "QP", "switch_QP_small_secret", "switch_QP_large_ring"}) {
    auto r = row->getObject(role);
    evidence(
        r &&
            r->getInteger("n") ==
                (role == "switch_QP_large_ring" ? int64_t(p.getRing()) : n) &&
            r->getString("modulus") == (role == "Q" ? Q.str() : QP.str()),
        "missing_conversion_key_instance");
    bool ext = role.starts_with("switch_");
    auto &minimum = ext ? switching : ordinary;
    auto &complete = ext ? completeSwitching : completeOrdinary;
    NamedAttrList attacks;
    attacks.set("n", b.getI64IntegerAttr(*r->getInteger("n")));
    attacks.set("modulus", b.getStringAttr(*r->getString("modulus")));
    for (auto attack : {"usvp", "dual_hybrid"}) {
      auto x = r->getNumber(attack);
      if (!x && r->getString(attack) == "inf") {
        complete = false;
        attacks.set(attack, b.getStringAttr("inf"));
        continue;
      }
      evidence(x && std::isfinite(*x) && *x > 0, "invalid_attack_estimate");
      minimum = std::min(minimum, *x);
      attacks.set(attack, b.getF64FloatAttr(*x));
    }
    checks.set(role, attacks.getDictionary(b.getContext()));
  }
  // These are mathematically identical LWE inputs, not two independent votes.
  evidence(checks.get("QP") == checks.get("switch_QP_small_secret"),
           "inconsistent_concrete_security_instance");
  if (ordinary < 128)
    out.reason = "target_key_modulus_security_rejected";
  else if (!completeOrdinary)
    out.reason = "target_key_modulus_estimate_unavailable";
  else if (static_cast<long double>(switching) <
           static_cast<long double>(switchBits))
    out.reason = "switching_key_ext_modulus_security_rejected";
  else if (!completeSwitching)
    out.reason = "switching_key_ext_modulus_estimate_unavailable";
  checks.set("decision",
             b.getStringAttr(out.reason.empty() ? "verified" : out.reason));
  out.bits = std::min(ordinary, switching);
  out.record = checks.getDictionary(b.getContext());
  return out;
}
ConvertOp conv(ModuleOp m) {
  ConvertOp c;
  m.walk([&](ConvertOp x) {
    require(!c, "multiple_cut_edges_not_emittable");
    c = x;
  });
  return c;
}
double cost(DictionaryAttr params, int64_t sourceN, int64_t small,
            double switchPrice) {
  double total = 0;
  bool post = false;
  for (auto item : field<ArrayAttr>(params, "edges")) {
    auto d = cast<DictionaryAttr>(item);
    auto op = field<StringAttr>(d, "op").getValue();
    int l = num(d, "Level");
    if (op == "ring_down") {
      total += conversionWork(sourceN, l, switchPrice);
      post = true;
      continue;
    }
    int n = post ? small : sourceN;
    double w = op == "baby" || op == "giant" || op == "relin"
                   ? double(l + 1 - num(d, "K")) * (l + 2) / (l + 1)
               : op == "source" || op == "drop"   ? 0
               : op == "add" || op == "add_plain" ? .15
                                                  : 1.;
    total += w * limbWork(n, l);
  }
  return total;
}
DictionaryAttr details(ModuleOp m, DictionaryAttr par, double price,
                       int64_t bits) {
  Builder b(m.getContext());
  auto p = source(m);
  auto c = conv(m);
  auto l = suffix(m);
  require(c && c.getLayoutContract() == "trace-invariant" &&
              l.getInput() == c.getOut() && c.getOut().hasOneUse() &&
              l.getOut().getType().getProfile() ==
                  c.getOut().getType().getProfile(),
          "stale_conversion_ssa");
  auto a = c.getInput().getType(), z = c.getOut().getType();
  // The current emitter converts one value before a one-level linear op.
  // Check the suffix demand against the actual (replanned) SSA availability.
  auto window = commonCutLevels({{l.getOut().getType().getLevel() + 1, a.getLevel()}});
  require(window && z.getLevel() >= window->required && z.getLevel() <= window->current,
          "cut_level_window_empty_or_exceeded");
  auto target =
      SymbolTable::lookupNearestSymbolFrom<ProfileOp>(c, z.getProfile());
  auto base = field<DictionaryAttr>(m, "snabf.pass3");
  require(target &&
              a.getProfile() == FlatSymbolRefAttr::get(p.getSymNameAttr()) &&
              z.getLevel() == a.getLevel() &&
              int64_t(target.getRing()) == z.getN() &&
              target.getQ() == p.getQ().take_front(a.getLevel() + 1) &&
              target.getP() == p.getP() && target.getSlots() == p.getSlots() &&
              target.getDefaultLogScale() == p.getDefaultLogScale() &&
              target.getLayout() == p.getLayout() &&
              target.getKeyId() ==
                  p.getKeyId().str() + "/N" + std::to_string(z.getN()),
          "stale_target_profile_or_key_scope");
  for (auto chain : {target.getQ(), target.getP()})
    for (auto q : chain)
      require(q % (2 * z.getN()) == 1 && pass3::prime(q),
              "invalid_target_ntt_prime");
  auto ctx = concreteSSAContext(m, par);
  auto cut = liveCut(m, l, c, ctx);
  require(cut.reason.empty(), cut.reason);
  auto sc = screen(p, z.getN(), a.getLevel(), bits);
  require(sc.reason.empty(), sc.reason);
  require(target.getKeyBytes() == uint64_t(sc.keyBytes),
          "stale_target_key_size");
  DictionaryAttr e;
  for (auto v : field<ArrayAttr>(par, "edges")) {
    auto d = cast<DictionaryAttr>(v);
    if (field<StringAttr>(d, "op") == "ring_down")
      e = d;
  }
  require(e && num(e, "Level") == a.getLevel() && num(e, "K") == 0,
          "missing_conversion_error_contract");
  double s = real(e, "physical_log_scale");
  auto from = layout(p, a.getN(), a.getLevel(), s,
                     a.getLogical().getNumElements()),
       to = layout(p, z.getN(), z.getLevel(), s,
                   z.getLogical().getNumElements());
  auto tr = transition(Action::Down, from, to, 1, price);
  require(tr.emittable, tr.reason);
  require(c.getProvidedAttr() == attr(b, from) &&
              c.getRequiredAttr() == attr(b, to),
          "stale_slot_layout_contract");
  NamedAttrList d;
  d.set("transition_work", b.getF64FloatAttr(tr.cost));
  d.set("consumed_levels", b.getI64IntegerAttr(0));
  d.set("cut_edges", cut.record.get("ciphertext_use_count"));
  d.set("cut", cut.record);
  d.set("key_domains", ctx.keyDomains);
  d.set("security", sc.record);
  d.set("switch_bytes", b.getI64IntegerAttr(sc.switchBytes));
  d.set("switch_level_q", b.getI64IntegerAttr(a.getLevel()));
  d.set("switch_level_p", b.getI64IntegerAttr(target.getP().size() - 1));
  d.set("provided", attr(b, from).getFields());
  d.set("required", attr(b, to).getFields());
  d.set("op_kind", b.getStringAttr("ring_down"));
  d.set("emittable", b.getBoolAttr(true));
  d.set("compatibility_status", b.getStringAttr("adaptable_via_op"));
  d.set("parallel_factor", b.getI64IntegerAttr(1));
  d.set("transition", e);
  d.set("target_q", target.getQAttr());
  d.set("target_p", target.getPAttr());
  d.set("source_key", p.getKeyIdAttr());
  d.set("target_key", target.getKeyIdAttr());
  auto sourceSecurity = sourceScreenRecord(m, p, base);
  auto srcBits = real(sourceSecurity, "min_bits");
  d.set("source_security", sourceSecurity);
  d.set("source_min_bits", b.getF64FloatAttr(srcBits));
  d.set("target_and_switch_min_bits", b.getF64FloatAttr(sc.bits));
  d.set("security_min_bits", b.getF64FloatAttr(std::min(sc.bits, srcBits)));
  d.set("inventory", field<DictionaryAttr>(m, "snabf.inventory"));
  d.set("cost", b.getF64FloatAttr(cost(par, p.getRing(), z.getN(), price)));
  d.set("baseline_cost", b.getF64FloatAttr(cost(base, p.getRing(), 0, price)));
  return d.getDictionary(m.getContext());
}
void materialize(ModuleOp m, int64_t n, int64_t level, double price,
                 int64_t bits) {
  IRRewriter b(m.getContext());
  auto p = source(m);
  auto l = suffix(m);
  auto a = l.getInput().getType();
  auto from = layout(p, p.getRing(), level, p.getDefaultLogScale(),
                     a.getLogical().getNumElements()),
       to = layout(p, n, level, p.getDefaultLogScale(),
                   a.getLogical().getNumElements());
  auto tr = transition(Action::Down, from, to, 1, price);
  if (!tr.emittable)
    throw RejectedProposal(tr.reason);
  // Key-size comes from Lattigo, not an N-ratio guess. Security admission is
  // rechecked after P3.
  auto sc = screen(p, n, level, bits, false);
  auto nm = "snabf_epoch_" + std::to_string(n);
  b.setInsertionPointAfter(p);
  auto t = ProfileOp::create(
      b, m.getLoc(), nm, n, p.getQ().take_front(level + 1), p.getP(),
      p.getDefaultLogScale(), p.getKeyId().str() + "/N" + std::to_string(n),
      p.getLayout(), p.getSlots(), sc.keyBytes);
  t->setAttr("snabf.target_epoch", b.getUnitAttr());
  auto ref = FlatSymbolRefAttr::get(b.getContext(), nm);
  b.setInsertionPoint(l);
  auto c = ConvertOp::create(
      b, l.getLoc(), CtType::get(b.getContext(), a.getLogical(), n, level, ref),
      l.getInput(), b.getStringAttr("trace-invariant"), attr(b, from),
      attr(b, to));
  c->setAttr("native", l.getInput().getDefiningOp()->getAttr("native"));
  l->setOperand(0, c.getOut());
  l.getOut().setType(
      CtType::get(b.getContext(), a.getLogical(), n, level - 1, ref));
}
} // namespace
bool isMixed(ModuleOp m) {
  return bool(conv(m)) || m->hasAttr("snabf.mixed_parameters") ||
         m->hasAttr("snabf.ring_decision");
}
void requireCheckedRingDown(ModuleOp m) {
  auto trace = m->getAttrOfType<DictionaryAttr>("snabf.ring_search");
  require(bool(trace), "[SNABF_FORCE_DOWN_REJECTED] missing_ring_search");
  if (isMixed(m) && conv(m) && field<StringAttr>(trace, "outcome").getValue() == "ring_down")
    return;
  std::set<std::string> reasons;
  for (Attribute a : field<ArrayAttr>(trace, "rounds"))
    reasons.insert(field<StringAttr>(cast<DictionaryAttr>(a), "reason").str());
  if (reasons.empty()) reasons.insert("no_checked_down_proposal");
  throw ContractError("[SNABF_FORCE_DOWN_REJECTED] " + llvm::join(reasons, ","));
}
static void clearSearch(ModuleOp x) {
  source(x)->removeAttr(policyName);
  for (auto k :
       {"snabf.ring_search", "snabf.mixed_parameters", "snabf.ring_decision",
        "snabf.switch_price", "snabf.ckks_epochs", "snabf.pass4_required",
        "snabf.pass4_effective_switching_key_security_bits",
        "snabf.emit_sha256"})
    x->removeAttr(k);
}
static void restoreBaseline(ModuleOp x) {
  auto c = conv(x);
  require(bool(c), "missing_cut_edge");
  auto lin = suffix(x);
  auto p = source(x);
  auto epoch = SymbolTable::lookupNearestSymbolFrom<ProfileOp>(
      c, c.getOut().getType().getProfile());
  require(epoch && epoch != p && epoch->hasAttr("snabf.target_epoch") &&
              c.getOut().hasOneUse() && lin.getInput() == c.getOut(),
          "invalid_epoch_boundary");
  lin->setOperand(0, c.getInput());
  auto t = c.getInput().getType();
  lin.getOut().setType(CtType::get(x.getContext(), t.getLogical(), t.getN(),
                                   t.getLevel() - 1, t.getProfile()));
  c.erase();
  epoch.erase();
  clearSearch(x);
  planModule(x, false);
  verifyPlan(x);
  verifyPass3(x);
}
// Does not consult the search trace. Used for the selected mixed IR only.
static void verifyMixed(ModuleOp m, int64_t bits) {
  auto p = source(m);
  auto base = field<DictionaryAttr>(m, "snabf.pass3");
  require(p->getAttr("pass3_parameters") == base &&
              base.get("q") == p.getQAttr() && base.get("p") == p.getPAttr() &&
              base.get("scale") == p.getDefaultLogScaleAttr(),
          "stale_source_parameters");
  verifySourceScreen(m, base);
  verifyPlan(m);
  auto par = field<DictionaryAttr>(m, "snabf.mixed_parameters");
  boundaryParameters(m, base, par);
  auto price = field<FloatAttr>(m, "snabf.switch_price").getValueAsDouble();
  auto d = details(m, par, price, bits);
  require(d == m->getAttr("snabf.ring_decision") &&
              real(d, "cost") < real(d, "baseline_cost"),
          "stale_or_unprofitable_ring_decision");
}
struct SearchPolicy {
  int64_t minRing, rounds;
  double price;
  int64_t switchBits;
};
static DictionaryAttr policyAttr(Builder &b, SearchPolicy v) {
  require(v.minRing >= 4 && std::has_single_bit(uint64_t(v.minRing)) &&
              v.rounds > 0 && v.rounds <= 16 && std::isfinite(v.price) &&
              v.price >= 0 && v.switchBits >= 128,
          "invalid_pass4_policy");
  return b.getDictionaryAttr(
      {b.getNamedAttr("min_ring", b.getI64IntegerAttr(v.minRing)),
       b.getNamedAttr("max_iters", b.getI64IntegerAttr(v.rounds)),
       b.getNamedAttr("switch_price", b.getF64FloatAttr(v.price)),
       b.getNamedAttr("source_security_bits", b.getI64IntegerAttr(128)),
       b.getNamedAttr("target_security_bits", b.getI64IntegerAttr(128)),
       b.getNamedAttr("switching_key_security_bits",
                      b.getI64IntegerAttr(v.switchBits))});
}
static SearchPolicy readPolicy(ModuleOp m, bool admitting = true) {
  auto p = field<DictionaryAttr>(source(m), policyName);
  SearchPolicy v{num(p, "min_ring"), num(p, "max_iters"),
                 real(p, "switch_price"),
                 num(p, "switching_key_security_bits")};
  Builder b(m.getContext());
  require(
      m->getAttrOfType<UnitAttr>("snabf.pass4_required") &&
          p == policyAttr(b, v) &&
          p.get("switching_key_security_bits") ==
              m->getAttr("snabf.pass4_effective_switching_key_security_bits") &&
          (!admitting || v.switchBits >= switchFloor(m)),
      "stale_or_weakened_pass4_policy");
  return v;
}
// Always starts with a verified, single-domain input. The verifier invokes
// this on a clone with every derived Pass 4 result removed.
static void searchFromBaseline(ModuleOp m, SearchPolicy policy) {
  auto [minRing, rounds, price, bits] = policy;
  verifyPlan(m);
  verifyPass3(m);
  auto base = field<DictionaryAttr>(m, "snabf.pass3");
  auto p = source(m);
  Builder b(m.getContext());
  auto policyRecord = policyAttr(b, policy);
  auto last = suffix(m);
  auto baseCtx = concreteSSAContext(m, base);
  auto sourceSecurity = sourceScreenRecord(m, p, base);
  auto baseCut = liveCut(m, last, {}, baseCtx);
  auto sourceN =
      int64_t(p.getRing()); // the original body can be replaced below
  double original = cost(base, p.getRing(), 0, price);
  std::vector<Proposal> props;
  std::map<std::string, int> levels;
  std::map<std::string, OwningOpRef<ModuleOp>> valid;
  // No self-reference or dependence on MLIR SSA printer naming / emitter
  // metadata.
  std::string ctx =
      hash(spelling(base) + spelling(baseCut.record) + spelling(policyRecord));
  int maxL = 1;
  {
    auto upstream = last.getInput().getDefiningOp();
    if (auto step = dyn_cast<StepOp>(upstream);
        step && step.getKind() == "ckks.add") {
      for (auto a : step.getInputs())
        if (a.getDefiningOp<PolyEvalOp>())
          upstream = a.getDefiningOp();
    }
    if (auto poly = dyn_cast<PolyEvalOp>(upstream)) {
      for (auto v : field<ArrayAttr>(poly, "poly_frontier"))
        maxL = std::max(maxL, int(poly.getInput().getType().getLevel() -
                                  num(cast<DictionaryAttr>(v), "delta_level")));
    } else
      maxL = last.getInput().getType().getLevel();
  }
  for (int64_t n = p.getRing() / 2; n >= minRing; n /= 2) {
    for (int l = std::max(1, maxL); l >= 1; --l) {
      std::string k = ctx + ":" + std::to_string(n) + ":" + std::to_string(l);
      props.push_back({k, n});
      levels[k] = l;
    }
  }
  std::map<std::string, NamedAttrList> phases;
  // Ring capacity and the SIMD map are invariant across boundary levels.
  // Security and numerical failures are only no-goods for their exact QP.
  std::map<int64_t, std::string> impossibleLayouts;
  auto d = search(
      Evaluation{true, original, {}}, props, rounds, [&](const Proposal &q) {
        auto &log = phases[q.fingerprint];
        auto phase = [&](StringRef s) { log.set("phase", b.getStringAttr(s)); };
        try {
          phase("layout_precheck");
          if (!baseCut.reason.empty())
            throw RejectedProposal(baseCut.reason);
          if (auto it = impossibleLayouts.find(q.target);
              it != impossibleLayouts.end()) {
            log.set("reused_layout_cut", b.getBoolAttr(true));
            return Evaluation{false, 0, it->second};
          }
          OwningOpRef<ModuleOp> scratch(cast<ModuleOp>(m->clone()));
          auto x = *scratch;
          materialize(x, q.target, levels.at(q.fingerprint), price, bits);
          phase("pass2_pass1_replan");
          planModule(x, false);
          verifyPlan(x);
          SmallVector<Attribute> polyIds;
          x.walk([&](PolyEvalOp op) {
            polyIds.push_back(field<StringAttr>(
                field<DictionaryAttr>(op, "poly_selected"), "candidate_id"));
          });
          log.set("poly_ids", b.getArrayAttr(polyIds));
          log.set("route_inventory",
                  field<DictionaryAttr>(x, "snabf.inventory"));
          phase("pass3_boundary_milp_and_replay");
          auto par = boundaryParameters(x, base);
          // Independently replay the concrete result even for a rejected
          // proposal.
          boundaryParameters(x, base, par);
          log.set("parameter_fingerprint",
                  b.getStringAttr(hash(spelling(par))));
          phase("security_and_benefit");
          auto c = conv(x);
          auto trialCtx = concreteSSAContext(x, par);
          auto trialCut = liveCut(x, suffix(x), c, trialCtx);
          log.set("cut", trialCut.record);
          log.set("key_domains", trialCtx.keyDomains);
          if (!trialCut.reason.empty())
            throw RejectedProposal(trialCut.reason);
          auto security = screen(source(x), q.target,
                                 c.getInput().getType().getLevel(), bits);
          log.set("security", security.record);
          if (!security.reason.empty())
            throw RejectedProposal(security.reason);
          for (auto raw : field<ArrayAttr>(par, "edges")) {
            auto e = cast<DictionaryAttr>(raw);
            if (field<StringAttr>(e, "op") != "ring_down")
              continue;
            auto t = c.getInput().getType();
            auto s = real(e, "physical_log_scale");
            auto w = t.getLogical().getNumElements();
            c.setProvidedAttr(attr(b, layout(p, t.getN(), t.getLevel(), s, w)));
            c.setRequiredAttr(attr(b, layout(p, q.target, t.getLevel(), s, w)));
          }
          auto rec = details(x, par, price, bits);
          double score = real(rec, "cost");
          x->setAttr("snabf.mixed_parameters", par);
          x->setAttr("snabf.ring_decision", rec);
          x->setAttr("snabf.switch_price", b.getF64FloatAttr(price));
          if (score < original) {
            valid.emplace(q.fingerprint, std::move(scratch));
          }
          return Evaluation{true, score, "verified"};
        } catch (const RejectedProposal &e) {
          StringRef code(e.what());
          if (code == "period_mismatch" || code == "slot_capacity_exceeded" ||
              code == "unsupported_heir_packing_map" ||
              code == "multiple_cut_edges_not_emittable")
            impossibleLayouts.emplace(q.target, code.str());
          return Evaluation{false, 0, e.what()};
        } catch (const std::runtime_error &e) {
          // A solver infeasibility or numerical budget failure is local to
          // this proposal. Bad input, stale evidence or an internal error is
          // not.
          StringRef why(e.what());
          if (why == "boundary_milp_infeasible" ||
              why == "no joint SiLU/route plan satisfies downstream level, "
                     "approximation and rotation budgets" ||
              why == "no optimal subgroup routing solution" ||
              why == "independent selection exceeds the sink rotation budget" ||
              why == "incompatible physical scales in polynomial DAG" ||
              why.starts_with("unaligned physical add: ") ||
              why.starts_with("boundary_concrete_replay: concrete "
                              "message/scale headroom at ") ||
              why.starts_with("boundary_concrete_replay: arithmetic allowance "
                              "exceeded at ") ||
              why == "boundary_concrete_replay: sink approximation + "
                     "arithmetic bound exceeds target" ||
              why == "boundary_concrete_replay: QP exceeds security envelope")
            return Evaluation{false, 0, e.what()};
          throw;
        }
      });
  SmallVector<Attribute> events;
  for (auto &r : d.trace) {
    NamedAttrList a = phases.at(r.proposal.fingerprint);
    a.set("fingerprint", b.getStringAttr(r.proposal.fingerprint));
    a.set("target", b.getI64IntegerAttr(r.proposal.target));
    a.set("level", b.getI64IntegerAttr(levels.at(r.proposal.fingerprint)));
    a.set("reason", b.getStringAttr(r.result.reason));
    a.set("cached", b.getBoolAttr(r.cached));
    a.set("checked", b.getBoolAttr(r.result.checked));
    if (r.result.checked)
      a.set("cost", b.getF64FloatAttr(r.result.cost));
    events.push_back(a.getDictionary(m.getContext()));
  }
  if (d.target) {
    auto x = *valid.at(d.winner);
    m->setAttrs(x->getAttrs());
    m.getBodyRegion().takeBody(x.getBodyRegion());
  }
  NamedAttrList log;
  log.set("policy", policyRecord);
  log.set("baseline_cut", baseCut.record);
  log.set("source_security", sourceSecurity);
  log.set("key_domains",
          d.target ? field<ArrayAttr>(
                         field<DictionaryAttr>(m, "snabf.mixed_parameters"),
                         "key_domains")
                   : baseCtx.keyDomains);
  log.set("outcome",
          b.getStringAttr(d.target ? "ring_down" : "retain_source_ring"));
  log.set("target", b.getI64IntegerAttr(d.target ? d.target : sourceN));
  log.set("baseline_cost", b.getF64FloatAttr(original));
  log.set("selected_cost", b.getF64FloatAttr(d.cost));
  log.set("limit_reached", b.getBoolAttr(d.limit));
  log.set("rounds", b.getArrayAttr(events));
  auto trace = log.getDictionary(m.getContext());
  source(m)->setAttr(policyName, policyRecord);
  m->setAttr("snabf.pass4_required", b.getUnitAttr());
  m->setAttr("snabf.pass4_effective_switching_key_security_bits",
             b.getI64IntegerAttr(bits));
  m->setAttr("snabf.ring_search", trace);
}

static void replaySearch(ModuleOp m, SearchPolicy policy) {
  auto trace = field<DictionaryAttr>(m, "snabf.ring_search");
  OwningOpRef<ModuleOp> base(cast<ModuleOp>(m->clone()));
  if (isMixed(*base))
    restoreBaseline(*base);
  else
    clearSearch(*base);
  searchFromBaseline(*base, policy);
  require(trace == (*base)->getAttr("snabf.ring_search") &&
              m->getAttr("snabf.ring_decision") ==
                  (*base)->getAttr("snabf.ring_decision") &&
              m->getAttr("snabf.mixed_parameters") ==
                  (*base)->getAttr("snabf.mixed_parameters") &&
              m->getAttr("snabf.switch_price") ==
                  (*base)->getAttr("snabf.switch_price"),
          "stale_or_forged_ring_search");
}
void checkRingSearch(ModuleOp m) {
  if (!m->hasAttr("snabf.ring_search")) {
    bool required =
        isMixed(m) || m->hasAttr("snabf.ckks_epochs") ||
        m->hasAttr("snabf.pass4_required") ||
        m->hasAttr("snabf.pass4_effective_switching_key_security_bits");
    m.walk([&](ProfileOp p) { required |= p->hasAttr(policyName); });
    require(!required, "missing_ring_search");
    return;
  }
  replaySearch(m, readPolicy(m));
}
void checkRings(ModuleOp m) {
  auto policy = readPolicy(m);
  verifyMixed(m, policy.switchBits);
  replaySearch(m, policy);
}
void planRings(ModuleOp m, int64_t minRing, unsigned rounds, double price,
               StringRef audit) {
  // Replanning a previously hardened input cannot weaken its persisted security
  // policy.
  int64_t bits = switchFloor(m);
  if (isMixed(m) || m->hasAttr("snabf.ring_search")) {
    // Authenticate the old decision under its recorded policy, but never emit
    // it when the new environment asks for more. Re-solve with the stronger
    // floor.
    auto old = readPolicy(m, false);
    bits = std::max(bits, old.switchBits);
    if (isMixed(m))
      verifyMixed(m, old.switchBits);
    replaySearch(m, old);
    if (isMixed(m))
      restoreBaseline(m);
    else
      clearSearch(m);
  }
  searchFromBaseline(m, {minRing, rounds, price, bits});
  if (!audit.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(audit, ec);
    require(!ec, "cannot_write_pass4_audit");
    auto j = jsonPass2(field<DictionaryAttr>(m, "snabf.ring_search"));
    if (isMixed(m))
      j["conversion"] =
          jsonPass2(field<DictionaryAttr>(m, "snabf.ring_decision"));
    os << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(j)));
  }
}
} // namespace snabf::ir
