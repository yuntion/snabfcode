#include "Pipeline.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include <bit>
#include <cmath>

namespace snabf::ir {
// Pinned HEIR custom type/attribute grammar at the LLVM-process boundary. Only
// decode configuration fields; graph structure and builtin attrs stay MLIR
// objects.
static StringRef after(StringRef s, StringRef key) {
  auto p = s.find(key);
  require(p != StringRef::npos, "missing HEIR field " + key);
  return s.drop_front(p + key.size());
}
static int64_t number(StringRef s, StringRef key) {
  auto t = after(s, key).ltrim();
  int64_t n;
  require(!t.consumeInteger(10, n), "invalid integer in " + key);
  return n;
}
static std::vector<int64_t> numbers(StringRef s, StringRef key, char end) {
  auto t = after(s, key).split(end).first;
  SmallVector<StringRef> entries;
  t.split(entries, ',');
  std::vector<int64_t> result;
  for (auto e : entries) {
    e = e.trim();
    int64_t n;
    require(!e.consumeInteger(10, n), "invalid chain in " + key);
    result.push_back(n);
  }
  return result;
}
NativeCT decodeCT(Type t) {
  auto text = spelling(t);
  StringRef s = text;
  require(s.starts_with("!lwe.lwe_ciphertext<"),
          "expected HEIR LWE ciphertext");
  return {number(s, "x**"), number(s, "current = "),
          number(s, "scaling_factor = "), numbers(s, "elements = <", '>'),
          after(s, "key = ").split(", modulus_chain").first.str()};
}
Parameters decodeParameters(Attribute a) {
  auto text = spelling(a);
  StringRef s = text;
  auto n = number(s, "logN = ");
  require(n >= 2 && n <= 30, "invalid logN");
  return {unsigned(n), numbers(s, "Q = [", ']'), numbers(s, "P = [", ']'),
          number(s, "logDefaultScale = ")};
}
std::string weightId(DenseFPElementsAttr weights) {
  auto data = weights.getRawData();
  auto hash = llvm::SHA256::hash(
      ArrayRef<uint8_t>((const uint8_t *)data.data(), data.size()));
  return spelling(weights.getType()) + ":" + llvm::toHex(hash, true);
}
static Operation *storedProducer(Value table) {
  auto load = table.getDefiningOp();
  require(load && name(load) == "preprocessing.load",
          "expected an outlined HEIR table load");
  auto f = load->getParentOfType<func::FuncOp>();
  auto m = f->getParentOfType<ModuleOp>();
  auto tag = field<IntegerAttr>(load, "site_id");
  require(load->getNumOperands() == 1, "invalid HEIR preprocessing load");
  auto arg = dyn_cast<BlockArgument>(load->getOperand(0));
  require(arg && arg.getOwner() == &f.getBody().front(), "unbound preprocessing handle");
  func::FuncOp producer;
  m.walk([&](func::CallOp call) {
    if (call.getCallee() != f.getName()) return;
    auto handle = call.getOperand(arg.getArgNumber());
    auto make = handle.getDefiningOp<func::CallOp>();
    require(handle.hasOneUse() && make && !make.getNumOperands() && make.getNumResults() == 1,
            "unbound preprocessing table producer");
    auto p = m.lookupSymbol<func::FuncOp>(make.getCallee());
    require(p && p.getBody().hasOneBlock() && (!producer || producer == p),
            "ambiguous preprocessing table producer");
    producer = p;
  });
  require(bool(producer), "missing preprocessing table caller");
  auto ret = producer.getBody().front().getTerminator();
  require(ret->getNumOperands() == 1, "expected one preprocessing table");
  Operation *value = nullptr;
  producer.walk([&](Operation *o) {
    if (name(o) == "preprocessing.store" &&
        o->getOperand(1) == ret->getOperand(0) &&
        field<IntegerAttr>(o, "site_id") == tag) {
      require(!value, "ambiguous HEIR preprocessing slot");
      value = o->getOperand(0).getDefiningOp();
    }
  });
  require(value, "unbound HEIR table slot");
  return value;
}
DenseFPElementsAttr preparedWeights(Value table, ArrayRef<int64_t> diagonals) {
  auto *o = storedProducer(table);
  require(name(o) == "kernel.prepare_linear_transform",
          "not a prepared transform");
  require(
      field<DenseI64ArrayAttr>(o, "diagonal_indices").asArrayRef() == diagonals,
      "apply and preparation disagree on diagonals");
  return field<DenseFPElementsAttr>(o->getOperand(0).getDefiningOp(), "value");
}
DenseFPElementsAttr plainWeights(Value table) {
  auto *o = storedProducer(table);
  require(name(o) == "lwe.rlwe_encode", "not an encoded mask");
  return field<DenseFPElementsAttr>(o->getOperand(0).getDefiningOp(), "value");
}

void importCKKS(ModuleOp m, int64_t bytes, double bound, double hoist,
                double giant, double eta) {
  require(bytes > 0 && std::isfinite(bound) && bound > 0,
          "supply physical key bytes and input interval");
  auto p = decodeParameters(field<Attribute>(m, "ckks.schemeParam"));
  auto slots = field<IntegerAttr>(m, "scheme.requested_slot_count").getInt();
  IRRewriter rw(m.getContext());
  StringRef symbol = "e01"; // frozen pre-product IR ABI
  if (m->hasAttr("snabf.initial_profile"))
    symbol = field<StringAttr>(m,"snabf.initial_profile").getValue();
  require(!symbol.empty() && !m.lookupSymbol(symbol), "empty or conflicting profile symbol");
  auto ref = FlatSymbolRefAttr::get(m.getContext(), symbol);
  func::FuncOp eval;
  for (auto f : m.getOps<func::FuncOp>()) {
    auto iface = f->getAttrOfType<DictionaryAttr>("heir.interface");
    if (!iface) continue;
    for (auto a : field<ArrayAttr>(iface, "roles"))
      if (cast<StringAttr>(a).getValue() == "server.evaluate") {
        require(!eval, "expected one server evaluator");
        eval = f;
      }
  }
  require(bool(eval) && eval.getBody().hasOneBlock(),
          "expected one straight-line HEIR server.evaluate");
  auto layout = eval.getArgAttr(0, "tensor_ext.original_type");
  auto info = eval.getArgAttrOfType<DictionaryAttr>(0, "heir.kernel_info");
  require(bool(layout) && bool(info), "missing HEIR logical shape/packing map");
  auto iface = field<DictionaryAttr>(eval, "heir.interface");
  auto entry = m.lookupSymbol<func::FuncOp>(
      field<StringAttr>(iface, "func_name").getValue());
  require(bool(entry), "missing HEIR entry");
  auto types = field<ArrayAttr>(field<DictionaryAttr>(entry, "heir.interface"),
                                "input_types");
  require(types.size() == 1, "expected one packed model input");
  auto logical = cast<RankedTensorType>(cast<TypeAttr>(types[0]).getValue());
  require(logical.getShape() ==
              field<DenseI64ArrayAttr>(info, "result_shape").asArrayRef(),
          "inconsistent logical shape");
  auto argTy = cast<RankedTensorType>(eval.getArgument(0).getType());
  auto input = decodeCT(argTy.getElementType());
  auto n = int64_t(1) << p.logN;
  auto ct = [&](Type native) {
    auto c = decodeCT(native);
    require(c.n == n && c.key == input.key && c.chain == p.q,
            "HEIR ciphertext/profile mismatch");
    return CtType::get(m.getContext(), logical, n, c.level, ref);
  };
  rw.setInsertionPointToStart(m.getBody());
  ProfileOp::create(rw, m.getLoc(), symbol, n, p.q, p.p, p.logScale,
                    entry.getName().str() + "__configure:" + input.key, layout,
                    slots, bytes);
  m->setAttr("snabf.hoist", rw.getF64FloatAttr(hoist));
  m->setAttr("snabf.giant", rw.getF64FloatAttr(giant));
  m->setAttr("snabf.key_units_per_mib", rw.getF64FloatAttr(eta));
  for (auto &op : llvm::make_early_inc_range(eval.getBody().front())) {
    auto *o = &op;
    auto k = name(o);
    rw.setInsertionPoint(o);
    if (k == "tensor.extract" && o->getOperand(0) == eval.getArgument(0)) {
      auto v = o->getResult(0);
      rw.setInsertionPointAfter(o);
      auto src = SourceOp::create(rw, o->getLoc(), ct(v.getType()), v);
      src->setAttr("input_bound", rw.getF64FloatAttr(bound));
      src->setAttr("native", TypeAttr::get(v.getType()));
      v.replaceAllUsesExcept(src.getOut(), src);
      continue;
    }
    if (!llvm::any_of(o->getOperandTypes(),
                      [](Type t) { return isa<CtType>(t); }))
      continue;
    if (k == "tensor.insert") {
      auto v = o->getOperand(0);
      auto native = field<TypeAttr>(v.getDefiningOp(), "native").getValue();
      auto sink = SinkOp::create(rw, o->getLoc(), native, v);
      o->setOperand(0, sink.getCipher());
      continue;
    }
    require(o->getNumResults() == 1, "expected one ciphertext result");
    auto native = o->getResult(0).getType();
    auto out = ct(native);
    Operation *next;
    if (k == "kernel.apply_linear_transform") {
      auto ds = field<DenseI64ArrayAttr>(o, "diagonal_indices");
      auto weights = preparedWeights(o->getOperand(1), ds.asArrayRef());
      require(weights.getType().getShape() ==
                      ArrayRef<int64_t>({ds.size(), slots}) &&
                  field<IntegerAttr>(o, "diagonal_width").getInt() == slots,
              "mismatched diagonal table shape");
      next = LintransOp::create(rw, o->getLoc(), out, o->getOperand(0),
                                o->getOperand(1), ds);
    } else if (k == "kernel.eval_chebyshev")
      next = PolyEvalOp::create(rw, o->getLoc(), out, o->getOperand(0),
                                field<ArrayAttr>(o, "coefficients"));
    else {
      // Transient only: the source-proven erf join must consume the encoded
      // +1, or liftActivationRegions rejects the module before verification.
      const bool erfOffset = k == "ckks.add_plain" && m->hasAttr("snabf.protect_erf") && m->hasAttr("snabf.activation_sources");
      if (!erfOffset && k != "ckks.mul_plain" && k != "ckks.mul" && k != "ckks.add" && k != "ckks.rescale" &&
          k != "ckks.level_reduce" && k != "ckks.relinearize") {
        o->emitError() << "[SNABF_UNSUPPORTED_OP] op=" << k << " reason=no_lossless_encrypted_import";
        throw ContractError(("no lossless import for " + k).str());
      }
      next = StepOp::create(rw, o->getLoc(), out, o->getOperands(), k);
    }
    next->setAttr("native", TypeAttr::get(native));
    next->setAttr("native_props", props(o));
    next->setAttr("native_attrs", o->getDiscardableAttrDictionary());
    rw.replaceOp(o, next->getResults());
  }
}
}  // namespace snabf::ir
