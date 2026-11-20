#include "Pass2Lowering.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace snabf::ir {
namespace {
// The independently built driver does not link the released wheel's CKKS,
// LWE or preprocessing dialects. Reparse their type/attr parameters only at
// this boundary, keeping the prototype's packing, key and chain untouched.
Type type(MLIRContext *ctx, const std::string &s) {
  auto t = parseType(s, ctx);
  require(bool(t), "cannot materialize HEIR type: " + s);
  return t;
}
Attribute attr(MLIRContext *ctx, const std::string &s) {
  auto a = parseAttribute(s, ctx);
  require(bool(a), "cannot materialize HEIR attribute: " + s);
  return a;
}
size_t after(const std::string &s, StringRef field) {
  auto i = s.find(field.str());
  require(i != std::string::npos, "unexpected native type: missing " + field);
  return i + field.size();
}
size_t angleEnd(const std::string &s, size_t first) {
  require(first < s.size() && s[first] == '<', "invalid HEIR type parameter");
  unsigned n = 0;
  for (size_t i = first; i < s.size(); ++i) {
    if (s[i] == '<') ++n;
    if (s[i] == '>' && --n == 0) return i + 1;
  }
  throw ContractError("unterminated HEIR type parameter");
}
std::string parameter(const std::string &s, StringRef field) {
  auto at = after(s, field);
  return s.substr(at, angleEnd(s, at) - at);
}
void integer(std::string &s, StringRef key, int64_t value) {
  auto at = after(s, key);
  auto end = s.find_first_not_of("0123456789", at);
  require(at < s.size() && end != at, "invalid native integer " + key);
  s.replace(at, end - at, std::to_string(value));
}
void setBasis(std::string &s, ArrayRef<int64_t> q, int64_t level) {
  require(level >= 0 && level < int64_t(q.size()), "invalid native Q level");
  std::string rns = "<";
  for (int64_t i = 0; i <= level; ++i) {
    if (i) rns += ", ";
    rns += "!mod_arith.int<" + std::to_string(q[i]) + " : i64>";
  }
  auto at = after(s, "!rns.rns");
  s.replace(at, angleEnd(s, at) - at, rns + ">");
}
Type ctType(Type prototype, int64_t level, int64_t scale, unsigned size = 2) {
  auto d = decodeCT(prototype);
  auto s = spelling(prototype);
  setBasis(s, d.chain, level);
  integer(s, "current = ", level);
  integer(s, "scaling_factor = ", scale);
  const std::string dim = ", size = ";
  if (auto at = s.find(dim); at != std::string::npos) {
    auto end = s.find_first_not_of("0123456789", at + dim.size());
    s.erase(at, end - at);
  }
  if (size != 2) {
    auto at = after(s, "ciphertext_space = ");
    s.insert(angleEnd(s, at) - 1, dim + std::to_string(size));
  }
  return type(prototype.getContext(), s);
}
unsigned ctSize(Type t) {
  auto s = spelling(t);
  auto pos = s.find(", size = ");
  if (pos == std::string::npos) return 2;
  unsigned n;
  auto suffix = StringRef(s).drop_front(pos + StringRef(", size = ").size());
  require(!suffix.consumeInteger(10, n), "invalid ciphertext dimension");
  return n;
}
Attribute cipherRing(Type ct) {
  auto space = parameter(spelling(ct), "ciphertext_space = ");
  return attr(ct.getContext(), "#polynomial.ring" + parameter(space, "ring = "));
}
Type ptType(Type ct, int64_t scale) {
  auto plain = parameter(spelling(ct), "plaintext_space = ");
  integer(plain, "scaling_factor = ", scale);
  return type(ct.getContext(), "!lwe.lwe_plaintext<plaintext_space = " + plain + ">");
}
Attribute encoding(MLIRContext *ctx, int64_t logScale) {
  return attr(ctx, "#lwe.inverse_canonical_encoding<scaling_factor = " +
                       std::to_string(logScale) + ">");
}
Operation *make(IRRewriter &rw, Location loc, StringRef op, ValueRange inputs,
                Type result, ArrayRef<NamedAttribute> fields = {}) {
  OperationState st(loc, op);
  st.addOperands(inputs);
  if (result) st.addTypes(result);
  if (!fields.empty()) st.propertiesAttr = rw.getDictionaryAttr(fields);
  return rw.create(st);
}
// These are unregistered upstream operations with a generic properties dict.
// Rebuild instead of relying on setAttr to modify an opaque inherent property.
Operation *withProps(IRRewriter &rw, Operation *o, ArrayRef<NamedAttribute> fields) {
  OpBuilder::InsertionGuard guard(rw);
  rw.setInsertionPoint(o);
  auto p = NamedAttrList(props(o));
  for (auto f : fields) p.set(f.getName(), f.getValue());
  OperationState st(o->getLoc(), name(o));
  st.addOperands(o->getOperands());
  st.addTypes(o->getResultTypes());
  st.propertiesAttr = p.getDictionary(rw.getContext());
  st.addAttributes(o->getDiscardableAttrDictionary().getValue());
  auto *newOp = rw.create(st);
  rw.replaceOp(o, newOp->getResults());
  return newOp;
}
int64_t nominal(Operation *op) {
  return field<IntegerAttr>(field<BudgetAttr>(op, "budget").getFields(),
                            "nominal_log_scale").getInt();
}

// These are the outlined, one-block HEIR functions supported by the importer.
// Derive the signature from SSA positions, not from a substitution of all
// occurrences of a type (the old input may also be a valid internal type).
void syncSignature(func::FuncOp f) {
  require(f.getBody().hasOneBlock(), "expected one outlined ABI block");
  auto &b = f.getBody().front();
  auto *ret = b.getTerminator();
  require(isa<func::ReturnOp>(ret), "expected outlined ABI return");
  SmallVector<Type> inputs, outputs;
  for (auto arg : b.getArguments()) inputs.push_back(arg.getType());
  for (auto t : ret->getOperandTypes()) outputs.push_back(t);
  f.setFunctionType(FunctionType::get(f.getContext(), inputs, outputs));
}
RankedTensorType repack(Type packed, Type from, Type to) {
  auto t = dyn_cast<RankedTensorType>(packed);
  require(t && t.hasStaticShape() && t.getNumElements() == 1 && t.getElementType() == from,
          "expected one packed ciphertext of the original ABI type");
  return RankedTensorType::get(t.getShape(), to, t.getEncoding());
}

struct Node {
  StringRef op;
  ArrayRef<int64_t> args;
  ArrayRef<double> weights;
  double constant;
  int64_t depth;
};

// In centered log2 units, a multiplication imposes s_v = s_a+s_b+D-log2(q).
// A plaintext-linear node can produce any nearby scale by choosing the actual
// encoding scale of each coefficient. Solve all fanout/add/return constraints
// together: independently rounding each prime to 2^D loses percent-level
// accuracy at joins. No ciphertext metadata is overwritten to fake alignment.
std::vector<double> physicalScales(ArrayRef<Node> nodes, unsigned out,
                                   int64_t level, int64_t D, ArrayRef<int64_t> q) {
  const auto n = nodes.size();
  std::vector<std::vector<double>> a;
  auto equation = [&](size_t v) -> std::vector<double> & {
    auto &row = a.emplace_back(n + 1, 0.0);
    row[v] = 1;
    return row;
  };
  equation(0);
  equation(out);
  for (size_t i = 1; i < n; ++i) {
    auto &v = nodes[i];
    if (v.op == "mul") {
      auto &row = equation(i);
      for (auto arg : v.args) row[arg] -= 1;
      row[n] = D - std::log2(double(q[level - v.depth + 1]));
    } else if (v.op == "add") {
      for (auto arg : v.args) equation(i)[arg] -= 1;
    }
  }
  std::vector<size_t> pivots;
  for (size_t c = 0, r = 0; c < n && r < a.size(); ++c) {
    auto best = r;
    for (size_t j = r + 1; j < a.size(); ++j)
      if (std::abs(a[j][c]) > std::abs(a[best][c])) best = j;
    if (std::abs(a[best][c]) < 1e-12) continue;
    std::swap(a[r], a[best]);
    auto diag = a[r][c];
    for (auto &e : a[r]) e /= diag;
    for (size_t j = 0; j < a.size(); ++j) {
      if (j == r || a[j][c] == 0) continue;
      auto factor = a[j][c];
      for (size_t k = c; k <= n; ++k) a[j][k] -= factor * a[r][k];
    }
    pivots.push_back(c);
    ++r;
  }
  for (size_t i = pivots.size(); i < a.size(); ++i)
    require(std::abs(a[i][n]) < 1e-9,
            "incompatible physical CKKS scales at polynomial addition or output");
  std::vector<double> s(n, double(D));
  for (size_t i = 0; i < pivots.size(); ++i) s[pivots[i]] += a[i][n];
  for (auto &v : s) {
    require(std::isfinite(v) && std::llround(v) == D,
            "selected DAG cannot use this default CKKS scale");
    v = std::exp2(v);
  }
  return s;
}
}  // namespace

Pass2Lowering::Pass2Lowering(ModuleOp m, ProfileOp p, IRRewriter &rewriter)
    : module(m), profile(p), rw(rewriter) {
  m.walk([&](PolyEvalOp poly) {
    if (!poly->hasAttr("activation")) return;
    auto f = poly->getParentOfType<func::FuncOp>();
    require(!eval || eval == f, "multiple Pass 2 evaluator families");
    eval = f;
  });
  if (!eval) return;
  auto name = eval.getName();
  require(name.consume_back("__preprocessed"), "missing outlined Pass 2 evaluator");
  prep = m.lookupSymbol<func::FuncOp>((name + "__preprocessing").str());
  require(prep && prep.getBody().hasOneBlock() && eval.getBody().hasOneBlock() &&
              eval.getNumArguments() == 2, "unsupported preprocessing ABI");
  readStorage = eval.getArgument(1);
  auto *ret = prep.getBody().front().getTerminator();
  require(ret->getNumOperands() == 1, "missing preprocessing table");
  writeStorage = ret->getOperand(0);
  require(readStorage.getType() == writeStorage.getType(), "inconsistent table ABI");
  prep.walk([&](Operation *o) {
    if (ir::name(o) != "preprocessing.store") return;
    require(o->getNumOperands() == 2 && o->getOperand(1) == writeStorage,
            "unsupported preprocessing table layout");
    auto site = field<IntegerAttr>(o, "site_id").getInt();
    require(site >= 0 && stores.emplace(site, o).second, "duplicate preprocessing site");
  });
}

void Pass2Lowering::bindSource(SourceOp src) {
  auto prototype = field<TypeAttr>(src, "native").getValue();
  auto original = decodeCT(prototype);
  auto level = src.getOut().getType().getLevel();
  require(src->getParentOfType<func::FuncOp>() == eval &&
              level >= 0 && level < int64_t(original.chain.size()) &&
              llvm::equal(original.chain, profile.getQ()) &&
              original.logScale == nominal(src) && ctSize(prototype) == 2,
          "invalid planned client input level, scale or profile");
  if (level == original.level) {
    require(src.getCipher().getType() == prototype, "conflicting source levels");
    return;
  }
  auto target = ctType(prototype, level, original.logScale);
  if (src.getCipher().getType() == target) return;  // another extraction of this input
  auto *ext = src.getCipher().getDefiningOp();
  require(ext && name(ext) == "tensor.extract" && ext->getOperand(0) == eval.getArgument(0) &&
              src.getCipher().getType() == prototype, "unexpected evaluator input extraction");
  auto oldPacked = eval.getArgument(0).getType();
  auto packed = repack(oldPacked, prototype, target);

  auto base = field<StringAttr>(field<DictionaryAttr>(eval, "heir.interface"), "func_name").getValue();
  auto entry = module.lookupSymbol<func::FuncOp>(base);
  auto client = module.lookupSymbol<func::FuncOp>((base + "__encrypt__arg0").str());
  require(entry && entry.getBody().hasOneBlock() && entry.getNumArguments() == 1 &&
              entry.getArgument(0).getType() == oldPacked && client &&
              client.getBody().hasOneBlock() && client.getNumArguments() == 2,
          "unexpected client/evaluator input ABI");
  auto inputArg = entry.getArgument(0);
  require(!inputArg.use_empty(), "missing entry-to-evaluator input");
  for (OpOperand &use : inputArg.getUses()) {
    auto call = dyn_cast<func::CallOp>(use.getOwner());
    require(call && call.getCallee() == eval.getName() && use.getOperandNumber() == 0,
            "entry does not forward input to the selected evaluator");
  }

  auto ret = client.getBody().front().getTerminator();
  require(ret->getNumOperands() == 1 && ret->getOperand(0).getType() == oldPacked,
          "unexpected packed result of client encryption");
  auto *pack = ret->getOperand(0).getDefiningOp();
  require(pack && name(pack) == "tensor.from_elements" && pack->getNumOperands() == 1,
          "expected one encrypted tensor element");
  auto *encrypt = pack->getOperand(0).getDefiningOp();
  require(encrypt && name(encrypt) == "lwe.rlwe_encrypt" && encrypt->getNumOperands() == 2 &&
              encrypt->getOperand(1) == client.getArgument(1) && pack->getOperand(0).getType() == prototype,
          "unbound client public-key encryption");
  auto *encode = encrypt->getOperand(0).getDefiningOp();
  require(encode && name(encode) == "lwe.rlwe_encode" &&
              field<IntegerAttr>(encode, "level").getInt() == original.level &&
              encode->getResult(0).getType() == ptType(prototype, original.logScale),
          "client plaintext level/encoding does not match the imported input");
  auto key = spelling(client.getArgument(1).getType());
  require(StringRef(key).starts_with("!lwe.lwe_public_key<") &&
              parameter(key, "key = ") == original.key &&
              parameter(key, "ring = ") == parameter(parameter(spelling(prototype), "ciphertext_space = "), "ring = "),
          "client public key does not match original ciphertext family and ring");
  setBasis(key, original.chain, level);

  for (auto *user : eval.getArgument(0).getUsers()) {
    require(name(user) == "tensor.extract" && user->getNumResults() == 1 &&
                user->getResult(0).getType() == prototype, "unsupported packed input consumer");
    for (auto *use : user->getResult(0).getUsers()) {
      auto other = dyn_cast<SourceOp>(use);
      require(other && other.getOut().getType().getLevel() == level,
              "one client input cannot be encrypted at conflicting source levels");
    }
    user->getResult(0).setType(target);
  }
  eval.getArgument(0).setType(packed);
  inputArg.setType(packed);
  client.getArgument(1).setType(type(rw.getContext(), key));
  withProps(rw, encode, {rw.getNamedAttr("level", rw.getI64IntegerAttr(level))});
  encrypt->getResult(0).setType(target);
  pack->getResult(0).setType(packed);
  for (auto f : {client, entry, eval}) syncSignature(f);
}

Value Pass2Lowering::coefficient(Location loc, double value, int64_t level,
                                 int64_t logScale, double physicalScale, Type ct) {
  require(std::isfinite(value) && physicalScale > 0 && std::isfinite(physicalScale) &&
              std::llround(std::log2(physicalScale)) == logScale,
          "invalid physical plaintext coefficient scale");
  auto t = ptType(ct, logScale);
  auto site = stores.empty() ? 0 : stores.rbegin()->first + 1;
  require(site <= std::numeric_limits<int32_t>::max(), "too many coefficient tables");
  auto id = rw.getNamedAttr("site_id", rw.getI32IntegerAttr(site));
  auto element = rw.getNamedAttr("element_type", TypeAttr::get(t));
  {
    OpBuilder::InsertionGuard guard(rw);
    rw.setInsertionPoint(prep.getBody().front().getTerminator());
    // One public scalar broadcasts to all requested packed slots during Encode.
    auto dataTy = RankedTensorType::get({1}, rw.getF64Type());
    auto splat = DenseElementsAttr::get(dataTy, ArrayRef<double>{value});
    auto data = arith::ConstantOp::create(rw, loc, splat);
    auto plainSpace = parameter(spelling(t), "plaintext_space = ");
    auto ring = attr(rw.getContext(), "#polynomial.ring" + parameter(plainSpace, "ring = "));
    auto *encode = make(rw, loc, "lwe.rlwe_encode", data.getResult(), t,
        {rw.getNamedAttr("encoding", encoding(rw.getContext(), logScale)),
         rw.getNamedAttr("ring", ring), rw.getNamedAttr("level", rw.getI64IntegerAttr(level)),
         rw.getNamedAttr("scale", rw.getI64IntegerAttr(logScale))});
    encode->setAttr("snabf.physical_scale", rw.getF64FloatAttr(physicalScale));
    encode->setAttr("snabf.coefficient_site", id.getValue());
    stores.emplace(site, make(rw, loc, "preprocessing.store",
                          {encode->getResult(0), writeStorage}, {}, {element, id}));
  }
  return make(rw, loc, "preprocessing.load", readStorage, t, {element, id})->getResult(0);
}

Value Pass2Lowering::emit(PolyEvalOp poly) {
  require(eval && poly->getParentOfType<func::FuncOp>() == eval,
          "unbound Pass 2 evaluator");
  auto plan = field<DictionaryAttr>(poly, "poly_selected");
  auto raw = field<ArrayAttr>(plan, "nodes");
  auto output = field<IntegerAttr>(plan, "output").getInt();
  auto input = poly->getOperand(0);
  auto in = decodeCT(input.getType());
  int64_t D = profile.getDefaultLogScaleAttr().getInt();
  require(output >= 0 && output < int64_t(raw.size()) && in.logScale == D &&
              nominal(poly) == D && ctSize(input.getType()) == 2,
          "invalid Pass 2 input, output index or nominal scale");
  SmallVector<Node> nodes;
  for (auto a : raw) {
    auto d = dyn_cast<DictionaryAttr>(a);
    require(bool(d), "invalid selected polynomial node");
    Node v{field<StringAttr>(d, "op").getValue(),
           field<DenseI64ArrayAttr>(d, "args").asArrayRef(),
           field<DenseF64ArrayAttr>(d, "weights").asArrayRef(),
           field<FloatAttr>(d, "constant").getValueAsDouble(),
           field<IntegerAttr>(d, "depth").getInt()};
    require(std::isfinite(v.constant) && llvm::all_of(v.weights, [](double w) {
              return std::isfinite(w);
            }), "nonfinite selected coefficient");
    int64_t parentDepth = 0;
    for (auto j : v.args) {
      require(j >= 0 && j < int64_t(nodes.size()), "non-topological polynomial DAG");
      parentDepth = std::max(parentDepth, nodes[j].depth);
    }
    bool mul = v.op == "mul", linear = v.op == "linear", add = v.op == "add";
    require((nodes.empty() ? v.op == "input" && v.args.empty()
                           : (mul || linear || add) && !v.args.empty()) &&
                (!mul || v.args.size() == 2) &&
                (linear ? v.weights.size() == v.args.size() : v.weights.empty()) &&
                (linear || add || v.constant == 0) &&
                v.depth == parentDepth + (mul || linear) && v.depth <= in.level,
            "invalid polynomial operation, arity or depth");
    if (mul || linear)
      require(std::llround(std::log2(double(in.chain[in.level - parentDepth]))) == D,
              "selected DAG needs a one-prime rescale at the default log-scale");
    nodes.push_back(v);
  }
  require(nodes[output].depth == field<IntegerAttr>(plan, "delta_level").getInt() &&
              in.level - nodes[output].depth == poly.getOut().getType().getLevel(),
          "selected polynomial result level does not match logical SSA");
  auto scales = physicalScales(nodes, output, in.level, D, in.chain);
  auto loc = poly.getLoc();
  auto primitive = [&](StringRef op, ValueRange args, int64_t level, int64_t scale,
                       unsigned size = 2, ArrayRef<NamedAttribute> fields = {}) {
    return make(rw, loc, op, args, ctType(input.getType(), level, scale, size), fields)
        ->getResult(0);
  };
  SmallVector<Value> values{input};
  for (size_t i = 1; i < nodes.size(); ++i) {
    const auto &v = nodes[i];
    bool mul = v.op == "mul", linear = v.op == "linear";
    auto endLevel = in.level - v.depth;
    auto startLevel = endLevel + (mul || linear);
    SmallVector<Value> args;
    for (auto j : v.args) {
      Value x = values[j];
      auto drops = in.level - nodes[j].depth - startLevel;
      if (drops)
        x = primitive("ckks.level_reduce", x, startLevel, D, 2,
                      {rw.getNamedAttr("levelToDrop", rw.getI64IntegerAttr(drops))});
      args.push_back(x);
    }
    Value acc;
    if (mul) {
      acc = primitive("ckks.mul", args, startLevel, 2 * D, 3);
      acc = primitive("ckks.relinearize", acc, startLevel, 2 * D, 2,
                      {rw.getNamedAttr("from_basis", rw.getDenseI32ArrayAttr({0, 1, 2})),
                       rw.getNamedAttr("to_basis", rw.getDenseI32ArrayAttr({0, 1}))});
    } else {
      for (size_t k = 0; k < args.size(); ++k) {
        Value term = args[k];
        if (linear) {
          // All products in this sum have the same physical scale, even when
          // two parent ciphertexts arose from different multiplication paths.
          double ptScale = (scales[i] / scales[v.args[k]]) * in.chain[startLevel];
          auto w = coefficient(loc, v.weights[k], startLevel, D, ptScale, input.getType());
          term = primitive("ckks.mul_plain", {term, w}, startLevel, 2 * D);
        }
        acc = acc ? primitive("ckks.add", {acc, term}, startLevel, linear ? 2 * D : D)
                  : term;
      }
    }
    if (mul || linear) {
      auto out = ctType(input.getType(), endLevel, D);
      acc = make(rw, loc, "ckks.rescale", acc, out,
                 {rw.getNamedAttr("to_ring", cipherRing(out))})->getResult(0);
    }
    if (v.constant != 0) {
      Value c = coefficient(loc, v.constant, endLevel, D, scales[i], input.getType());
      acc = primitive("ckks.add_plain", {acc, c}, endLevel, D);
    }
    values.push_back(acc);
  }
  return values[output];
}

void Pass2Lowering::bindTable(Value table, int64_t level, bool prepared) {
  Operation *load = table.getDefiningOp();
  require(load && name(load) == "preprocessing.load" && load->getNumOperands() == 1 &&
              load->getOperand(0) == readStorage, "expected an outlined table load");
  auto id = field<IntegerAttr>(load, "site_id").getInt();
  auto it = stores.find(id);
  require(it != stores.end(), "unbound outline table site");
  auto [pos, inserted] = boundLevels.emplace(id, level);
  require(inserted || pos->second == level,
          "one preprocessing table used at conflicting levels");
  auto *store = it->second;
  auto *producer = store->getOperand(0).getDefiningOp();
  require(producer && name(producer) == (prepared ? "kernel.prepare_linear_transform"
                                                   : "lwe.rlwe_encode"),
          "inconsistent prepared table producer");
  if (prepared) {
    auto s = spelling(producer->getResult(0).getType());
    integer(s, "level = ", level);
    auto t = type(rw.getContext(), s);
    producer->getResult(0).setType(t);
    auto f = rw.getNamedAttr("element_type", TypeAttr::get(t));
    if (field<TypeAttr>(store, "element_type").getValue() != t)
      it->second = withProps(rw, store, {f});
    if (field<TypeAttr>(load, "element_type").getValue() != t) {
      load->getResult(0).setType(t);
      withProps(rw, load, {f});
    }
  } else if (field<IntegerAttr>(producer, "level").getInt() != level) {
    withProps(rw, producer, {rw.getNamedAttr("level", rw.getI64IntegerAttr(level))});
  }
}

Type Pass2Lowering::nativeResult(Operation *op, NamedAttrList &fields) {
  auto prototype = field<TypeAttr>(op, "native").getValue();
  auto level = cast<CtType>(op->getResult(0).getType()).getLevel();
  auto scale = nominal(op);
  auto in = decodeCT(op->getOperand(0).getType());
  auto result = ctType(prototype, level, scale, ctSize(prototype));
  if (auto lin = dyn_cast<LintransOp>(op)) {
    require(level == in.level - 1 && scale == in.logScale,
            "lintrans requires one rescale and preserves input scale");
    bindTable(lin.getTable(), in.level, /*prepared=*/true);
  } else if (auto step = dyn_cast<StepOp>(op)) {
    if (step.getKind() == "ckks.add") {
      require(op->getNumOperands() == 2, "expected binary residual add");
      auto rhs = decodeCT(op->getOperand(1).getType());
      require(level == std::min(in.level, rhs.level) && in.logScale == scale &&
                  rhs.logScale == scale, "residual add level or scale disagrees with plan");
      for (OpOperand &edge : op->getOpOperands()) {
        auto ct = edge.get();
        auto t = ct.getType();
        auto drops = decodeCT(t).level - level;
        auto aligned = drops ? ctType(t, level, scale, ctSize(t)) : t;
        require(aligned == result, "residual add has incompatible native ciphertext types");
        // Align only this use. Fanout users may still need the original level.
        if (drops)
          edge.set(make(rw, op->getLoc(), "ckks.level_reduce", ct, aligned,
                        {rw.getNamedAttr("levelToDrop", rw.getI64IntegerAttr(drops))})
                       ->getResult(0));
      }
    } else if (step.getKind() == "ckks.rescale") {
      require(in.level == level + 1 && scale == in.logScale -
                  std::llround(std::log2(double(in.chain[in.level]))),
              "suffix rescale inconsistent with selected native type");
      fields.set("to_ring", cipherRing(result));
    } else if (step.getKind() == "ckks.level_reduce") {
      require(level < in.level && scale == in.logScale, "invalid suffix level reduction");
      fields.set("levelToDrop", rw.getI64IntegerAttr(in.level - level));
    } else if (step.getKind() == "ckks.mul_plain" || step.getKind() == "ckks.add_plain") {
      bindTable(op->getOperand(1), in.level, /*prepared=*/false);
    }
  }
  return result;
}

void Pass2Lowering::bindSink(SinkOp sink) {
  require(!oldSink && sink->getParentOfType<func::FuncOp>() == eval,
          "Pass 2 expects one evaluator sink");
  oldSink = sink->getResult(0).getType();
  newSink = sink->getOperand(0).getType();
  auto d = decodeCT(newSink), orig = decodeCT(oldSink);
  require(d.n == orig.n && d.key == orig.key && d.chain == orig.chain &&
              d.logScale == profile.getDefaultLogScaleAttr().getInt(),
          "invalid Pass 2 result ABI, key family or scale");
  if (newSink != oldSink) {
    auto *ret = eval.getBody().front().getTerminator();
    require(ret->getNumOperands() == 1, "expected one evaluator result");
    auto *insert = ret->getOperand(0).getDefiningOp();
    require(insert && name(insert) == "tensor.insert" && insert->getNumOperands() >= 2 &&
                insert->getOperand(0) == sink.getCipher(), "sink is not the returned packed element");
    auto dest = insert->getOperand(1);
    require(dest.getDefiningOp() && name(dest.getDefiningOp()) == "tensor.empty" &&
                dest.hasOneUse() && dest.getType() == ret->getOperand(0).getType(),
            "unsupported shared output packing buffer");
    auto packed = repack(dest.getType(), oldSink, newSink);
    dest.setType(packed);
    insert->getResult(0).setType(packed);
  }
  sink->getResult(0).setType(newSink);
}

void Pass2Lowering::finish() {
  require(oldSink && newSink, "missing selected sink");
  llvm::SetVector<Type> elements;
  for (auto [_, store] : stores) elements.insert(store->getOperand(0).getType());
  std::string storage = "!preprocessing.storage<";
  for (size_t i = 0; i < elements.size(); ++i) {
    if (i) storage += ", ";
    storage += spelling(elements[i]);
  }
  auto oldStorage = writeStorage.getType();
  auto newStorage = type(rw.getContext(), storage + ">");
  auto newCT = decodeCT(newSink);
  auto translate = [&](Type t, bool output, bool decrypt) -> Type {
    if (t == oldStorage) return newStorage;
    if (output) {
      if (t == oldSink) return newSink;
      if (auto rt = dyn_cast<RankedTensorType>(t); rt && rt.getElementType() == oldSink)
        return repack(rt, oldSink, newSink);
    }
    if (decrypt && isa<OpaqueType>(t)) {
      auto s = spelling(t);
      if (StringRef(s).starts_with("!lwe.lwe_secret_key<")) {
        setBasis(s, newCT.chain, newCT.level);
        return type(rw.getContext(), s);
      }
      if (StringRef(s).starts_with("!lwe.lwe_plaintext<"))
        return ptType(newSink, newCT.logScale);
    }
    return t;
  };
  auto base = eval.getName().drop_back(StringRef("__preprocessed").size());
  for (auto f : module.getOps<func::FuncOp>()) {
    bool decrypt = f.getName() == (base + "__decrypt__result0").str();
    if (f != eval && f != prep && f.getName() != base && !decrypt) continue;
    f.walk([&](Block *b) {
      for (auto a : b->getArguments()) a.setType(translate(a.getType(), decrypt, decrypt));
    });
    SmallVector<Operation *> decodes;
    f.walk([&](Operation *op) {
      auto call = dyn_cast<func::CallOp>(op);
      bool evalCall = f.getName() == base && call && call.getCallee() == eval.getName();
      // Native arithmetic was already typed at its individual SSA definition.
      // Only the wrapper return and decrypt interface inherit the sink type.
      for (Value v : op->getResults()) v.setType(translate(v.getType(), decrypt || evalCall, decrypt));
      if (decrypt && name(op) == "lwe.rlwe_decode") decodes.push_back(op);
    });
    for (auto *op : decodes)
      withProps(rw, op, {rw.getNamedAttr("encoding", encoding(rw.getContext(), newCT.logScale))});
    syncSignature(f);
  }
}

llvm::json::Array pass2ScaleBindings(ModuleOp m, int64_t expected) {
  llvm::json::Array records;
  int64_t marked = 0;
  llvm::SmallDenseSet<int64_t> sites;
  m.walk([&](Operation *op) {
    if (name(op) != "lattigo.ckks.encode" ||
        !op->getParentOfType<func::FuncOp>().getName().ends_with("__preprocessing")) return;
    auto *alloc = op->getOperand(2).getDefiningOp();
    require(alloc && name(alloc) == "lattigo.ckks.new_plaintext", "unbound CKKS encoding");
    auto log = field<IntegerAttr>(op, "scale").getInt();
    llvm::json::Object rec{{"Level", field<IntegerAttr>(alloc, "level").getInt()},
                           {"LogScale", log}};
    if (auto s = op->getAttrOfType<FloatAttr>("snabf.physical_scale")) {
      auto id = field<IntegerAttr>(op, "snabf.coefficient_site").getInt();
      require(sites.insert(id).second && s.getValueAsDouble() > 0 &&
                  std::llround(std::log2(s.getValueAsDouble())) == log,
              "invalid bound physical coefficient scale");
      rec["Scale"] = s.getValueAsDouble();
      ++marked;
    }
    records.push_back(std::move(rec));
  });
  require(marked == expected, "lost physical coefficient scale across HEIR conversion");
  return records;
}
}  // namespace snabf::ir
