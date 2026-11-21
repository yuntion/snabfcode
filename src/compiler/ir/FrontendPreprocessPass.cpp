#include "NativeLinalgImportInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include <cmath>

using namespace mlir;
namespace snabf::native {
std::pair<unsigned, unsigned> Source::point(Operation *op) const {
  // Named operations can have parser-created region operations. Attribute
  // their diagnostics to the actual named op; never trust a source loc(...).
  for (; op; op = op->getParentOp())
    if (auto *def = parser.getOpDef(op))
      return mgr.getLineAndColumn(def->loc.Start);
  return {1, 1};
}

json::Object Source::location(Operation *op) const {
  auto [line, column] = point(op);
  return json::Object{{"line", line}, {"column", column}};
}

LogicalResult Source::reject(Operation *op, const llvm::Twine &why) const {
  auto [line, column] = point(op);
  auto file = mgr.getMemoryBuffer(mgr.getMainFileID())->getBufferIdentifier();
  emitError(FileLineColLoc::get(op->getContext(), file, line, column)) << why;
  return failure();
}

LogicalResult Source::attrs(Operation *op, llvm::ArrayRef<StringRef> allowed) const {
  for (auto attr : op->getAttrs())
    if (!llvm::is_contained(allowed, attr.getName().getValue()))
      return reject(op, "[SNABF_NATIVE_ATTR] unsupported attribute " + attr.getName().getValue());
  return success();
}

LogicalResult Source::strictFloatOp(Operation *op) const {
  if (failed(attrs(op, {"fastmath"}))) return failure();
  if (auto a = op->getAttr("fastmath")) {
    auto fm = dyn_cast<arith::FastMathFlagsAttr>(a);
    if (!fm || fm.getValue() != arith::FastMathFlags::none)
      return reject(op, "[SNABF_NATIVE_FLOAT] fast math changes source semantics");
  }
  return success();
}

std::string typeText(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

std::string floatBits(const llvm::APFloat &value) {
  auto bits = value.bitcastToAPInt();
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::format_hex(bits.getZExtValue(), bits.getBitWidth()/4 + 2);
  return text;
}

namespace {
bool onlyKeys(const json::Object &o, llvm::ArrayRef<StringRef> keys) {
  return llvm::all_of(o, [&](const auto &kv) {
    return llvm::is_contained(keys, StringRef(kv.first));
  });
}

// llvm::json validates the grammar but collapses repeated keys. Visit the
// ordered syntax tree as well, comparing JSON-decoded keys (including \u
// escapes). Do not accept a spec that another consumer could read differently.
bool uniqueKeys(llvm::yaml::Node *node) {
  if (!node) return false;
  if (auto *obj = dyn_cast<llvm::yaml::MappingNode>(node)) {
    llvm::StringSet<> seen;
    for (auto &member : *obj) {
      auto *key = dyn_cast_or_null<llvm::yaml::ScalarNode>(member.getKey());
      if (!key) return false;
      auto decoded = json::parse(key->getRawValue());
      if (!decoded) { llvm::consumeError(decoded.takeError()); return false; }
      auto name = decoded->getAsString();
      if (!name || !seen.insert(*name).second || !uniqueKeys(member.getValue())) return false;
    }
  } else if (auto *seq = dyn_cast<llvm::yaml::SequenceNode>(node)) {
    for (auto &element : *seq) if (!uniqueKeys(&element)) return false;
  }
  return true;
}
} // namespace

LogicalResult inspectEntry(const Source &s, ModuleOp module, StringRef specBytes, Entry &e) {
  if (failed(s.attrs(module, {}))) return failure();
  auto &body = module.getBody()->getOperations();
  if (!llvm::hasSingleElement(body) || !(e.func = dyn_cast<func::FuncOp>(body.front()))) {
    for (auto &op : body)
      if (!isa<func::FuncOp>(op)) return s.reject(&op, "[SNABF_NATIVE_MODULE] expected func.func");
    return s.reject(module, "[SNABF_NATIVE_MODULE] expected exactly one function");
  }
  auto f = e.func;
  if (failed(s.attrs(f, {"function_type", "sym_name", "arg_attrs", "res_attrs"}))) return failure();
  if (f.getNumArguments() != 1 || f.getNumResults() != 1 || !f.getBody().hasOneBlock())
    return s.reject(f, "[SNABF_NATIVE_ABI] expected a single defined block, argument and result");
  e.vector = dyn_cast<RankedTensorType>(f.getArgumentTypes().front());
  if (!e.vector || !e.vector.hasStaticShape() || e.vector.getRank() != 1 || e.vector.getEncoding() ||
      !(e.vector.getElementType().isF32() || e.vector.getElementType().isF64()) ||
      !llvm::isPowerOf2_64(e.vector.getDimSize(0)) || e.vector.getDimSize(0) > 1024 ||
      f.getResultTypes().front() != e.vector)
    return s.reject(f, "[SNABF_NATIVE_SHAPE] expected matching static power-of-two f32/f64 vectors (width <= 1024)");
  bool marked = false;
  for (auto attr : cast<FunctionOpInterface>(f.getOperation()).getArgAttrs(0)) {
    if (attr.getName() != "secret.secret" || !isa<UnitAttr>(attr.getValue()))
      return s.reject(f, "[SNABF_NATIVE_ATTR] only the argument unit marker secret.secret is allowed");
    marked = true;
  }
  if (!f.getResultAttrs(0).empty())
    return s.reject(f, "[SNABF_NATIVE_ATTR] result attributes are not supported");

  auto badSpec = [&](const llvm::Twine &why) { return s.reject(f, "[SNABF_NATIVE_INPUT_SPEC] " + why); };
  auto parsed = json::parse(specBytes);
  if (!parsed) return badSpec(llvm::toString(parsed.takeError()));
  llvm::SourceMgr specMgr;
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(specBytes, "InputSpec.json");
  llvm::yaml::Stream syntax(buffer->getMemBufferRef(), specMgr, /*ShowColors=*/false);
  auto doc = syntax.begin();
  if (doc == syntax.end() || !uniqueKeys(doc->getRoot()) || syntax.failed())
    return badSpec("ambiguous or duplicate JSON member name");
  auto *spec = parsed->getAsObject();
  if (!spec || !onlyKeys(*spec, {"schema", "inputs"}) || spec->getString("schema") != "snabf.native-input-spec.v1")
    return badSpec("expected snabf.native-input-spec.v1 and inputs");
  auto *inputs = spec->getArray("inputs");
  auto *port = inputs && inputs->size()==1 ? (*inputs)[0].getAsObject() : nullptr;
  if (!port || !onlyKeys(*port, {"func_symbol", "arg_index", "source_id", "classification", "domain"}) ||
      port->getString("func_symbol") != f.getSymName() || port->getInteger("arg_index") != 0 ||
      port->getString("source_id").value_or("").empty())
    return badSpec("expected one port bound by (func_symbol, arg_index) with a nonempty source_id");
  if (port->getString("classification") != "secret")
    return badSpec(marked ? "secret.secret requires explicit secret classification" : "C-1 requires an explicit secret input");
  json::Value domain = nullptr;
  if (port->get("domain")) {
    auto *a = port->getArray("domain");
    if (!a || a->size() != 2) return badSpec("domain must be [lower,upper] or omitted");
    auto lo = (*a)[0].getAsNumber(), hi = (*a)[1].getAsNumber();
    if (!lo || !hi || !std::isfinite(*lo) || !std::isfinite(*hi) || *lo > *hi)
      return badSpec("domain endpoints must be ordered finite binary64 numbers");
    domain = json::Array{*lo, *hi};
  }
  e.input = json::Object{{"func_symbol", f.getSymName().str()}, {"arg_index", 0}, {"id", "arg0"},
                         {"source_id", port->getString("source_id")->str()}, {"classification", "secret"},
                         {"secret_marker", marked}, {"domain", std::move(domain)},
                         {"logical_type", typeText(e.vector)}, {"layout", "periodic"}, {"period", e.vector.getDimSize(0)}};
  return success();
}
} // namespace snabf::native
