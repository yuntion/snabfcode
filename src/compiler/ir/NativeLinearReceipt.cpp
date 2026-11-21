#include "NativeLinearRoute.h"
#include "llvm/Support/Format.h"
#include <utility>

namespace snabf::native {
namespace json = llvm::json;
namespace {
template <typename T> json::Array numbers(const T &v) {
  json::Array a;
  for (auto x : v) a.push_back(x);
  return a;
}

json::Object route(const SubgroupRoute &r) {
  json::Array terms, splits;
  for (auto t : r.terms) terms.push_back(json::Array{t.diagonal, t.baby, t.giant});
  for (auto s : r.splits) splits.push_back(json::Array{s.rung, s.stride, s.order, s.index});
  return json::Object{{"terms", std::move(terms)}, {"equivalent_splits", std::move(splits)},
                      {"baby_rotations", numbers(r.babyRotations)}, {"giant_rotations", numbers(r.giantRotations)},
                      {"galois_keys", numbers(r.galoisKeys)}};
}
} // namespace

mlir::FailureOr<json::Object> takeNativeLinearReceipt(NativeLinearProgram &&p) {
  if (!p.domain.valid() || !p.width || p.domain.encodedSlots % p.width ||
      p.inspection.empty() || p.matvecs.empty()) return mlir::failure();
  unsigned ring = 1u << p.domain.logN, slots = p.domain.encodedSlots;
  json::Array firstSlots, sites;
  for (unsigned i = 0; i < p.width; ++i) firstSlots.push_back(i);
  json::Object layout{{"kind", "periodic_single"}, {"rotate_direction", "left"}, {"ring_dimension", ring},
                      {"rotation_modulus", ring / 2}, {"encoded_slots", slots}, {"logical_width", p.width},
                      {"first_slot_for_row", std::move(firstSlots)}, {"replica_stride", p.width},
                      {"replica_count", slots / p.width}};
  for (const auto &m : p.matvecs) {
    json::Array ds, routes;
    for (const auto &[physical, diag] : m.diagonals) {
      json::Array bits;
      for (auto word : diag.rowBits) {
        std::string text;
        llvm::raw_string_ostream(text) << llvm::format_hex(word, m.elementBits / 4 + 2);
        bits.push_back(std::move(text));
      }
      ds.push_back(json::Object{{"signed_offset", diag.signedOffset}, {"physical_offset", physical},
                                {"encoded_offset", physical % slots}, {"row_bits", std::move(bits)}});
    }
    for (const auto &r : m.candidates) routes.push_back(route(r));
    sites.push_back(json::Object{{"result_id", m.resultID}, {"input_id", m.inputID}, {"weight_id", m.weightID},
                                 {"outs_id", m.outsID}, {"outs_kind", m.outsKind}, {"weight_sha256", m.weightSHA256},
                                 {"element_bits", m.elementBits}, {"diagonals", std::move(ds)},
                                 {"route_candidates", std::move(routes)}});
  }
  return json::Object{{"schema", "snabf.native-linear.v1"}, {"stage", "pass1-route-candidates"},
                      {"inspection", std::exchange(p.inspection, {})}, {"layout", std::move(layout)}, {"matvecs", std::move(sites)}};
}
} // namespace snabf::native
