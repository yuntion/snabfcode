#include "pass4/DynamicRingPlanner.h"
#include "pass4/RingSwitchPlan.h"
#include <stdexcept>
using namespace snabf::pass4;
static void ck(bool b) {
  if (!b)
    throw std::runtime_error("Pass4Test failure");
}
int main() {
  auto w = commonCutLevels({{2,5},{3,4}});
  ck(w && w->required==3 && w->current==4);
  w = commonCutLevels({{2,2},{1,4}});
  ck(w && w->required==2 && w->current==2);
  ck(!commonCutLevels({{3,5},{1,2}}) && !commonCutLevels({}) && !commonCutLevels({{-1,4}}));
  SlotLayoutDesc a{16384, 8192, 64, 1, 4096, 2, 38, Packing::PeriodicSingle,
                   {0},   {0},  {}};
  auto b = a;
  b.ring /= 2;
  b.slots /= 2;
  ck(transition(Action::Down, a, b, 1, 1).emittable &&
     compare(b, a).status == Compatibility::Adaptable &&
     transition(Action::Up, b, a, 1, 1).reason == "ring_up_not_emittable");
  auto x = b;
  x.packing=Packing::SingleStrided;auto strided=a;strided.packing=Packing::SingleStrided;
  ck(transition(Action::Down,strided,x,1,1).reason=="layout_transform_not_emittable");x=b;
  x.width = 4097;
  ck(validate(x) == "slot_capacity_exceeded");
  x = b;
  x.period = 33;
  ck(validate(x) == "period_mismatch");
  x = b;
  x.stride = 3;
  ck(validate(x) == "stride_unaligned");
  x = b;
  x.packing = Packing::PackedMulti;
  x.offsets = {0, 32};
  x.active = {0, 1};
  ck(validate(x) == "lane_overlap");
  x = b;
  x.level--;
  ck(compare(a, x).reason == "level_insufficient");
  x = b;
  x.mask = {1, 2};
  ck(compare(a, x).reason == "mask_mismatch");
  ck(!transition(Action::SplitRepack, a, b, 2, 1).emittable);
  int calls = 0;
  auto d = search(Evaluation{true, 20, {}},
                  {{"a", 8}, {"a", 8}, {"b", 4}, {"c", 2}}, 2, [&](auto p) {
                    ++calls;
                    return p.target == 8
                               ? Evaluation{false, 1, "security_rejected"}
                               : Evaluation{true, 15, {}};
                  });
  ck(d.target == 4 && d.limit && d.trace[1].cached && calls == 2);
  ck(search(Evaluation{true, 10, {}}, {{"a", 8}}, 1, [](auto) {
       return Evaluation{true, 10, {}};
     }).target == 0);
  try {
    search({}, {{"x", 4}}, 1, [](auto) {
      return Evaluation{false, 0, "level_insufficient"};
    });
  } catch (const std::runtime_error &) {
    return 0;
  }
  return 1;
}
