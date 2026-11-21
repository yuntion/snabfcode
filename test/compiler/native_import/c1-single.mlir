#id = affine_map<(i) -> (i)>
module {
func.func @f88280ebd6b8f89e(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
%zero = arith.constant 0.0 : f32
%one = arith.constant 1.0 : f32
%half = arith.constant 0.5 : f32
%inv = arith.constant 0.7071067811865475244 : f32
%empty = tensor.empty() : tensor<4xf32>
%init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
%w0 = arith.constant dense<[[0.125, 0.078125, -0.109375, -0.140625], [-0.140625, -0.046875, -0.109375, 0.203125], [-0.109375, 0.125, -0.140625, 0.203125], [0.078125, -0.046875, -0.109375, -0.140625]]> : tensor<4x4xf32>
%l0 = linalg.matvec ins(%w0, %x : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
%gelu = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"], domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} ins(%l0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
^bb0(%in: f32, %old: f32):
%z = arith.mulf %in, %inv : f32
%e = math.erf %z : f32
%d = arith.addf %one, %e : f32
%s = arith.mulf %in, %half : f32
%p = arith.mulf %d, %s : f32
linalg.yield %p : f32
} -> tensor<4xf32>
%w1 = arith.constant dense<[[0.125, 0.078125, 0.078125, 0.078125], [0.125, 0.203125, -0.1875, -0.109375], [-0.1875, -0.1875, -0.109375, -0.140625], [0.03125, -0.1875, -0.140625, -0.046875]]> : tensor<4x4xf32>
%l1 = linalg.matvec ins(%w1, %gelu : tensor<4x4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) -> tensor<4xf32>
return %l1 : tensor<4xf32>
}
}
