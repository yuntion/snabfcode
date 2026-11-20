#id = affine_map<(i) -> (i)>
module {
  func.func @linear_silu_linear(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
    %w0 = arith.constant dense<[[0.50, -0.25, 0.125, 0.0],
                                [0.0, 0.375, 0.125, 0.25],
                                [-0.25, 0.0, 0.50, 0.125],
                                [0.125, -0.125, 0.25, 0.25]]> : tensor<4x4xf32>
    %w1 = arith.constant dense<[[0.25, 0.50, 0.0, -0.125],
                                [0.375, -0.125, 0.25, 0.0],
                                [0.0, 0.25, -0.25, 0.50],
                                [-0.125, 0.0, 0.50, 0.25]]> : tensor<4x4xf32>
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<4xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
    %h = linalg.matvec ins(%w0, %x : tensor<4x4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xf32>) -> tensor<4xf32>
    // The largest first-layer row L1 norm is 7/8, so x in [-1,1]^4
    // gives h in [-7/8,7/8]^4, strictly inside the activation domain.
    %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"],
        domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}
        ins(%h : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
      ^bb0(%in: f32, %ignored: f32):
        %neg = arith.negf %in : f32
        %exp = math.exp %neg : f32
        %den = arith.addf %exp, %one : f32
        %sig = arith.divf %one, %den : f32
        %silu = arith.mulf %in, %sig : f32
        linalg.yield %silu : f32
    } -> tensor<4xf32>
    %y = linalg.matvec ins(%w1, %a : tensor<4x4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xf32>) -> tensor<4xf32>
    return %y : tensor<4xf32>
  }
}
