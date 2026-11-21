"""Standalone Python 3: independent cleartext checks for the RFC-004 C review."""


def rot(x, k):
    return [x[(i + k) % len(x)] for i in range(len(x))]


def add(a, b):
    return [i + j for i, j in zip(a, b)]


def main():
    x = [1, 2, 4, 8, 16, 32, 64, 128]
    # 2x3 in slots 0..5; 6 and 7 are deliberately nonzero sentinels.
    tree = add(x, rot(x, 1))
    tree = add(tree, rot(tree, 2))
    got = [tree[0], tree[3]]
    row_sums = [sum(x[:3]), sum(x[3:6])]
    col_sums = [x[i] + x[i + 3] for i in range(3)]
    assert (row_sums, got, col_sums) == ([7, 56], [15, 120], [9, 18, 36])
    print('sum axis=1:', row_sums, 'unmasked doubling:', got, 'sum axis=0:', col_sums)
    corner = [x[0]]                       # NHWC 2x2 window, top=left=1
    print('avg top-left (exclude padding):', sum(corner)/len(corner), 'incorrect /4:', sum(corner)/4)
    assert sum(corner)/len(corner) == 1 and sum(corner)/4 == 0.25

    # The only secret operand of matvec is zero, but outs is not zero.
    m, init = [[1, 2, 3], [-1, 0, 5]], [7, 11]
    v = [0, 0, 0]
    mv = [sum(a*b for a, b in zip(r, v)) for r in m]
    assert add(mv, init) == [7, 11] and mv == [0, 0]
    print('matvec(0) with outs:', add(mv, init), 'bare lintrans:', mv)

    # A valid linalg.generic computes a*b; it is not linear in the joint inputs.
    f = lambda a, b: a*b
    assert f(1+0, 0+1) != f(1, 0) + f(0, 1)
    print('generic multiply: f(1,1) =', f(1,1), 'f(1,0)+f(0,1) =', f(1,0)+f(0,1))


if __name__ == '__main__':
    main()
