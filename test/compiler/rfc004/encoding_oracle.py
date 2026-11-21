"""Predict the fixed Go probes using only integers/rationals; read observations last."""
import argparse
import json
from fractions import Fraction as Q
from pathlib import Path


def rn(x, p):
    x = Q(x)
    if x == 0:
        return x
    if x < 0:
        return -rn(-x, p)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    e -= x < Q(2)**e
    u = Q(2)**(e + 1 - p)
    scaled = x / u
    k, r = divmod(scaled.numerator, scaled.denominator)
    k += (2*r > scaled.denominator or (2*r == scaled.denominator and k % 2))
    return k*u


def coeff(x, scale, p):
    z = rn(abs(x)*scale, p)
    return (-1 if x < 0 else 1) * int(rn(z+Q(1, 2), p))


def join(ma, mb, xa, xb, sign):
    out = max(ma, mb)
    ka = int(rn(Q(out, ma), 128)) if ma < mb else 1
    kb = int(rn(Q(out, mb), 128)) if mb < ma else 1
    return dict(coeff0=ka*ma*xa + sign*kb*mb*xb, out_metadata=str(out))


def expected():
    m = 2**53+1
    e64 = rn(m, 53)
    half = Q(1, 2)
    b = half-Q(1, 2**54)
    a = 2**40
    s = Q(65536)
    rs = rn(rn(s/449, 128)/353, 128)
    one = rn(s/(449*353), 128)
    req = Q(97,3)
    got = rn(req,128)
    d = dict(meta=str(m), float64=str(e64), scale_prec=128,
        coeff0=dict(batch_53_f64=coeff(1,e64,53), batch_60_f64=coeff(1,m,128),
                    coeff_60_f64=coeff(1,e64,53), coeff_53_big128=coeff(1,m,128),
                    coeff_53_big53=coeff(1,m,53), coeff_half=coeff(half,1,53),
                    coeff_neg_half=coeff(-half,1,53), coeff_below_half=coeff(b,1,53),
                    coeff_below_half_big=coeff(b,1,128)),
        near_add=join(a,a+1,1,0,1), near_sub_reverse=join(a+1,a,0,1,-1),
        four_seven=join(4,7,1,0,1), four_nine=join(4,9,1,0,1),
        alignment_trunc=str(int(Q(7,4))), scale_bigint=str(int(Q(7,4)+half)),
        rescale=dict(q=[97,353,449],before=str(s),after=str(rs),after_level=0,
                     drop_scale=str(s),drop_level=0,one_div=str(one)),
        one_third_as_scale=str(rn(Q(1,3),128)),fold_requested=str(req),
        fold_actual_meta=str(got),fold_f64_eff=str(rn(got,53)))
    notes=dict(join_rfc_rhs_only=0,join_required_local_units=(a+1)-d['near_add']['coeff0'],
               join_required_message_units=str(Q(1,a+1)),
               rescale_sequential_minus_one_shot=str(rs-one),
               fold_rho=str(req/32),fold_hi_residual=str(3*got/97-1),
               fold_f64_residual=str(3*rn(got,53)/97-1),
               f64_below_half_quantization_error=str(abs(d['coeff0']['coeff_below_half']-b)))
    assert notes['join_required_local_units'] == 1 and abs(rs-one) == Q(1,2**129)
    assert got != req and rn(got,53) != req and Q(notes['f64_below_half_quantization_error']) > half
    return d, notes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observed',type=Path)
    args=parser.parse_args()
    ex,notes=expected()
    obs=json.loads(args.observed.read_text(encoding='utf-8'))
    canonical=lambda x:json.dumps(x,allow_nan=False,sort_keys=True,separators=(',',':'))
    if canonical(ex)!=canonical(obs):
        raise SystemExit('observed != expected: '+str(sorted(k for k in ex.keys()|obs.keys() if ex.get(k)!=obs.get(k))))
    print(json.dumps(dict(exact_match=True,scope='fixed arithmetic probes, not a certificate',**notes),indent=2))


if __name__=='__main__': main()
