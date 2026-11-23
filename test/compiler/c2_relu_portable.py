"""Second proof of the emitted real polynomial, independent of MPFI and libm.

All endpoints/coefficients are exact fractions. Bernstein convex hulls bound
whole cells; only the intermediate stage range is rounded outward (to 160
fractional bits). No sampling, gap around zero or cross-platform native ABI.
"""
from fractions import Fraction as F
from math import comb

Q=1<<160


def hull(bern,depth):
    lo,hi=min(bern),max(bern)
    ends=(min(bern[0],bern[-1]),max(bern[0],bern[-1]))
    # An unresolved extremum contributes its entire hull, also at the cap.
    if (lo,hi)==ends or depth==20: return lo,hi
    row=bern; left=[row[0]]; right=[row[-1]]
    while len(row)>1:
        row=[(a+b)/2 for a,b in zip(row,row[1:])];left.append(row[0]);right.append(row[-1])
    a,b=hull(left,depth+1);c,d=hull(right[::-1],depth+1)
    return min(a,c),max(b,d)


def stage(coeff,axis):
    lo,hi=axis;d=len(coeff)-1
    # Shift x=lo+(hi-lo)t then convert ordinary powers to Bernstein on [0,1].
    shifted=[(hi-lo)**j*sum(coeff[i]*comb(i,j)*lo**(i-j) for i in range(j,d+1)) for j in range(d+1)]
    bern=[sum(shifted[j]*F(comb(k,j),comb(d,j)) for j in range(k+1)) for k in range(d+1)]
    a,b=hull(bern,0)
    return F((a*Q).__floor__(),Q),F((b*Q).__ceil__(),Q)


def portable_proof(stages,axis,budget=F(1,4096)):
    cs=[[F(x) for x in s['coefficients']] for s in stages]; a,b=map(F,axis)
    if not cs or not a<=0<=b or any(not 2<=len(row)<=16 or any(row[::2]) for row in cs):
        raise ValueError('invalid portable ReLU chain/domain')
    def branch(left,right,sign):
        work=[(left,right,0)]; leaves=[]
        while work:
            lo,hi,k=work.pop(); z=(lo,hi)
            for row in cs: z=stage(row,z)
            error=max(abs(x*(y+sign)/2) for x in (lo,hi) for y in z)
            if error<budget: leaves.append((lo,hi,error)); continue
            if k>=32 or len(leaves)+len(work)>=2048:
                raise ValueError('portable enclosure did not establish the full-domain budget')
            m=(lo+hi)/2;work.extend(((m,hi,k+1),(lo,m,k+1)))
        if leaves[0][0]!=left or leaves[-1][1]!=right or any(l[1]!=r[0] for l,r in zip(leaves,leaves[1:])):
            raise ValueError('incomplete closed-half cover')
        return dict(domain=[str(left),str(right)],closed_cells=len(leaves),upper=str(max(l[2] for l in leaves)))
    return dict(method='exact-rational-bernstein-v1',budget=str(budget),left=branch(a,F(0),1),right=branch(F(0),b,-1))
