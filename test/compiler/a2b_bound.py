"""Replay the conditional two-half transfer without any ciphertext or decrypt trace."""
from fractions import Fraction as F
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/tools'))
from exact_scale_replay import need, rn


def replay(original,basis,split):
    n=16384; h=n//4; q=basis['q']; m=F(2**basis['d']); p=F(basis['p'][0]); eps=F(1,4096)
    rows=[F(0)]*(2*h); quadrants=[[set(),set()],[set(),set()]]
    for diag in original['diagonals']:
        k=diag['offset'];w=diag['w64'];need(len(w)==2*h,'invalid weight row')
        for i,x in enumerate(w):
            if x:
                c=(i+k)%(2*h);need(i<original['output_width'] and c<original['input_width'],'weight in padding')
                rows[i]+=F(abs(x),64);quadrants[i//h][c//h].add((c-i)%h)
    norm=max(rows);L=max(sum(map(len,row)) for row in quadrants) if split else len(original['diagonals'])
    def rounding(r):return F(r*(1+r),2)
    def key(r,level,scale):return (sum(F(10*r*r*(v+1),p) for v in q[:level+1])+rounding(r))/scale
    def at(level,scale,amp):
        prod=1
        for v in q[:level+1]:prod*=v
        return 2*scale*amp<prod
    g=m*m/q[3];mag=F(17,256);cap=[]
    x=(rounding(n)+n+F(20*n*(2*n+1),p))/m;cap.append(at(5,m,F(1,2)+x))
    z=2*x+key(n,5,m);cap.append(at(5,m,1+z));pre=z/8+F(n,q[5])*(1+z);cap.append(at(5,m*q[5],F(1,8)+pre))
    u=pre+rounding(n)/m;cap.append(at(4,m,F(1,8)+u));pre=u/4+F(n,q[4])*(F(1,8)+u);cap.append(at(4,m*q[4],F(1,32)+pre))
    v=pre+(rounding(n)+n)/m;cap.extend([at(3,m,F(17,32)+v),at(3,m,F(1,8)+u)])
    e=F(17,32)*u+F(1,8)*v+u*v+key(n,3,m*m);cap.append(at(3,m*m,mag+e));e+=rounding(n)/g;cap.append(at(2,g,mag+e))
    meta=rn(g);transfer=F(1);fresh=down=F(0);linear=meta
    if split:
        # Complementary masks pass the cut error once on ideal slots; both
        # encoding cross terms, both post-mask rounds, fold and down KS survive.
        delta=F(n,q[2]);down=key(n,1,g);transfer+=2*delta;fresh=2*(mag*delta+rounding(n)/g+down)
        cap.extend([at(2,g*q[2],(mag+e)*(1+delta)),at(1,g,mag+fresh+transfer*e)])
        linear=rn(rn(meta*q[2])/q[2])
    r,level=(n//2,1) if split else (n,2);baby=key(r,level,g);delta=F(r,q[level]);cap.append(at(level,g,mag+fresh+transfer*e+baby))
    # Aggregate by original output row, without treating shared keys as independent.
    t=(norm+L*delta)*transfer
    injected=norm*(fresh+baby)+L*delta*(mag+fresh+baby)+L*baby/q[level]
    cap.append(at(level,g*q[level],norm*mag+injected+t*e));injected+=rounding(r)/g;cap.append(at(level-1,g,norm*mag+injected+t*e))
    sink=rn(rn(linear*q[level])/q[level]);ratio=g/sink
    gain=t*ratio;f=injected*ratio+norm*mag*abs(ratio-1)+norm*F(1,196608)
    return dict(cut_scale=str(meta),linear_input_scale=str(linear),sink_scale=str(sink),cut_error=str(e),fresh=str(f),gain=str(gain),
                budget=str((eps-f)/gain),total=str(f+gain*e),down_key_charge=str(down),headroom=all(cap))


if __name__=='__main__':
    import json
    src,plan=(json.loads(Path(arg).read_bytes()) for arg in sys.argv[1:]);fb=plan['feedback']
    for name,split in [('keep',False),('down',True)]:
        need(replay(src,plan['basis'],split)==plan[name] and replay(src,fb['initial'],split)==fb['initial_'+name],'physical replay mismatch')
    print('Replayed both physical transfers on the initial and final integer basis.')
