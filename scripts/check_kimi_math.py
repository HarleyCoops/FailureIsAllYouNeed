"""Check the documented algebra, not game performance or new model repairs."""
import numpy as np

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)

def main():
    rng=np.random.default_rng(42)
    dim=32; n=87; lam=100.; target=.7; mu=1.
    u=rng.normal(size=dim); uf=u.copy()
    h=np.tanh(rng.normal(size=dim)); H=np.tanh(rng.normal(size=(n,dim)))
    M=np.eye(dim)+lam*H.T@H/n
    direction=np.linalg.solve(M,h); d=target*direction/(h@direction)
    close(h@d,target)
    W3=rng.normal(size=(dim,5)); new=W3.copy(); new[:,3]-=d/2; new[:,4]+=d/2
    close((h@new)[4]-(h@new)[3]-((h@W3)[4]-(h@W3)[3]),target)
    close(np.linalg.norm(new-W3)**2,np.linalg.norm(d)**2/2)
    close(np.linalg.norm(H@(new-W3))**2,np.linalg.norm(H@d)**2/2)
    feasible=rng.normal(size=dim); feasible-=h*(h@feasible)/(h@h)
    assert (d+feasible)@M@(d+feasible)>=d@M@d
    print('PASS: exact W3 gap, edit norm, probe-logit penalty, constrained optimum.')

    U=rng.normal(size=(n,dim)); D=1-H**2; Df=1-h**2
    w=W3[:,4]-W3[:,3]
    a=(uf@uf)*Df*w; coupling=(U@uf)**2
    A=mu+lam*np.mean(coupling[:,None]*D**2,axis=0)
    v=target*(a/A)/np.sum(a*a/A); delta=np.outer(uf,v)
    close(a@v,target)
    close(w@(Df*(delta.T@uf)),target)
    close(np.sum(A*v*v),mu*(v@v)+lam*np.mean(np.sum((D*(U@delta))**2,axis=1)))
    close(np.linalg.norm(delta)**2,(uf@uf)*(v@v))
    print('PASS: rank-one constraint, diagonal objective, parameter-norm scaling.')

    G=np.stack([((U*D[:,j,None]**2).T@U)/n for j in range(dim)])
    constraint=uf[:,None]*(Df*w)[None,:]
    y=np.column_stack([np.linalg.solve(mu*np.eye(dim)+lam*G[j],constraint[:,j]) for j in range(dim)])
    full=target*y/np.sum(constraint*y)
    close(np.sum(constraint*full),target)
    close(w@(Df*(full.T@uf)),target)
    block_penalty=sum(full[:,j]@G[j]@full[:,j] for j in range(dim))
    close(block_penalty,np.mean(np.sum((D*(U@full))**2,axis=1)))
    assert np.linalg.norm(full)<1  # This fixture tests the solve before clipping.
    z=np.arctanh(h); eps=1e-5
    finite=(np.tanh(z+eps*(full.T@uf))-np.tanh(z-eps*(full.T@uf)))/(2*eps)
    np.testing.assert_allclose(finite,Df*(full.T@uf),atol=1e-10,rtol=1e-7)
    raw=rng.normal(size=(dim,dim))*10
    clipped=raw*min(1,1/np.linalg.norm(raw)); close(np.linalg.norm(clipped),1)
    print('PASS: full W2 block solve, probe penalty, tanh derivative, step cap.')
    print('These are algebra checks only; saved experimental results are unchanged.')

if __name__=='__main__':
    main()
