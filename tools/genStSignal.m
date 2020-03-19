function sig = genStSignal(m,n,nu)
% generate spike signal 
sig = zeros(n,1);
ids = randperm(n,m);
val = trnd(nu,m,1);
sig(ids) = val ;
% sig(abs(sig)./norm(sig)<1e-3) = 0 ;