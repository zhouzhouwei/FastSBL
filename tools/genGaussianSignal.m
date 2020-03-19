function sig = genGaussianSignal(m,n)

ids = randperm(n,m);
sig = zeros(n,1);
sig(ids) = randn(m,1);
% sig(abs(sig)./norm(sig)<1e-3) = 0 ;
