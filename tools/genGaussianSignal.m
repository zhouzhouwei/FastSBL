function sig = genGaussianSignal(m,n)
% generate Gaussian signal
ids = randperm(n,m);
sig = zeros(n,1);
sig(ids) = randn(m,1);

