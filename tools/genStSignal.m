function sig = genStSignal(m,n,nu)
% generate student's t signal with freedom nu  
sig = zeros(n,1);
ids = randperm(n,m);
val = trnd(nu,m,1);
sig(ids) = val ;
