function sig = genSpikeSignal(m,n)
% generate spike signal 
sig = zeros(n,1);
ids = randperm(n,m);
val = randi(2,m,1)-1.5 ;
sig(ids) = val*2;
