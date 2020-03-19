function x = EM_SBL(y, Phi, paras)
% the code for paper "Sparse Bayesian Learning for Basis Selection".  
threshold1 = paras.threshold;
iters = paras.iters ;
nmd = paras.normalized ;
delta = paras.delta;
[M,N] = size(Phi) ;
Gamma = 1e-2*eye(N) ;
Hygamma = diag(Gamma);
lambda = 1e-2 ;
lambdas = zeros(iters,1);
errs = zeros(iters,1);
xhat = zeros(N,1);

% normlized data
if nmd ==1
    y_max = max(abs(y)) ;
    y = y/ y_max ;
    Phi_norm = vecnorm(Phi) ;
    Phi = Phi ./ vecnorm(Phi) ;
end

% main loop 
for iter = 1:iters
    xhat_old = xhat;
    %if Phi is over-determined using Woodbury identity to calculate Sigma
    if M > N
       Sigma = inv( diag(1./Hygamma)+Phi'*Phi/lambda ) ;
    else
       Sigma = Gamma - Gamma*Phi'*((lambda*eye(M)+Phi*Gamma*Phi')\Phi)*Gamma; 
    end
    xhat = 1/lambda*Sigma*Phi'*y ;
    
%     %prune the small terms in x
%     xhat(abs(xhat)./norm(abs(xhat))<threshold1) = 0;
    
    %update the hyperparameters 
    temp = sum(1-diag(Sigma)./Hygamma) ;
    Hygamma = xhat.^2 + diag(Sigma) ;
    lambda = (norm(y-Phi*xhat,2)^2+lambda*temp)/M ;
    Gamma = diag(Hygamma) ;
    lambdas(iter,1) = lambda ;

    % stopping criterion 
    errs(iter) = norm(xhat - xhat_old)/norm(xhat);
    if errs(iter) <= delta 
        fprintf(1,'EM-SBL Algorithm converged, # iterations : %d \n',iter);
        break;
    end   
end


x = xhat ;
if nmd ==1
    x = x * y_max ./Phi_norm' ;
end
%prune the small terms in x
x(abs(x)./norm(x)<threshold1) = 0;

end