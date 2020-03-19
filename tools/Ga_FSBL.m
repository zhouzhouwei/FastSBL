function [x, iter] = Ga_FSBL(y, Phi, paras, InitVal)
% Gamma hyperprior

% setting the initial values 
if nargin < 4
    gamma_init  = 1e-3;
    lambda_init = 1e-3;
    betav = Phi'*y;
else
    gamma_init = InitVal.gamma_init ;
    lambda_init = InitVal.lambda_init ;
    betav = InitVal.beta_init ;
end
threshold1 = paras.threshold ;
delta = paras.delta ;
a = paras.a ;
nmd = paras.normalized;
iters = paras.iters;
a0 = paras.a0 ;
b0 = paras.b0 ;
c0 = paras.c0 ;
d0 = paras.d0 ;
[m,n] = size(Phi) ;
n1 = n - m - 2*a0 + 2 ;
c1 = 2*c0-2 ;

% normlized data if necessary
if nmd ==1
    y_max = max(abs(y)) ;
    y = y/ y_max ;
    Phi_norm = vecnorm(Phi) ;
    Phi = Phi ./ vecnorm(Phi) ;
end

% initilization
D = Phi'*Phi ;
Phiy = Phi'*y ;
gamma = gamma_init*ones(n,1);
lambda = lambda_init ;
% betav = zeros(n,1);
% betav = Phiy ;
% betav = (lambda*eye(n)+D)\Phi'*y;


errs = zeros(iters,1) ;
theta = zeros(n,1);
% lambdas = zeros(iters,1);
% gammas = [];
% thetas = [];

% main loop
for iter = 1:iters
    theta_old = theta ;
    temp = lambda + a*gamma ;
    theta = (gamma./temp).*(a*betav-D*betav+Phiy) ;
    betav = theta ;
    
    % update the hyparameters
    rho = sum(1./temp) ;
    lambda = ( n1 + sqrt(n1^2+4*(norm(y-Phi*betav)^2+2*b0)*rho) ) / (2*rho) ;
    ga_den = a./temp + c1./gamma ;
    gamma = sqrt(2*d0 + abs(theta).^2) ./ sqrt(ga_den) ;
%     gamma(gamma < 1e-4) = 0 ;
    
    % stopping criterion
%     lambdas(iter,1) = lambda ;
%     gammas = [gammas gamma];
%     thetas = [thetas theta];
    if norm(theta)<1e-10       
        fprintf(1,'La-FSBL Algorithm converged to 0, # iterations : %d \n',iter);
        break;
    end
    errs(iter) = norm(theta - theta_old)/norm(theta);
    if errs(iter) <= delta
        fprintf(1,'Ga-FSBL Algorithm converged, # iterations : %d \n',iter);
        break;
    end
end
 
x = theta ;
if nmd ==1
    x = x * y_max ./Phi_norm' ;
end

% prune the small elements
x(abs(x)./norm(x)<threshold1) = 0;

end