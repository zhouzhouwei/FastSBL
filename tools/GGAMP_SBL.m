function x = GGAMP_SBL(y, A, paras, sigma_init,gamma_init)
% the code for paper "A GAMP-Based Low Complexity Sparse Bayesian Learning Algorithm"
[m,n] = size(A);
if nargin<4
    gamma = 1*ones(n,1);
    sigma = var(y)/1e2;
else
    gamma = gamma_init*ones(n,1) ;
    sigma = sigma_init ;
end
S = abs(A).^2;
tau_xt = 1*ones(n,1);
stilde = zeros(m,1) ;
xtilde = zeros(n,1);
epsilon1 = paras.delta ;
epsilon2 = paras.delta ;
threshold1 = paras.threshold;
Imax = paras.iters;
Kmax = 100;
errs2 = zeros(Imax,1);

% set theta_x and theta_s following the original paper 
A1 = m*n*1.1*norm(A,2)^2/norm(A,'fro')^2 ;
B1 = 2*(n-m);
C1 = -4*n ;
theta_x = (-B1+sqrt(B1^2-4*A1*C1))/(2*A1) ;
theta_s = theta_x ;

for i=1:Imax
    tau_x =  tau_xt ;
    xhat = xtilde ;
    xtilde_old = xtilde ;
    s = stilde ;
    
    % E-step approximation
%     errs = zeros(Kmax,1);
    for k=1:Kmax
        xhat_old = xhat ;
        tau_p = 1./(S*tau_x) ;
        pk = s + tau_p .*(A*xhat) ;
        tau_s = tau_p ./(1+sigma*tau_p) ;
        s = (1-theta_s)*s + theta_s*(pk./tau_p-y)./(sigma+1./tau_p) ;
        tau_r = 1./(S'*tau_s);
        rk = xhat - tau_r.*(A'*s) ;
        tau_x = tau_r .*gamma./(gamma+tau_r) ;
        xhat = (1-theta_x)*xhat + theta_x*gamma.*rk./(gamma+tau_r);
%         errs(k,1) = norm(xhat-xhat_old)/norm(xhat);
        if norm(xhat-xhat_old)/norm(xhat)<epsilon1
            break;
        end
    end
    % update 
    stilde = s ;
    xtilde = xhat ;
    tau_xt = tau_x ;
    
    % M-step
    gamma_old = gamma ;
    gamma = xtilde.^2 + tau_xt ;
    sigma = (norm(y-A*xtilde)^2 + sigma*sum(1-tau_xt./gamma_old))/m ;
    errs2(i,1) = norm(xtilde-xtilde_old)/norm(xtilde) ;
    if errs2(i,1) < epsilon2
        fprintf(1,'GGAMP-SBL Algorithm converged, # iterations : %d \n',i);
        break;
    end
end

x = xtilde ;
x(abs(x)/norm(x)<threshold1) = 0;
