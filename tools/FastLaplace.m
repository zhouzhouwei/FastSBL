function [weights,used,sigma2,errbars,basis,selected,alpha,lambda] = FastLaplace(PHI,y,sigma2,eta,lambda_init, adaptive,optimal,scale, verbose)
% This code implements the fast Laplace algorithm from the following paper:
% [1] S. D. Babacan, R. Molina, A. K. Katsaggelos. “Bayesian Compressive Sensing using Laplace Priors,? 
% submitted for publication, IEEE Transactions on Image Processing, September 2008.
% 
% This code is based on the BCS code available from http://people.ee.duke.edu/~lihan/cs/ from the following paper
% [2] S. Ji, Y. Xue, L. Carin, "Bayesian Compressive Sensing," IEEE Trans. Signal Processing, vol. 56, no. 6, June 2008.
% 
% Please check the accompanying license and the license of [2] before using. 
% Note that this code is not optimized for speed, and it is generalized to investigate different choices of the
% parameters.
%
% Author (aka person to blame): Derin Babacan, sdb@northwestern.edu
% Last modified by: 08/08/08
%
% Usage: 
% [weights,used,sigma2,errbars,basis,alpha,lambda] = FastLaplace(PHI,y,sigma2,eta, lambda_init, adaptive,optimal,scale, verbose)
% Inputs:
%   Required: 
%       PHI: measurement matrix 
%       y:   CS measurements
%   Optional: 
%       sigma2: initial noise variance (default : std(t)^2/1e2)
%       eta:  threshold for stopping the algorithm (default : 1e-8)
%       lambda_init : To set lambda equal to a fixed nonegative value. 
%                     if lambda_init = [], lambda will be computed automatically, which is the suggested method.
%                     lambda_init = 0 corresponds to the BCS algorithm in [2], see [1] for technical details.
%
%   Inputs for Adaptive CS (this part is left unchanged from the BCS code, see [2])
%       adaptive: generate basis for adpative CS? (default: 0)
%       optimal: use the rigorous implementation of adaptive CS? (default: 1)
%       scale: diagonal loading parameter (default: 0.1)
% Outputs:
%   weights:  sparse weights
%   used:     the positions of sparse weights
%   sigma2:   re-estimated noise variance
%   errbars:  one standard deviation around the sparse weights 
%   basis:    if adaptive==1, then basis = the next projection vector, see [2]
%   alpha:    sparse hyperparameters (1/gamma), see [1]
%   lambda:   parameter controlling the sparsity , see [1]

%% Check inputs
if nargin < 2
    error('Not enough inputs');
end
if nargin < 3
    sigma2 =  std(y)^2/1e2;
end
if nargin < 4 
    eta = 1e-8 ;
end
if nargin < 5 
    lambda_init = [];
end
if nargin < 6
    adaptive = 0;
end
if nargin < 7
    optimal = 1;
end
if nargin < 8
    scale = 0.1;
end
if nargin < 9 
    verbose = 0;
end

% find initial alpha
[N,M] = size(PHI);
PHIy = PHI'*y;
PHI2 = sum(PHI.^2)';
ratio = (PHIy.^2)./PHI2;
[maxr,index] = max(ratio);

alpha = PHI2(index)/(maxr-sigma2);
% compute initial mu, Sig, S, Q
phi = PHI(:,index);
Hessian = alpha + phi'*phi/sigma2;
Sig = 1/Hessian;
mu = Sig*PHIy(index)/sigma2;
left = PHI'*phi/sigma2;
S = PHI2/sigma2-Sig*left.^2;
Q = PHIy/sigma2-Sig*PHIy(index)/sigma2*left;

% Keep track of the positions selected during the iterations
selected = index;
deleted = [];
  
max_it = 5000;

for count = 1:max_it

    s = S; q = Q;
    s(index) = alpha.*S(index)./(alpha-S(index));
    q(index) = alpha.*Q(index)./(alpha-S(index));

    if isempty(lambda_init)
        lambda = 2*( length(index) - 1 ) / sum(1./alpha);
    else
        lambda = lambda_init;
    end
   
    A = lambda + s - q.^2;
    B = 2*lambda.*s + s.^2;
    C = lambda.*s.^2;
        
    theta = q.^2-s;
    
    discriminant = B.^2 - 4.*A.*C;
    
    nextAlphas = (-B - sqrt(discriminant) ) ./ (2*A);

    % choose the next alpha that maximizes marginal likelihood
    ml = -inf*ones(1,M);
    
    ig0 = find(theta>lambda);
    
    % indices for reestimation
    [ire,foo,which] = intersect(ig0,index);
    if ~isempty(ire)
        
        Alpha = nextAlphas(ire);
        
        delta = (alpha(which)-Alpha)./(Alpha.*alpha(which));

        ml(ire) = q(ire).^2./ (Alpha + s(ire)) + log(Alpha ./ (Alpha + s(ire))) - lambda./ Alpha ...
            -q(ire).^2./ (alpha(which) + s(ire)) - log(alpha(which) ./ (alpha(which) + s(ire))) + lambda./ alpha(which);
    end
    
    % indices for adding
    iad = setdiff(ig0,ire);
    if ~isempty(iad)
        
        Alpha = nextAlphas(iad);
        ml(iad) = log(Alpha ./ (Alpha + s(iad)) )+ q(iad).^2 ./ (Alpha + s(iad)) - lambda./Alpha;
        which = intersect(deleted,iad);
        ml(which) = -inf;
        
    end
    is0 = setdiff([1:M],ig0);
    % indices for deleting
    [ide,foo,which] = intersect(is0,index);
    if ~isempty(ide)
        
         if length(index) == 1
             ml(ide) = -inf;
         else
             ml(ide) = -q(ide).^2 ./ (alpha(which) + s(ide)) - log( alpha(which) ./(alpha(which) + s(ide))) + lambda./alpha(which);
         end

    end

    [ML(count),idx] = max(ml);

     % check convergence
    if count > 2 && abs(ML(count)-ML(count-1)) < abs(ML(count)-ML(1))*eta
        break;
    end

    % update alphas
    % Choose the basis which results in the largest increase in the
    % likelihood
    which = find(index==idx);
    
    if theta(idx) > lambda
        if ~isempty(which) % reestimate a basis
            
            Alpha = nextAlphas(idx);
            
            Sigii = Sig(which,which); mui = mu(which); Sigi = Sig(:,which);
            delta = Alpha-alpha(which);
            ki = delta/(1+Sigii*delta);
            mu = mu-ki*mui*Sigi;
            Sig = Sig-ki*Sigi*Sigi';
            comm = PHI'*(phi*Sigi)/sigma2;
            S = S + ki*comm.^2;
            Q = Q + ki*mui*comm;
            alpha(which) = Alpha;
            
            if verbose,  fprintf(2,'Reestimate %d..\n',idx);end

        else % add a basis
            
            Alpha = nextAlphas(idx);
            phii = PHI(:,idx); Sigii = 1/(Alpha+S(idx)); mui = Sigii*Q(idx);
            comm1 = Sig*(phi'*phii)/sigma2;
            ei = phii-phi*comm1;
            off = -Sigii*comm1;
            Sig = [Sig+Sigii*comm1*comm1', off; off', Sigii];
            mu = [mu-mui*comm1; mui];
            comm2 = PHI'*ei/sigma2;
            S = S - Sigii*comm2.^2;
            Q = Q - mui*comm2;
            
            index = [index;idx];
            alpha = [alpha;Alpha];
            phi = [phi,phii];
            if verbose,  fprintf(2,'Add %d.. \n',idx); end
            
        end
    else
        if ~isempty(which) && length(index) > 1  % delete a basis
            
            deleted = [deleted idx];
            Sigii = Sig(which,which); mui = mu(which); Sigi = Sig(:,which);
            Sig = Sig-Sigi*Sigi'/Sigii; Sig(:,which) = []; Sig(which,:) = [];
            mu  = mu-mui/Sigii*Sigi; mu(which) = [];
            comm = PHI'*(phi*Sigi)/sigma2;
            S = S + comm.^2/Sigii;
            Q = Q + mui/Sigii*comm;
            
            index(which) = [];
            alpha(which) = [];
            phi(:,which) = [];
            if verbose,  fprintf(2,'Delete %d.. \n',idx); end
        elseif ~isempty(which) && length(index) == 1 
            % Something is wrong, trying to delete the only coefficient
            % that has been added.
            break;
        end
            
    end

    selected = [selected; idx];
        
end
weights	= mu;
used = index;
% re-estimated sigma2
sigma2 = sum((y-phi*mu).^2)/(N-length(index)+alpha'*diag(Sig)); 
errbars = sqrt(diag(Sig));

basis = [];
% generate a basis for adaptive CS?
if adaptive
    if optimal
        [V,D] = eig(Sig);
        [foo,idx] = max(diag(D));
        basis = V(:,idx)';
    else
        temp = phi'*phi/sigma2;
        Sig_inv = temp + scale*mean(diag(temp))*eye(length(used));
        [V,D] = eig(Sig_inv);
        [foo,idx] = min(diag(D));
        basis = V(:,idx)';
    end
end
fprintf(1,'FastLaplace Algorithm converged, # iterations : %d \n',count);



