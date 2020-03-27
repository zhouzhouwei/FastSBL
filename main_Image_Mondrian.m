clear ; clc ;
addpath(genpath('SBL_Tipping2001\'))
addpath(genpath('SparseLab2.1-Core'))
addpath('tools\')
rand_state = rng(0,'v5normal') ;

% load image data
I0 = imread('Dataset\Mondrian.tif');
I0 = double(I0);

% experimental setting
num_method = 5 ;     nums = 10 ;
% filename = 'ResultsData\Image_Reconstruction\MondrianRes5_5.mat' ; 

% Set finest, coarsest scales
j1 = 7;
j0 = 3;
c = 0.65 ;
qmf = MakeONFilter('Symmlet',8);
diary(['Image_Mondrian_',num2str(c),'.txt'])
diary on

% parameters
paras.a0 = 1e-6 ;       paras.b0 = 1e-6 ;
paras.c0 = 1+1e-6;     paras.d0 = 1e-6 ;
paras.e0 = 1e-2 ;
paras.iters = 5000 ;    paras.threshold = 1e-3;
paras.delta = 1e-4 ;   % for stopping criterion
paras.normalized = 1 ;
epsilon = 1e-4 ;
tau0 = 1e-2 ;
threshold = paras.threshold ;


% Do Hybrid-CS scheme:
% Sample 4^j0 resume coefficients (coarse-scale
% coeffs) at scale 2^(-j0) x 2^(-j0)
alpha0 = FWT2_PO(I0, j0, qmf) ;
alpha_BCS0 = zeros(size(alpha0));
alpha_BCS0(1:2^j0,1:2^j0) = alpha0(1:2^j0,1:2^j0);
alpha_LIN = zeros(size(alpha0));
alpha_LIN(1:2^j1,1:2^j1) = alpha0(1:2^j1,1:2^j1);
I_LIN = IWT2_PO(alpha_LIN, j0, qmf);
err_linear = norm(I0 - I_LIN,'fro') / norm(I0,'fro');


% Construct the vector theta of detail wavelet
% coeffs on scales j0 <= j < j1
theta1 = alpha0((2^j0+1):2^j1,1:2^j0);
theta2 = alpha0(1:2^j1,(2^j0+1):2^j1);
n1 = numel(theta1);
n2 = numel(theta2);
theta = [theta1(:); theta2(:)];
N = 4^j1 - 4^j0;
m = floor(c*N);
M = m + 4^j0 ;


% storage the results
time_SBL = zeros(nums, num_method);
errs = zeros(nums, num_method+1);
Iterations = zeros(nums,2);
Nzeros_num = zeros(nums,num_method+1) ;
Irecon = cell(nums, num_method+1);
Irecon{1,num_method+1} = I_LIN ;
errs(:,end) = err_linear ;
theta_all = cell(nums,1);
theta_hat = zeros(length(theta),num_method);

% main loop
for kk = 1:nums
    
    fprintf('The %d th experiment:\n',kk) ;
    % generate matrix Phi 
    Phi = MatrixEnsemble(m, N, 'USE');
    
    % generate the vector S (random measurments of detail coeffs)
    y_noise = Phi * theta;
    if paras.normalized==0
        a = max(eig(Phi'*Phi)) + epsilon;
        InitVal.beta_init  = Phi'*y_noise;
        InitVal.lambda_init= var(y_noise)/100 ;
    else
        a = max(eig(normc(Phi)'*normc(Phi))) + epsilon ;
        InitVal.beta_init  = normc(Phi)'*y_noise ;
        y_n = y_noise/max(abs(y_noise)) ;
        InitVal.lambda_init= var(y_n)/100 ;
    end
    paras.a = a;
    InitVal.gamma_init = 1/m^2 ;
    
    
    % method 1
    ii = 1;   % index of method
    tic
    [temp,Iterations(kk,1)] = Ga_FSBL(y_noise, Phi, paras, InitVal) ;
    time_SBL(kk,ii) = toc ;
    alpha_BCS = alpha_BCS0;
    alpha_BCS((2^j0+1):2^j1,1:2^j0) = reshape(temp(1:n1), 2^j1-2^j0, 2^j0);
    alpha_BCS(1:2^j1,(2^j0+1):2^j1) = reshape(temp(n1+1:n1+n2), 2^j1, 2^j1-2^j0);
    % Reconstruct
    I_BCS = IWT2_PO(alpha_BCS, j0, qmf);
    % compute error
    errs(kk,ii) = norm(I0 - I_BCS,'fro') / norm(I0,'fro');
    Nzeros_num(kk,ii) = length(nonzeros(temp));
    theta_hat(:,ii) = temp;
    Irecon{kk,ii} = I_BCS ;
    
    
    % method 2
    ii = ii+1 ;
    tic
    [temp,Iterations(kk,2)] = La_FSBL(y_noise, Phi, paras, InitVal) ;
    time_SBL(kk,ii) = toc;
    alpha_BCS = alpha_BCS0;
    alpha_BCS((2^j0+1):2^j1,1:2^j0) = reshape(temp(1:n1), 2^j1-2^j0, 2^j0);
    alpha_BCS(1:2^j1,(2^j0+1):2^j1) = reshape(temp(n1+1:n1+n2), 2^j1, 2^j1-2^j0);
    % Reconstruct
    I_BCS = IWT2_PO(alpha_BCS, j0, qmf);
    % compute error
    errs(kk,ii) = norm(I0 - I_BCS,'fro') / norm(I0,'fro');
    Nzeros_num(kk,ii) = length(nonzeros(temp));
    theta_hat(:,ii) = temp;
    Irecon{kk,ii} = I_BCS ;
    
    
    % method 3
    ii = ii+1 ;
    sigma2 = var(y_noise)/1e2 ;
    delta_La = 1e-8 ;
    tic
    [weights,used_ids] = FastLaplace(Phi,y_noise, sigma2, delta_La);
    time_SBL(kk,ii) = toc;
    temp = zeros(N,1);
    temp(used_ids) = weights ;
    temp(abs(temp)./norm(temp)<threshold) = 0 ;
    alpha_BCS = alpha_BCS0;
    alpha_BCS((2^j0+1):2^j1,1:2^j0) = reshape(temp(1:n1), 2^j1-2^j0, 2^j0);
    alpha_BCS(1:2^j1,(2^j0+1):2^j1) = reshape(temp(n1+1:n1+n2), 2^j1, 2^j1-2^j0);
    % Reconstruct
    I_BCS = IWT2_PO(alpha_BCS, j0, qmf);
    % compute error
    errs(kk,ii) = norm(I0 - I_BCS,'fro') / norm(I0,'fro');
    Nzeros_num(kk,ii) = length(nonzeros(temp));
    theta_hat(:,ii) = temp;    
    Irecon{kk,ii} = I_BCS ;
    
    
    % method 4
    ii = ii+1 ;
    ii = 4;
    sigma4 = var(y_noise)/1e3 ;
    tic
    temp = GGAMP_SBL(y_noise, Phi, paras,sigma4,1) ;
    time_SBL(kk,ii) = toc ;
    alpha_BCS = alpha_BCS0;
    alpha_BCS((2^j0+1):2^j1,1:2^j0) = reshape(temp(1:n1), 2^j1-2^j0, 2^j0);
    alpha_BCS(1:2^j1,(2^j0+1):2^j1) = reshape(temp(n1+1:n1+n2), 2^j1, 2^j1-2^j0);
    % Reconstruct
    I_BCS = IWT2_PO(alpha_BCS, j0, qmf);
    % compute error
    errs(kk,ii) = norm(I0 - I_BCS,'fro') / norm(I0,'fro');
    Nzeros_num(kk,ii) = length(nonzeros(temp));
    theta_hat(:,ii) = temp;
    Irecon{kk,ii} = I_BCS ;
    
    
    % method  5
    ii = ii + 1;
    tic
    [weights2,ids_tipping] = SB1_Estimate(Phi,y_noise,InitVal.gamma_init,...
        InitVal.lambda_init,paras.iters,0,paras.delta) ;
    time_SBL(kk,ii) = toc;
    temp = zeros(N,1);
    temp(ids_tipping) = weights2 ;
    temp(abs(temp)./norm(temp)<threshold)=0 ;
    alpha_BCS = alpha_BCS0;
    alpha_BCS((2^j0+1):2^j1,1:2^j0) = reshape(temp(1:n1), 2^j1-2^j0, 2^j0);
    alpha_BCS(1:2^j1,(2^j0+1):2^j1) = reshape(temp(n1+1:n1+n2), 2^j1, 2^j1-2^j0);
    % Reconstruct
    I_BCS = IWT2_PO(alpha_BCS, j0, qmf);
    % compute error
    errs(kk,ii) = norm(I0 - I_BCS,'fro') / norm(I0,'fro');
    Nzeros_num(kk,ii) = length(nonzeros(temp));
    theta_hat(:,ii) = temp;
    Irecon{kk,ii} = I_BCS ;
    theta_all{kk,1} = theta_hat;
    
    % linear reconstruction 
    Nzeros_num(kk,end) = length(nonzeros(theta));
    Irecon{kk,end} = I_LIN ;
    
    % save results 
    clear Phi
%     save(filename,'-v7.3')
end

%
disp(errs)
disp(Nzeros_num)
disp(time_SBL)
beep;
disp('Done!') ;

%%  plot figures and save data 

figure()
nn=4 ;
subplot(2,3,1); AutoImage(Irecon{nn,end});
title(['(a) Linear, m=' num2str(4^j1)]);
tihao = {'(b) GFSBL','(c) LFSBL','(d) LPBCS','(e) GGAMP-SBL','(f) RVM-SBL' } ;
for i=1:num_method
    subplot(2,3,i+1); AutoImage(Irecon{nn,i});
    title([tihao{i},', m=' num2str(M)]);
end
