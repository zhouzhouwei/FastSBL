clear ;  clc ;
addpath(genpath('SBL_Tipping2001\'))
addpath('tools\')
rand_state = rng(0,'v5normal') ;

% experimental setting
n = 2000 ; num_method = 4 ; Ns=2000:1000:1e4;
nums = 100;  snr = 40 ;  sp = 0.1 ; mnRatio = 0.6;
% filename = ['ResultsData\Sparse_Signal_Recovery\Spike_',num2str(snr),'dB_n',num2str(Ns(end)),'_2_1.mat'] ;

% parameters
paras.a0 = 1e-6 ;       paras.b0 = 1e-6 ;
paras.c0 = 1+1e-6;     paras.d0 = 1e-6 ;
paras.e0 = 1e-2 ;
paras.iters = 5000;    paras.threshold = 1e-3;
paras.delta = 1e-5 ;   % for stopping criterion
paras.normalized = 1 ;
epsilon = 1e-4 ;
tau0 = 1e-2 ;
threshold = paras.threshold ;


% storage the results
Nmax = length(Ns);
time_SBL = zeros(nums,Nmax, num_method);
errs = zeros(nums, Nmax, num_method);
num_fails = zeros(Nmax, num_method);
Iterations = zeros(nums,Nmax,2);
Nzeros_num = zeros(nums,Nmax,num_method+1) ;

% main loop
for jj=1:Nmax
    n = Ns(jj);
    m = n*mnRatio;
    fprintf(2,'The matrix has %d basis functions:\n',n) ;

    for kk=1:nums
        % generate the data
        Phi = randn(m,n);
        w = genSpikeSignal(fix(sp*n),n) ;
        y = Phi * w  ;
        y_noise = awgn(y, snr, 'measured');
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
        xhat = zeros(n,num_method);
        fprintf('The %d th experiment:\n',kk) ;


        % method 1
        ii = 1;   % index of method
        tic
        [xhat(:,ii),Iterations(kk,jj,1)] = Ga_FSBL(y_noise, Phi, paras, InitVal) ;
        time_SBL(kk,jj,ii) = toc ;
        errs(kk,jj,ii) = norm(xhat(:,ii)-w)/norm(w) ;
        if norm(xhat(:,ii)-w,'inf')/norm(w)>tau0
            num_fails(jj,ii) = num_fails(jj,ii)+1;
        end
        Nzeros_num(kk,jj,ii) = length(nonzeros(xhat(:,ii)));


        % method 2
        ii = ii+1 ;
        tic
        [xhat(:,ii),Iterations(kk,jj,2)] = La_FSBL(y_noise, Phi, paras, InitVal) ;
        time_SBL(kk,jj,ii) = toc;
        errs(kk,jj,ii) = norm(xhat(:,ii)-w)/norm(w) ;
        if norm(xhat(:,ii)-w,'inf')/norm(w)>tau0
            num_fails(jj,ii) = num_fails(jj,ii)+1;
        end
        Nzeros_num(kk,jj,ii) = length(nonzeros(xhat(:,ii)));


        % method 3
        ii = ii+1 ;
        sigma2 = var(y_noise)/1e2 ;
        delta_La = 1e-10 ;
        tic
        [weights,used_ids] = FastLaplace(Phi,y_noise,sigma2,delta_La);
        time_SBL(kk,jj,ii) = toc;
        temp = zeros(n,1);
        temp(used_ids) = weights ;
        temp(abs(temp)./norm(temp)<threshold) = 0 ;
        xhat(:,ii) = temp ;
        errs(kk,jj,ii) = norm(xhat(:,ii)-w)/norm(w) ;
        if norm(xhat(:,ii)-w,'inf')/norm(w)> tau0
            num_fails(jj,ii) = num_fails(jj,ii)+1;
        end
        Nzeros_num(kk,jj,ii) = length(nonzeros(xhat(:,ii)));


        % method 4
        ii = ii+1 ;
        tic
        xhat(:,ii) = GGAMP_SBL(y_noise, Phi, paras) ;
        time_SBL(kk,jj,ii) = toc ;
        errs(kk,jj,ii) = norm(xhat(:,ii)-w)/norm(w) ;
        if norm(xhat(:,ii)-w,'inf')/norm(w)> tau0
            num_fails(jj,ii) = num_fails(jj,ii)+1;
        end
        Nzeros_num(kk,jj,ii) = length(nonzeros(xhat(:,ii)));


        % method  5
        ii = ii + 1;
        tic
        [weights2,ids_tipping] = SB1_Estimate(Phi,y_noise,InitVal.gamma_init,...
            InitVal.lambda_init,paras.iters,0,paras.delta) ;
        time_SBL(kk,jj,ii) = toc;
        temp = zeros(n,1);
        temp(ids_tipping) = weights2 ;
        temp(abs(temp)./norm(temp)<threshold)=0 ;
        xhat(:,ii) = temp ;
        errs(kk,jj,ii) = norm(xhat(:,ii)-w)/norm(w) ;
        if norm(xhat(:,ii)-w,'inf')/norm(w)> tau0
            num_fails(jj,ii) = num_fails(jj,ii)+1;
        end
        Nzeros_num(kk,jj,ii) = length(nonzeros(xhat(:,ii)));
        
        % nonzeros of true signal
        Nzeros_num(kk,jj,ii+1) = length(nonzeros(w)) ;
    end
    clear Phi ;
%     save(filename) ;
   % end all experiments
end

% the average results
err_mean = mean(errs);
err_mean = squeeze(err_mean) ;
time_mean = mean(time_SBL);
time_mean = squeeze(time_mean) ;
Nzeros_mean = mean(Nzeros_num);
Nzeros_mean = squeeze(Nzeros_mean) ;
Iter_mean = mean(Iterations) ;
Iter_mean = squeeze(Iter_mean) ;

% save the results
% save(filename) ;


%%   plot figures
markers = {'-o','-*','-<','-x','-s','->','-v'};
figure()
subplot(2,1,1)
for i=1:num_method
    plot(Ns,err_mean(:,i),markers{i},'LineWidth',2);
    hold on
end
set(gca,'FontSize',12)
legend('GFSBL','LFSBL','LPBCS','GGAMP-SBL','RVM-SBL')
xlabel('Signal Length n')
ylabel('nRMSE')
grid on
subplot(2,1,2)
for i=1:num_method
    plot(Ns,time_mean(:,i),markers{i},'LineWidth',2);
    hold on
end
set(gca,'FontSize',12,'Yscale','log')
legend('GFSBL','LFSBL','LPBCS','GGAMP-SBL','RVM-SBL')
xlabel('Signal Length n')
ylabel('Running Time')
grid on
