%% Calculate prior via PE-SBL
clear; clear global; close all; clc
global sigma_ksi2 mu_ksi
restoredefaultpath
addpath('G_code')  
load('Datasets\NS_N30_FFT2.mat')
% sparse regression for PDE learning (Ut = Phi*ksi)
A = [A1;A2];
b = [b1;b2];
[model, llh] = parsiRvmRegSeq(A',b');
% prior of coefficent vector ksi(Gaussian)
mu_ksi = model.w;
sigma_ksi2 = 1./model.alpha;
clearvars -except mu_ksi sigma_ksi2
N_ksi = length(mu_ksi);
% prior of error mean mu_e: zero mean Gaussian: mu_e ~ N(0,sigma_mu_e2)
sigma_mu_e2 = (1/3)^2;
% prior of error variance: non-informative Inverser Gammma: 
% sigma_e2 ~ InvGamma(alpha_e, beta_e)
alpha_e = 1;
beta_e = 2;

% For likelihood, the error function follows a Gaussian distribution
% err ~ N(mu_e,sigma_e2)

% Conditional posteriors
% p(ksi|.)∝exp[-(Jksi+Je)]
% p(mu_e|.)∝N(C2,1/C1), 
% C1 = Ne/sigma_e2+1/sigma_mu_e2, 
% C2 = sum(e_i)/sigma_e2/C1
% p(sigma_e2|.)∝InvGamma(Ne/2+alpha_e,beta_e+1/2*sum(e_i-mu_e)^2)

%% Data preparation
global un vn pn
[~,~,~,u,v,p] = mit18086_navierstokes(0,-1,-1,0.01);
rng(1)
noiseL = 0.50;
un = u+noiseL*std(u,0,'all')*randn(size(u));
vn = v+noiseL*std(v,0,'all')*randn(size(v));
pn = p+noiseL*std(p,0,'all')*randn(size(p));
Ne = size(un,1)*size(un,2)*size(un,3)*3;
clear u v p noiseL
rng default

%% Gibbs_M-H sampling
sigmaM1 = 0.0015;
sigmaM2 = 0.000018;
sigmaM3 = 0.0027;
numSamps = 10000;
% initial values:
global mu_e sigma_e2
ksi = mu_ksi;
err = errNS(ksi);
mu_e = mean(err);
sigma_e2 = var(err);
ksiRec = zeros(length(ksi),numSamps);
mu_eRec = zeros(1,numSamps);
sigma_e2Rec = zeros(1,numSamps);
rej_count1 = 0;
rej_count2 = 0;
rej_count3 = 0;
for n = 1:numSamps
    
    % display progress
    if mod(10*n,numSamps) == 0
        disp(n/numSamps); 
    end
    
    % record samples
    ksiRec(:,n) = ksi;
    mu_eRec(:,n) = mu_e;
    sigma_e2Rec(:,n) = sigma_e2;
    
    % METROPOLIS MARKOV CHAIN
    ksiNew1 = ksi(1) + randn*sigmaM1;
    ksiNew = [ksiNew1;ksi(2);ksi(3)];
    LLH1_Old = likelihood_ksi(ksi);
    LLH1_New = likelihood_ksi(ksiNew);
    if rand(1) < exp(LLH1_New-LLH1_Old) % LLH1_New/LLH1_Old
        ksi = ksiNew;
    else
        rej_count1 = rej_count1+1;
    end    
 
    ksiNew2 = ksi(2) + randn*sigmaM2;
    while(ksiNew2<0)
        ksiNew2 = ksi(2) + randn*sigmaM2;
    end
    ksiNew = [ksi(1);ksiNew2;ksi(3)];
    LLH1_Old = likelihood_ksi(ksi);
    LLH1_New = likelihood_ksi(ksiNew);
    if rand(1) < exp(LLH1_New-LLH1_Old)
        ksi = ksiNew;
    else
        rej_count2 = rej_count2+1;
    end
    
    ksiNew3 = ksi(3) + randn*sigmaM3;
    ksiNew = [ksi(1);ksi(2);ksiNew3];
    LLH1_Old = likelihood_ksi(ksi);
    LLH1_New = likelihood_ksi(ksiNew);
    if rand(1) < exp(LLH1_New-LLH1_Old) % LLH1_New/LLH1_Old
        ksi = ksiNew;
    else
        rej_count3 = rej_count3+1;
    end
    
    err = errNS(ksi);
    C1 = Ne/sigma_e2+1/sigma_mu_e2;
    C2 = sum(err)/sigma_e2/C1;
    mu_e = C2;
    sigma_e2 = (beta_e+1/2*sum((err-mu_e).^2))/(Ne/2+alpha_e+1);
end
rejection_rate1 = rej_count1/numSamps;
rejection_rate2 = rej_count2/numSamps;
rejection_rate3 = rej_count3/numSamps;

%% Save data
% save('NSN50_MCMC.mat','rejection_rate1','rejection_rate2','rejection_rate3','mu_ksi','sigma_ksi2','ksiRec','mu_eRec','sigma_e2Rec');

%% Visulize distributions: prior and posterior
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\NSN20_MCMC.mat')
% fit Gaussian distribution
ksi1 = ksiRec(1,101:end)';
ksi2 = ksiRec(2,101:end)';
ksi3 = ksiRec(3,101:end)';
pd1 = fitdist(ksi1,'Normal');
pd2 = fitdist(ksi2,'Normal');
pd3 = fitdist(ksi3,'Normal');
% prior std of ksi
sigma_ksi = sqrt(sigma_ksi2);
close all
ff = fig('units','inches','width',9,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(2,3,[.10 .05],[.08 .06],[.045 .04]);
% plot the prior Gaussian distribution
axes(h(1));box on;hold on;
x1 = linspace(mu_ksi(1)-5*sigma_ksi(1),mu_ksi(1)+5*sigma_ksi(1),100);
ksi1prior = normpdf(x1,mu_ksi(1),sigma_ksi(1));
plot(x1,ksi1prior,'b','LineWidth',1.5)
title('(a) $\xi_1^0\sim \mathcal{N}(-0.9941,0.9902)$','Interpreter','latex')
ylabel('probability')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x1,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_2^0\sim \mathcal{N}(0.0102,1.07\times10^{-4})$','Interpreter','latex')
axes(h(3));box on;hold on;
x3 = linspace(mu_ksi(3)-5*sigma_ksi(3),mu_ksi(3)+5*sigma_ksi(3),100);
ksi3prior = normpdf(x3,mu_ksi(3),sigma_ksi(3));
plot(x3,ksi3prior,'b','LineWidth',1.5)
title('(c) $\xi_3^0\sim \mathcal{N}(-0.9746,0.9693)$','Interpreter','latex')
% plot the posterior distribution with normal fit
axes(h(4));box on;hold on;
histogram(ksi1,50,'Normalization','pdf');
x1 = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x1,pd1.mu,pd1.sigma);
plot(x1,y1,'r','LineWidth',1.5)
xlim([-1.01 -0.99])
xlabel('$\xi_1$','Interpreter','latex')
title('(d) $\xi_1^1\sim \mathcal{N}(-1.0001,1.89\times10^{-6})$','Interpreter','latex')
axes(h(5));box on;hold on;
histogram(ksi2,50,'Normalization','pdf');
x2 = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x2,pd2.mu,pd2.sigma);
plot(x2,y2,'r','LineWidth',1.5)
xlim([9.9 10.1]*1E-3)
xlabel('$\xi_2$','Interpreter','latex')
title('(e) $\xi_2^1\sim \mathcal{N}(0.0100,2.04\times10^{-10})$','Interpreter','latex')
axes(h(6));box on;hold on;
histogram(ksi3,50,'Normalization','pdf');
x3 = linspace(pd3.mu-5*pd3.sigma,pd3.mu+5*pd3.sigma,1000);
y3 = normpdf(x3,pd3.mu,pd3.sigma);
plot(x3,y3,'r','LineWidth',1.5)
xlim([-1.01 -0.99])
xlabel('$\xi_3$','Interpreter','latex')
title('(f) $\xi_3^1\sim \mathcal{N}(-1.0000,1.92\times10^{-6})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_NSN20.png')
% saveas(ff,'Figures\BMU_NSN20.eps','epsc') 
% savefig(ff,'Figures\BMU_NSN20.fig')

%% Functions
function err = errNS(a)
global un vn pn
[~,~,~,u,v,p] = mit18086_navierstokes(0,a(1),a(3),a(2));
norm_u= sqrt(sum(un.^2,'all'));
norm_v= sqrt(sum(vn.^2,'all'));
norm_p= sqrt(sum(pn.^2,'all'));
err_u = reshape((u-un)/norm_u,[],1);
err_v = reshape((v-vn)/norm_v,[],1);
err_p = reshape((p-pn)/norm_p,[],1);
err = [err_u;err_v;err_p];
end

function LLH = likelihood_ksi(ksi)
global sigma_ksi2 mu_ksi mu_e sigma_e2
err = errNS(ksi);
Jksi = 1/2*sum(1./sigma_ksi2.*(ksi-mu_ksi).^2);
Jerr = 1/2*sum((err-mu_e).^2/sigma_e2);
LLH = -Jksi-Jerr;
LLH = double(LLH);
end