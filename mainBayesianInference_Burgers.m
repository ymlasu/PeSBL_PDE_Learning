%% Notes:
% This script conducts Bayesian inference of PDE forms using measured data
% Prior of ksi: results of sparse Bayesian learning (SBL)
% Likelikhood function is build using the error of PDE solutions

%% Calculate prior via PE-SBL
clear; clear global; close all; clc
global sigma_ksi2 mu_ksi
restoredefaultpath
load('Datasets\BurgersMATLAB_N10_FFT.mat')
% sparse regression for PDE learning (Ut = Phi*ksi)
[model, llh] = parsiRvmRegSeq(A', b');
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
global t x u0
load('Datasets\BurgersMATLAB_N0.mat')
rng(1)
noiseL = 0.40;
un = u+noiseL*std(u,0,'all')*randn(size(u));
u0 = un';clear un
Ne = size(u0,1)*size(u0,2);
rng default

%% Gibbs_M-H sampling: Noise = 0,10,20,30
sigmaM1 = 0.05;
sigmaM2 = 0.002;
numSamps = 10000;
% initial values:
global mu_e sigma_e2
ksi = mu_ksi;
err = errBurgers(ksi);
mu_e = mean(err);
sigma_e2 = var(err);
ksiRec = zeros(length(ksi),numSamps);
mu_eRec = zeros(1,numSamps);
sigma_e2Rec = zeros(1,numSamps);
rej_count1 = 0;
rej_count2 = 0;
for n = 1:numSamps
    if mod(10*n,numSamps) == 0
        disp(n/numSamps); 
    end
    ksiRec(:,n) = ksi;
    mu_eRec(:,n) = mu_e;
    sigma_e2Rec(:,n) = sigma_e2;
    % METROPOLIS MARKOV CHAIN
    ksiNew1 = ksi(1) + randn*sigmaM1;
    ksiNew = [ksiNew1;ksi(2)];
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
    ksiNew = [ksi(1);ksiNew2];
    LLH1_Old = likelihood_ksi(ksi);
    LLH1_New = likelihood_ksi(ksiNew);
    if rand(1) < exp(LLH1_New-LLH1_Old)
        ksi = ksiNew;
    else
        rej_count2 = rej_count2+1;
    end
    %{
%     ksiNew = ksi + randn(N_ksi,1)*sigmaM;
%     while(ksiNew(2)<0)
%         ksiNew = ksi + randn(N_ksi,1)*sigmaM;
%     end
%     LLH1_Old = likelihood_ksi(ksi);
%     LLH1_New = likelihood_ksi(ksiNew);
%     if rand(1) < LLH1_New/LLH1_Old
%         ksi = ksiNew;
%     else
%         rej_count = rej_count+1;
%     end
    %}
    err = errBurgers(ksi);
    C1 = Ne/sigma_e2+1/sigma_mu_e2;
    C2 = sum(err)/sigma_e2/C1;
    mu_e = C2;
    sigma_e2 = (beta_e+1/2*sum((err-mu_e).^2))/(Ne/2+alpha_e+1);
end
rejection_rate1 = rej_count1/numSamps;
rejection_rate2 = rej_count2/numSamps;

%% Save data
% save('BurgersN40_MCMC2.mat','rejection_rate1','rejection_rate2','mu_ksi','sigma_ksi2','ksiRec','mu_eRec','sigma_e2Rec');

%% Visulize distributions: prior and posterior
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\BurgersN40_MCMC2.mat')
% fit Gaussian distribution
ksi1 = ksiRec(1,101:end)';
ksi2 = ksiRec(2,101:end)';
pd1 = fitdist(ksi1,'Normal');
pd2 = fitdist(ksi2,'Normal');
% prior std of ksi
sigma_ksi = sqrt(sigma_ksi2);
close all
ff = fig('units','inches','width',6,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(2,2,[.10 .05],[.08 .06],[.08 .03]);
% plot the prior Gaussian distribution
axes(h(1));box on;hold on;
x1 = linspace(mu_ksi(1)-5*sigma_ksi(1),mu_ksi(1)+5*sigma_ksi(1),100);
ksi1prior = normpdf(x1,mu_ksi(1),sigma_ksi(1));
plot(x1,ksi1prior,'b','LineWidth',1.5)
title('(a) $\xi_1^0\sim \mathcal{N}(-0.974,0.955)$','Interpreter','latex')
ylabel('probability')
% plot the posterior distribution with normal fit
axes(h(3));box on;hold on;
histogram(ksi1,30,'Normalization','pdf');
x = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x,pd1.mu,pd1.sigma);
plot(x,y1,'r','LineWidth',1.5)
xlabel('$\xi_1$','Interpreter','latex')
title('(c) $\xi_1^1\sim \mathcal{N}(-0.9987,6.77\times10^{-4})$','Interpreter','latex')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x2,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_2^0\sim \mathcal{N}(0.0074,5.55\times10^{-5})$','Interpreter','latex')
axes(h(4));box on;hold on;
histogram(ksi2,30,'Normalization','pdf');
x = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x,pd2.mu,pd2.sigma);
plot(x,y2,'r','LineWidth',1.5)
xlabel('$\xi_2$','Interpreter','latex')
title('(d) $\xi_2^1\sim \mathcal{N}(0.0034,7.26\times10^{-7})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_burgersN40_2.png')
% saveas(ff,'Figures\BMU_burgersN40_2.eps','epsc') 
% savefig(ff,'Figures\BMU_burgersN40_2.fig')
%%
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\BurgersN50_MCMC.mat')
% fit Gaussian distribution
ksi1 = ksiRec(1,101:end)';
ksi2 = ksiRec(2,101:end)';
ksi3 = ksiRec(3,101:end)';
ksi4 = ksiRec(4,101:end)';
pd1 = fitdist(ksi1,'Normal');
pd2 = fitdist(ksi2,'Normal');
pd3 = fitdist(ksi3,'Normal');
pd4 = fitdist(ksi4,'Normal');
% prior std of ksi
sigma_ksi = sqrt(sigma_ksi2);
close all
ff = fig('units','inches','width',9,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(2,4,[.10 .05],[.08 .06],[.045 .04]);
% plot the prior Gaussian distribution
axes(h(1));box on;hold on;
x1 = linspace(mu_ksi(1)-5*sigma_ksi(1),mu_ksi(1)+5*sigma_ksi(1),100);
ksi1prior = normpdf(x1,mu_ksi(1),sigma_ksi(1));
plot(x1,ksi1prior,'b','LineWidth',1.5)
title('(a) $\xi_1^0\sim \mathcal{N}(-1.05,1.11)$','Interpreter','latex')
ylabel('probability')
axes(h(3));box on;hold on;
x3 = linspace(mu_ksi(3)-5*sigma_ksi(3),mu_ksi(3)+5*sigma_ksi(3),100);
ksi3prior = normpdf(x3,mu_ksi(3),sigma_ksi(3));
plot(x3,ksi3prior,'b','LineWidth',1.5)
title('(c) $\xi_2^0\sim \mathcal{N}(0.0086,7.52\times10^{-5})$','Interpreter','latex')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x1,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_3^0\sim \mathcal{N}(0.11,0.02)$','Interpreter','latex')
axes(h(4));box on;hold on;
x4 = linspace(mu_ksi(4)-5*sigma_ksi(4),mu_ksi(4)+5*sigma_ksi(4),100);
ksi4prior = normpdf(x4,mu_ksi(4),sigma_ksi(4));
plot(x4,ksi4prior,'b','LineWidth',1.5)
title('(d) $\xi_4^0\sim \mathcal{N}(0.02,5.4\times10^{-4}))$','Interpreter','latex')
% plot the posterior distribution with normal fit
axes(h(5));box on;hold on;
histogram(ksi1,30,'Normalization','pdf');
x1 = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x1,pd1.mu,pd1.sigma);
plot(x1,y1,'r','LineWidth',1.5)
xlim([-1.2 -0.8])
xlabel('$\xi_1$','Interpreter','latex')
title('(e) $\xi_1^1\sim \mathcal{N}(-1.016,0.0015)$','Interpreter','latex')
axes(h(6));box on;hold on;
histogram(ksi2,30,'Normalization','pdf');
x2 = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x2,pd2.mu,pd2.sigma);
plot(x2,y2,'r','LineWidth',1.5)
xlim([-0.17 0.23])
xlabel('$\xi_3$','Interpreter','latex')
title('(f) $\xi_3^1\sim \mathcal{N}(0.0337,0.0023)$','Interpreter','latex')
axes(h(7));box on;hold on;
histogram(ksi3,30,'Normalization','pdf');
x3 = linspace(pd3.mu-5*pd3.sigma,pd3.mu+5*pd3.sigma,1000);
y3 = normpdf(x3,pd3.mu,pd3.sigma);
plot(x3,y3,'r','LineWidth',1.5)
xlim([-0.001 0.009])
xlabel('$\xi_2$','Interpreter','latex')
title('(g) $\xi_2^1\sim \mathcal{N}(0.0039,1.26\times10^{-6})$','Interpreter','latex')
axes(h(8));box on;hold on;
histogram(ksi4,30,'Normalization','pdf');
x4 = linspace(pd4.mu-5*pd4.sigma,pd4.mu+5*pd4.sigma,1000);
y4 = normpdf(x4,pd4.mu,pd4.sigma);
plot(x4,y4,'r','LineWidth',1.5)
% xlim([-0.001 0.009])
xlabel('$\xi_4$','Interpreter','latex')
title('(h) $\xi_4^1\sim \mathcal{N}(3.57\times10^{-4},3.53\times10^{-6})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_burgersN50.png')
% saveas(ff,'Figures\BMU_burgersN50.eps','epsc') 
% savefig(ff,'Figures\BMU_burgersN50.fig')
%%
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\BurgersN50_MCMC2.mat')
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
title('(a) $\xi_1^0\sim \mathcal{N}(-1.05,1.11)$','Interpreter','latex')
ylabel('probability')
axes(h(3));box on;hold on;
x3 = linspace(mu_ksi(3)-5*sigma_ksi(3),mu_ksi(3)+5*sigma_ksi(3),100);
ksi3prior = normpdf(x3,mu_ksi(3),sigma_ksi(3));
plot(x3,ksi3prior,'b','LineWidth',1.5)
title('(c) $\xi_2^0\sim \mathcal{N}(0.0086,7.52\times10^{-5})$','Interpreter','latex')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x1,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_3^0\sim \mathcal{N}(0.11,0.02)$','Interpreter','latex')
% plot the posterior distribution with normal fit
axes(h(4));box on;hold on;
histogram(ksi1,30,'Normalization','pdf');
x1 = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x1,pd1.mu,pd1.sigma);
plot(x1,y1,'r','LineWidth',1.5)
xlim([-1.2 -0.8])
xlabel('$\xi_1$','Interpreter','latex')
title('(d) $\xi_1^1\sim \mathcal{N}(-1.016,0.0015)$','Interpreter','latex')
axes(h(5));box on;hold on;
histogram(ksi2,30,'Normalization','pdf');
x2 = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x2,pd2.mu,pd2.sigma);
plot(x2,y2,'r','LineWidth',1.5)
xlim([-0.17 0.23])
xlabel('$\xi_3$','Interpreter','latex')
title('(e) $\xi_3^1\sim \mathcal{N}(0.0337,0.0023)$','Interpreter','latex')
axes(h(6));box on;hold on;
histogram(ksi3,30,'Normalization','pdf');
x3 = linspace(pd3.mu-5*pd3.sigma,pd3.mu+5*pd3.sigma,1000);
y3 = normpdf(x3,pd3.mu,pd3.sigma);
plot(x3,y3,'r','LineWidth',1.5)
xlim([-0.001 0.009])
xlabel('$\xi_2$','Interpreter','latex')
title('(f) $\xi_2^1\sim \mathcal{N}(0.0039,1.26\times10^{-6})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_burgersN50_2.png')
% saveas(ff,'Figures\BMU_burgersN50_2.eps','epsc') 
% savefig(ff,'Figures\BMU_burgersN50_2.fig')
%%
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\BurgersN40_MCMC.mat')
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
title('(a) $\xi_1^0\sim \mathcal{N}(-0.974,0.955)$','Interpreter','latex')
ylabel('probability')
axes(h(3));box on;hold on;
x3 = linspace(mu_ksi(3)-5*sigma_ksi(3),mu_ksi(3)+5*sigma_ksi(3),100);
ksi3prior = normpdf(x3,mu_ksi(3),sigma_ksi(3));
plot(x3,ksi3prior,'b','LineWidth',1.5)
title('(c) $\xi_3^0\sim \mathcal{N}(0.0288,9.36\times10^{-4})$','Interpreter','latex')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x1,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_2^0\sim \mathcal{N}(0.0074,5.55\times10^{-5})$','Interpreter','latex')
% plot the posterior distribution with normal fit
axes(h(4));box on;hold on;
histogram(ksi1,30,'Normalization','pdf');
x1 = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x1,pd1.mu,pd1.sigma);
plot(x1,y1,'r','LineWidth',1.5)
xlim([-1.2 -0.8])
xlabel('$\xi_1$','Interpreter','latex')
title('(d) $\xi_1^1\sim \mathcal{N}(-0.9984,6.64\times10^{-4})$','Interpreter','latex')
axes(h(5));box on;hold on;
histogram(ksi2,30,'Normalization','pdf');
x2 = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x2,pd2.mu,pd2.sigma);
plot(x2,y2,'r','LineWidth',1.5)
xlabel('$\xi_2$','Interpreter','latex')
title('(e) $\xi_2^1\sim \mathcal{N}(0.0035,7.34\times10^{-7})$','Interpreter','latex')
axes(h(6));box on;hold on;
histogram(ksi3,30,'Normalization','pdf');
x3 = linspace(pd3.mu-5*pd3.sigma,pd3.mu+5*pd3.sigma,1000);
y3 = normpdf(x3,pd3.mu,pd3.sigma);
plot(x3,y3,'r','LineWidth',1.5)
xlabel('$\xi_3$','Interpreter','latex')
title('(f) $\xi_3^1\sim \mathcal{N}(2.01\times10^{-4},3.25\times10^{-6})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_burgersN40.png')
% saveas(ff,'Figures\BMU_burgersN40.eps','epsc') 
% savefig(ff,'Figures\BMU_burgersN40.fig')
%% Uncertainty propagation
% This script propagates uncertainty obtained from BMU to system respnose
clear; clear global; close all; clc
load('.\Datasets\BurgersN20_MCMC.mat')
ksi1 = ksiRec(1,101:end)';
ksi2 = ksiRec(2,101:end)';
clearvars -except ksi1 ksi2
global a1 a2 t x u0
load('Datasets\BurgersMATLAB_N0.mat','t','x')
t(end+1:end+100) = t+t(end)+t(2)-t(1);
m = 0;uRec = zeros(length(ksi1),length(x),length(t));
for i = 1:length(ksi1)
    i
    a1 = ksi2(i); % coeff of u_xx
    a2 = ksi1(i); % coeff of uu_x
    u1 = pdepe(m,@pdeBurgersOPT,@pdex1ic,@pdex1bc,x,t);
    uRec(i,:,:) = u1(:,:,1)';
end
save('Datasets\BurgersUP_N20_2.mat','ksi1','ksi2','uRec')
%% Visualize uncertainty in system responses
clear;clear global;clc
load('Datasets\BurgersMATLAB_N0.mat','t','x')
load('Datasets\BurgersUP_N20_2.mat')
global a1 a2 t x u0
t(end+1:end+100) = t+t(end)+t(2)-t(1);
m = 0; a1 = 0.01/pi; a2 = -1;
u = pdepe(m,@pdeBurgersOPT,@pdex1ic,@pdex1bc,x,t);
u = u(:,:,1)';
%%
close all;clc;xx = 52;
ff = fig('units','inches','width',9,'height',4,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,3,[.10 .05],[.12 .06],[.06 .02]);
axes(h(1));box on;hold on;
for i=1:length(t)
L1 = scatter(t(i)*ones(size(uRec,1),1),squeeze(uRec(:,xx,i)),5,'MarkerEdgeColor','g',...
              'MarkerFaceColor','g');
end
L2 = plot(t,mean(squeeze(uRec(:,xx,:))),'r-','LineWidth',1.0);
xlim([1 2])
set(gca,'ytick',0.14:0.04:0.34)
legend([L1,L2],{'samples','mean'})
legend('boxoff')
xlabel('$t$','Interpreter','latex')
ylabel('$u$','Interpreter','latex')
title('(a)','FontWeight','normal')
text(1.6,0.3,'$x = -0.6$','Interpreter','latex')
axes(h(2));box on;hold on;
L1 = plot(t,u(xx,:),'b--','LineWidth',1.5);
meanU = mean(squeeze(uRec(:,xx,:)));
L2 = plot(t,meanU,'r-','LineWidth',1.0);
stdU = std(squeeze(uRec(:,xx,:)));
curve1 = meanU+3*stdU;
curve2 = meanU-3*stdU;
t2 = [t',fliplr(t')];
inBetween = [curve1,fliplr(curve2)];
L3 = fill(t2,inBetween,'g');
alpha(.2)
xlim([1 2])
legend([L1,L2,L3],{'truth','mean','mean$\pm$3std'},'Interpreter','latex')
legend('boxoff')
xlabel('$t$','Interpreter','latex')
title('(b)','FontWeight','normal')
axes(h(3));box on;hold on;
plot(t,u(xx,:),'b--','LineWidth',2.5)
meanU = mean(squeeze(uRec(:,xx,:)));
plot(t,meanU,'r-','LineWidth',2.0)
stdU = std(squeeze(uRec(:,xx,:)));
curve1 = meanU+3*stdU;
curve2 = meanU-3*stdU;
t2 = [t',fliplr(t')];
inBetween = [curve1,fliplr(curve2)];
fill(t2,inBetween,'g');
alpha(.2)
xlabel('$t$','Interpreter','latex')
title('(c)','FontWeight','normal')
xlim([1.6 1.8])
set(gca,'ytick',0.175:0.01:0.225)
% saveas(ff,'Figures\BurgersUP_N20_2.png')
% saveas(ff,'Figures\BurgersUP_N20_2.eps','epsc') 
% savefig(ff,'Figures\BurgersUP_N20_2.fig')
%% Function definitions
function err = errBurgers(a)% regular
m = 0;
global a1 a2 t x u0
a1 = a(2); % coeff of u_xx
a2 = a(1); % coeff of uu_x
u1 = pdepe(m,@pdeBurgersOPT,@pdex1ic,@pdex1bc,x,t);
u1 = u1(:,:,1);
err = reshape((u1-u0)/norm(u0),[],1);
end

function [c,f,s] = pdeBurgersOPT(x,t,u,dudx) % Equation to solve
global a1 a2
c = 1;
f = a1*dudx;
s = a2*u*dudx;
end

function [c,f,s] = pdeBurgersOPT2(x,t,u,dudx) % Equation to solve
global a1 a2 a3 a4
c = 1;
f = a1*dudx;
s = a2*u*dudx+a3*u+a4*dudx;
end

function [c,f,s] = pdeBurgersOPT3(x,t,u,dudx) % Equation to solve
global a1 a2 a3
c = 1;
f = a1*dudx;
s = a2*u*dudx+a3*u;
end

function [c,f,s] = pdeBurgersOPT4(x,t,u,dudx) % Equation to solve
global a1 a2 a4
c = 1;
f = a1*dudx;
s = a2*u*dudx+a4*dudx;
end

function u0 = pdex1ic(x) % Initial conditions
u0 = -sin(pi*x);
end

function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t) % Boundary conditions
pl = ul;
ql = 0;
pr = ur;
qr = 0;
end

function LLH = likelihood_ksi(ksi)
global sigma_ksi2 mu_ksi mu_e sigma_e2
err = errBurgers(ksi);
Jksi = 1/2*sum(1./sigma_ksi2.*(ksi-mu_ksi).^2);
Jerr = 1/2*sum((err-mu_e).^2/sigma_e2);
% LLH = exp(-Jksi-Jerr);
LLH = -Jksi-Jerr;
LLH = double(LLH);
end

