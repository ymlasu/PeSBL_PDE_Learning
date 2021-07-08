%% Calculate prior via PE-SBL
clear; clear global; close all; clc
global sigma_ksi2 mu_ksi
restoredefaultpath
addpath('G_code')  
load('Datasets\Burgers2D_N40_FFT.mat')
% sparse regression for PDE learning (Ut = Phi*ksi)
[model, llh] = parsiRvmRegSeq(A', b');
% prior of coefficent vector ksi(Gaussian)
mu_ksi = model.w;
sigma_ksi2 = 1./model.alpha;
% in case noise = 50%
mu_ksi(1) = [];
sigma_ksi2(1) = [];
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
global a1 a2 u0 uData t x y dt
% prepare data
L=2;Nx=101;Ny=51;Nt=100;
x=linspace(-L/2,L/2,Nx);
y=linspace(-L/2,L/2,Ny);
dt=0.02;
t = 0:dt:(Nt-1)*dt;
[X,Y]=meshgrid(x,y);
u0=0.1*sech(20*X.^2+25*Y.^2);
a1 = 0.01;
a2 = -1;
u = solBurgers2D();
rng(1)
noiseL = 0.50;
un = u+noiseL*std(u,0,'all')*randn(size(u));
uData = un;
Ne = size(uData,1)*size(uData,2)*size(uData,3);
clear L Nx Ny Nt u un X Y noiseL
rng default

%% Gibbs_M-H sampling
sigmaM1 = 0.12;
sigmaM2 = 0.0007;
numSamps = 10000;
% initial values:
global mu_e sigma_e2
ksi = mu_ksi;
err = errBurgers2D(ksi);
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
    err = errBurgers2D(ksi);
    C1 = Ne/sigma_e2+1/sigma_mu_e2;
    C2 = sum(err)/sigma_e2/C1;
    mu_e = C2;
    sigma_e2 = (beta_e+1/2*sum((err-mu_e).^2))/(Ne/2+alpha_e+1);
end
rejection_rate1 = rej_count1/numSamps;
rejection_rate2 = rej_count2/numSamps;

%% Save data
% save('Burger2DN50_MCMC.mat','rejection_rate1','rejection_rate2','mu_ksi','sigma_ksi2','ksiRec','mu_eRec','sigma_e2Rec');

%% Visulize distributions: prior and posterior
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\Burger2DN20_MCMC.mat')
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
title('(a) $\xi_1^0\sim \mathcal{N}(-0.9881,0.9964)$','Interpreter','latex')
ylabel('probability')
% plot the posterior distribution with normal fit
axes(h(3));box on;hold on;
histogram(ksi1,30,'Normalization','pdf');
x = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x,pd1.mu,pd1.sigma);
plot(x,y1,'r','LineWidth',1.5)
xlabel('$\xi_1$','Interpreter','latex')
title('(c) $\xi_1^1\sim \mathcal{N}(-1.0049,0.0033)$','Interpreter','latex')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x2,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_2^0\sim \mathcal{N}(0.0102,1.05\times10^{-4})$','Interpreter','latex')
axes(h(4));box on;hold on;
histogram(ksi2,30,'Normalization','pdf');
x = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x,pd2.mu,pd2.sigma);
plot(x,y2,'r','LineWidth',1.5)
xlabel('$\xi_2$','Interpreter','latex')
title('(d) $\xi_2^1\sim \mathcal{N}(0.0100,1.02\times10^{-7})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_Burgers2DN20.png')
% saveas(ff,'Figures\BMU_Burgers2DN20.eps','epsc') 
% savefig(ff,'Figures\BMU_Burgers2DN20.fig')

%% Functions
function err = errBurgers2D(a)
global a1 a2 uData
a1 = a(2);
a2 = a(1);
u1 = solBurgers2D();
norm_uData= sqrt(sum(uData.^2,'all'));
err = reshape((u1-uData)/norm_uData,[],1);
end

function u = solBurgers2D()
global a1 a2 u0 t x y dt
dx=x(2)-x(1);
dy=y(2)-y(1);
Nt = length(t);
Nx = length(x);
Ny = length(y);
u = zeros(Nt,Ny,Nx);
u(1,:,:)=u0;
for i=2:Nt
    u_1=reshape(u(i-1,:,:),Ny,Nx);
    formwork1=[-1:1];formwork2=[-1:1];
    du_t1=(a1*(differx(u_1,formwork1,2)/dx^2+differy(u_1,formwork1,2)/dy^2)...
        +a2*u_1.*((differx(u_1,formwork2,1)/dx+differy(u_1,formwork2,1)/dy)));
    u1=u_1+dt*du_t1;
    du_t2=(0.01*(differx(u1,formwork1,2)/dx^2+differy(u1,formwork1,2)/dy^2)...
        -u1.*((differx(u1,formwork2,1)/dx+differy(u1,formwork2,1)/dy)));
    u2=0.75*u_1+0.25*(u1+dt*du_t1);
    du_t3=(0.01*(differx(u2,formwork1,2)/dx^2+differy(u2,formwork1,2)/dy^2)...
        -u2.*((differx(u2,formwork2,1)/dx+differy(u2,formwork2,1)/dy)));
    unew=1/3*u_1+2/3*(u2+dt*du_t3);
    u(i,:,:)=unew;
end
end

function LLH = likelihood_ksi(ksi)
global sigma_ksi2 mu_ksi mu_e sigma_e2
err = errBurgers2D(ksi);
Jksi = 1/2*sum(1./sigma_ksi2.*(ksi-mu_ksi).^2);
Jerr = 1/2*sum((err-mu_e).^2/sigma_e2);
LLH = -Jksi-Jerr;
LLH = double(LLH);
end