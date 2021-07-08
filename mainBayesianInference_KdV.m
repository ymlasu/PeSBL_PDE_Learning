%% Calculate prior via PE-SBL
clear; clear global; close all; clc
global sigma_ksi2 mu_ksi
restoredefaultpath
load('Datasets\KdVMATLAB_N50_FFT.mat')
% sparse regression for PDE learning (Ut = Phi*ksi)
[model, llh] = parsiRvmRegSeq(A', b');
% prior of coefficent vector ksi(Gaussian)
mu_ksi = double(model.w);
sigma_ksi2 = 1./model.alpha;
% !!!flip elements in case with 50% noise
if model.index(1)>model.index(2)
mu_ksi = flip(mu_ksi);
sigma_ksi2 = flip(sigma_ksi2);
end
clearvars -except mu_ksi sigma_ksi2
N_ksi = length(mu_ksi);
% prior of error mean mu_e: zero mean Gaussian: mu_e ~ N(0,sigma_mu_e2)
sigma_mu_e2 = (1/3)^2;
% prior of error variance: non-informative Inverser Gammma: 
% sigma_e2 ~ InvGamma(alpha_e, beta_e)
alpha_e = 1;
beta_e = 2;

%% Data preparation
global a1 a2 t k u0t uData
Lx = 2; Lt = 1;
Nx = 512; Nt = 201;
x = Lx/Nx*(-Nx/2:Nx/2-1);
k = 2*pi/Lx*[0:Nx/2-1 -Nx/2:-1].';
u0 = cos(pi*x);   % initial condition
u0t = fft(u0);
t = Lt/Nt*(0:Nt-1);
% load('Datasets\KdV.mat','uu')
a1 = -1;
a2 = -0.0025;
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
uu=real(usol);
rng(1)
noiseL = 0.10;
un = uu+noiseL*std(uu,0,'all')*randn(size(uu));
uData = un;
Ne = size(uData,1)*size(uData,2);
clear Lx Lt Nx Nt un
rng default

%% Gibbs_M-H sampling
sigmaM1 = 0.0055;
sigmaM2 = 0.000015;
numSamps = 10000;
% initial values:
global mu_e sigma_e2
ksi = mu_ksi;
err = errKdV(ksi);
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
    while(ksiNew2>0)
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
    err = errKdV(ksi);
    C1 = Ne/sigma_e2+1/sigma_mu_e2;
    C2 = sum(err)/sigma_e2/C1;
    mu_e = C2;
    sigma_e2 = (beta_e+1/2*sum((err-mu_e).^2))/(Ne/2+alpha_e+1);
end
rejection_rate1 = rej_count1/numSamps;
rejection_rate2 = rej_count2/numSamps;

%% Save data
% save('KdVN50_MCMC.mat','rejection_rate1','rejection_rate2','mu_ksi','sigma_ksi2','ksiRec','mu_eRec','sigma_e2Rec');

%% Visulize distributions: prior and posterior
clear;close all;clear global;clc
addpath 'C:\Users\zzhan506\Google Drive\ASUResearch\PDE_learning\Bayesian\Code\G_code'
load('Datasets\KdVN20_MCMC.mat')
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
title('(a) $\xi_1^0\sim \mathcal{N}(-0.9524,0.9108)$','Interpreter','latex')
ylabel('probability')
% plot the posterior distribution with normal fit
axes(h(3));box on;hold on;
histogram(ksi1,30,'Normalization','pdf');
x = linspace(pd1.mu-5*pd1.sigma,pd1.mu+5*pd1.sigma,1000);
y1 = normpdf(x,pd1.mu,pd1.sigma);
plot(x,y1,'r','LineWidth',1.5)
xlabel('$\xi_1$','Interpreter','latex')
title('(c) $\xi_1^1\sim \mathcal{N}(-1.0004,1.26\times10^{-5})$','Interpreter','latex')
axes(h(2));box on;hold on;
x2 = linspace(mu_ksi(2)-5*sigma_ksi(2),mu_ksi(2)+5*sigma_ksi(2),100);
ksi2prior = normpdf(x2,mu_ksi(2),sigma_ksi(2));
plot(x2,ksi2prior,'b','LineWidth',1.5)
title('(b) $\xi_2^0\sim \mathcal{N}(-0.0024,5.71\times10^{-6})$','Interpreter','latex')
axes(h(4));box on;hold on;
histogram(ksi2,30,'Normalization','pdf');
x = linspace(pd2.mu-5*pd2.sigma,pd2.mu+5*pd2.sigma,1000);
y2 = normpdf(x,pd2.mu,pd2.sigma);
plot(x,y2,'r','LineWidth',1.5)
xlabel('$\xi_2$','Interpreter','latex')
title('(d) $\xi_2^1\sim \mathcal{N}(-0.0025,9.42\times10^{-11})$','Interpreter','latex')
% saveas(ff,'Figures\BMU_KdVN20.png')
% saveas(ff,'Figures\BMU_KdVN20.eps','epsc') 
% savefig(ff,'Figures\BMU_KdVN20.fig')

%% Uncertainty propagation
% This script propagates uncertainty obtained from BMU to system respnose
clear; clear global; close all; clc
load('.\Datasets\KdVN20_MCMC.mat')
ksi1 = ksiRec(1,101:end)';
ksi2 = ksiRec(2,101:end)';
clearvars -except ksi1 ksi2
global a1 a2 t k u0t
Lx = 2; Lt = 2;
Nx = 512; Nt = 201;
x = Lx/Nx*(-Nx/2:Nx/2-1);
k = 2*pi/Lx*[0:Nx/2-1 -Nx/2:-1].';
t = Lt/Nt*(0:Nt-1);
u0 = cos(pi*x);   % initial condition
u0t = fft(u0);
clear Lx Nx Lt Nt tt u0;
uRec = zeros(length(ksi1),length(x),length(t));
for i = 1:length(ksi1)
    i
    a1 = ksi1(i); % coeff of u_xx
    a2 = ksi2(i); % coeff of uu_x
    [t,utso1]=ode23tb('KdV',t,u0t,[],k);
    usol=ifft(utso1,[],2);
    u1=real(usol);
    uRec(i,:,:) = u1';
end
save('Datasets\KdVUP_N20_2.mat','ksi1','ksi2','uRec')

%% Visualize uncertainty in system responses
clear;clear global;clc
addpath 'G_code'
load('Datasets\KdVUP_N20_2.mat')
global a1 a2 t k u0t
Lx = 2; Lt = 2;
Nx = 512; Nt = 201;
x = Lx/Nx*(-Nx/2:Nx/2-1);
k = 2*pi/Lx*[0:Nx/2-1 -Nx/2:-1].';
t = Lt/Nt*(0:Nt-1);
u0 = cos(pi*x);   % initial condition
u0t = fft(u0);
clear Lx Nx Lt Nt tt u0;
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
uu=real(usol)';
%%
close all;clc;tt = 162;
ff = fig('units','inches','width',9,'height',4,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,3,[.10 .05],[.12 .06],[.06 .02]);
axes(h(1));box on;hold on;
for i=1:length(x)
L1 = scatter(x(i)*ones(size(uRec,1),1),squeeze(uRec(:,i,tt)),5,'MarkerEdgeColor','g',...
              'MarkerFaceColor','g');
end
L2 = plot(x,mean(squeeze(uRec(:,:,tt))),'r-','LineWidth',1.0);
legend([L1,L2],{'samples','mean'},'location','northwest')
legend('boxoff')
xlabel('$x$','Interpreter','latex')
ylabel('$u$','Interpreter','latex')
title('(a)','FontWeight','normal')
text(-0.8,1.8,'$t = 1.6$','Interpreter','latex')
axes(h(2));box on;hold on;
L1 = plot(x,uu(:,tt),'b--','LineWidth',1.5);
meanU = mean(squeeze(uRec(:,:,tt)));
L2 = plot(x,meanU,'r-','LineWidth',1.0);
% ylim([0.2 1.0])
stdU = std(squeeze(uRec(:,:,tt)));
curve1 = meanU+3*stdU;
curve2 = meanU-3*stdU;
x2 = [x,fliplr(x)];
inBetween = [curve1,fliplr(curve2)];
L3 = fill(x2,inBetween,'g');
alpha(.2)
legend([L1,L2,L3],{'truth','mean','mean$\pm$3std'},'Interpreter','latex','location','northwest')
legend('boxoff')
xlabel('$x$','Interpreter','latex')
title('(b)','FontWeight','normal')
axes(h(3));box on;hold on;
plot(x,uu(:,tt),'b--','LineWidth',2.5)
meanU = mean(squeeze(uRec(:,:,tt)));
plot(x,meanU,'r-','LineWidth',2.0)
% ylim([0.2 1.0])
stdU = std(squeeze(uRec(:,:,tt)));
curve1 = meanU+3*stdU;
curve2 = meanU-3*stdU;
x2 = [x,fliplr(x)];
inBetween = [curve1,fliplr(curve2)];
fill(x2,inBetween,'g');
alpha(.2)
xlabel('$x$','Interpreter','latex')
title('(c)','FontWeight','normal')
xlim([0 0.5])
set(gca,'ytick',-0.6:0.2:0.4)
% saveas(ff,'Figures\KdVUP_N20_2.png')
% saveas(ff,'Figures\KdVUP_N20_2.eps','epsc') 
% savefig(ff,'Figures\KdVUP_N20_2.fig')

%% functions
function err = errKdV(a)
global a1 a2 t k u0t uData
a1 = a(1);
a2 = a(2);
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
u1=real(usol);
err = reshape((u1-uData)/norm(uData),[],1);
end

function LLH = likelihood_ksi(ksi)
global sigma_ksi2 mu_ksi mu_e sigma_e2
err = errKdV(ksi);
Jksi = 1/2*sum(1./sigma_ksi2.*(ksi-mu_ksi).^2);
Jerr = 1/2*sum((err-mu_e).^2/sigma_e2);
LLH = -Jksi-Jerr;
LLH = double(LLH);
end