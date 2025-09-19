%% Generate a Random NMF problem
% This is a script for showing how to use the various functison in this
% repo.
% Feel free to run it and see how some of the algorithms work!

k = 25;
m = 1000;
n = 1000;

A = rand(2*k,k);
R1 = chol(A'*A);

A = rand(2*k,k);
R2 = chol(A'*A);

[W,~] = qr(normrnd(0,1,[m,k]),0);
[H,~] = qr(normrnd(0,1,[n,k]),0);

W = W*R1; W = W + abs(min(min(W)));
H = H*R1; H = H + abs(min(min(H)));

X = W*H';
N = normrnd(0,1,size(X));

u = 0.2;
X = X + u*(norm(X,'fro')/norm(N,'fro'))*N;

fprintf('Min X = %d\n', min(min(X))); % nonnegativity check

% LAI - NMF
[W0,H0,output0] = LAI_NMF(X,k);
[W1,H1,output1] = LAI_NMF(X,k,'p',5,'q',2);
[W7,H7,output7] = LAI_NMF(X,k,'p',5,'q',2,'do_AQB',true);
[W8,H8,output8] = LAI_NMF(X,k,'p',5,'q',2,'do_AQB',true,'alg_str','hals');

% fprintf('Min X = %d\n', min(min(X)));
figure; hold on;

cstruct = output0;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*b')

cstruct = output1;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*r')

cstruct = output7;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*g')

cstruct = output8;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*k')

title('Sample NMF Problem');
legend({'LAI-BPP-NMF','LAI-BPP-NMF-2','LAI-BPP-NMF-AQB','LAI-HALS-NMF-AQB'});
xlabel('Normalized Residual');
ylabel('Time in Seconds');
%% Do a symmetric Problem
% A = (X*X')/2;
W_old = max(0,randn(m,k));
A = W_old*W_old';


[W2,H2,output2] = LAI_NMF(A,k,'p',5,'q',2,'doSym',true);
[W3,H3,output3] = LAI_NMF(A,k,'p',5,'doSym',true,'do_AQB',true);
[H4,output4,U4] = LAI_SymPGNCG(A,k,'p',5,'q',2);
[W5,output5]    = lvs_symNMF(A,k); % leverage score sampling
[W6,output6]    = lvs_symNMF(A,k,'hybrid_sample',true);

figure; hold on;

cstruct = output2;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*r')

cstruct = output3;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*b')

cstruct = output4;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*g')

cstruct = output5;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*k')

cstruct = output6;
eiter = nnz(cstruct.relres(2:end));
plot(cumsum(cstruct.time(2:eiter)), cstruct.relres(2:eiter),'-*c')

legend({'LAI-SymNMF','LAI-SymNMF-AQB','LAI-PGNCG-SymNMF','LvS-SymNMF','LvS-hybrid-SymNMF'});
title('Sample SymNMF Problem');
xlabel('Normalized Residual');
ylabel('Time in Seconds');