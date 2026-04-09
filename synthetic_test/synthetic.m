clear;clc;

m = 1000;
r = 30;

for i=1:1
    W_sketch = max(randn(m,r), 0);
    V = W_sketch*W_sketch';

    W0 = 2 * full(sqrt(mean(mean(V)) / r)) * rand(m, r);
    H0 = W0';

    [~,~,output_SNMPBB,~] = NMF_SNMPBB(V,r,'W_INIT',W0,'H_INIT',H0);
    params.Hinit = W0;
    [~,output_ANLS,~] = symnmf_anls(V,r,params);
    [~,output_Newton,~] = symnmf_newton(V,r,params);
    [~,output_PGD,~] = PGD(V,r);



end
set(groot, 'DefaultLineLineWidth', 2)
plot(output_SNMPBB.total_time, output_SNMPBB.relres); hold on;
plot(output_ANLS.total_time, output_ANLS.relres);
plot(output_Newton.total_time, output_Newton.relres);
plot(output_PGD.total_time, output_PGD.relres);
legend("SNMPBB", "ANLS", "Newton", "PGD");
hold off