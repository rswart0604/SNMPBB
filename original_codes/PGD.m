function [U,e,t,acc] = PGD(X,U,maxiter,truelabel);
%losses=zeros(maxiter,1);

% Initialization
etime = tic; 
[m,r] = size(U);
e = []; t = []; iter = 0; 

%initial error
e = [e 0.5*norm(X-U*U','fro')^2/norm(X,'fro')^2]; t = [t 0];

if nargin >= 4
    [~,label] = max(U, [], 2);
    tempacc = ClusteringMeasure(truelabel, label);
    acc(1) = tempacc(1);
end

% Main loop
%stepsize = 0.02;
while iter <= maxiter 
    %---one column by one column update----- 
    
    %losses(iter+1)=norm(X-U*U', 'fro')^2 ;
    
    step=1/(norm(X-U*U')+2*norm(U'*U))
    %step = stepsize;
    U=max(U-step*(U*U'*U-X*U),0);
%     grad = (U*U'- X)*U;
%     if norm(grad,'fro')>1
%         U = U - grad/norm(grad,'fro');
%     else
%         U = U - grad;
%     end
%     
%     
%     function [H,losses] = matrixFac_pos(X, itr, H)
%     losses=zeros(itr,1);
%     for t=1:itr
%         losses(t)=norm(X-H*H','fro')^2;
%         step=1/(norm(X-H*H')+2*norm(H'*H));
%         H=max(H-step*(H*H'*H-X*H),0);
%     end
    
    
%     U = U - 1e-3*grad;
    %projection onto positive constraints 
    %U = max(U,0);
    if nargout >= 3
        e = [e 0.5*norm(X-U*U','fro')^2/norm(X,'fro')^2];
        t = [t toc(etime)];
        
        if nargin >= 4
            [~,label] = max(U, [], 2);
            tempacc = ClusteringMeasure(truelabel, label);
            acc(iter+2) = tempacc(1);
        end
    end
    iter = iter + 1; 
end

