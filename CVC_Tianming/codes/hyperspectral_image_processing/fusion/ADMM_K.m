function [K,err] = ADMM_K(RHS_1,K0,D,FX,ind,IND,beta,tau,N1,N2,N3)

mu = 50*beta; % for TV

[p,q] = size(K0);

K = K0;

gg = D*K0(:); % vector
l1 = zeros(size(gg)); % vector

KK = K0;
L2 = zeros(p,q);

err = zeros(100,1);
    
for i = 1:999
    
    RHS_2 = mu*D.'*(gg+l1);
    RHS_2 = reshape(RHS_2,[p q]);
    RHS_2 = RHS_2+mu*(KK+L2);
    
    RHS = RHS_1+RHS_2;
    
    % update K
    K_pre = K;
    K = CG_K(RHS,K,D,FX,ind,IND,mu,tau,N1,N2,N3);
    
    % update g
    g = D*K(:);
    gg = soft_isotropic(g-l1,beta/(2*mu));
    
    % update KK
    k = K(:)-L2(:);
    
    u = sort(k,'descend'); % projection onto the probability simplex
    for j = 1:p*q
        s = sum(u(1:j));
        temp = u(j)+1/j*(1-s);
        if temp > 0
            lambda = temp-u(j);
        else
            break
        end
    end
    k = max(k+lambda,0);
    
    KK = reshape(k,p,q);
    
    err(i) = norm(K-K_pre,'fro')/norm(K,'fro');
    
    if err(i) < 1e-4
        err = err(1:i);
        fprintf('ADMM_K is successful in %3d iterations \n',i)
        break
    else        
        % update multipliers
        l1 = l1+gg-g;
        L2 = L2+KK-K;  
    end
     
end