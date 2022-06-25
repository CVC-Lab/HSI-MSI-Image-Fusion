function K = CG_K(H,K0,D,FX,ind,IND,mu,tau,N1,N2,N3)

normH = norm(H(:));
tolH = 1e-4*normH;

r0 = H - LHS_K(K0,D,FX,ind,IND,mu,tau,N1,N2,N3);
p0 = r0;

for i = 1:999
    
    pp = LHS_K(p0,D,FX,ind,IND,mu,tau,N1,N2,N3);
    pp1 = p0(:)'*pp(:);
    a = (r0(:)'*r0(:))/pp1; % compute alpha_k
    
    K = K0+a*p0;
    r1 = r0-a*pp;
    
    res = norm(r1(:));
    if res < tolH
        break;
    else       
        b1 = res^2/(r0(:)'*r0(:)); % compute beta_k
        p1 = r1+b1*p0;
        
        p0 = p1;
        r0 = r1;
        K0 = K;
    end
      
end