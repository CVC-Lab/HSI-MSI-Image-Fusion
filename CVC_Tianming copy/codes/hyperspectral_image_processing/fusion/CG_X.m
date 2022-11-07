function X = CG_X(H,X0,Fk,L,IND,N1,N2,N3,tau)

normH = norm(H(:));
tolH = 1e-3*normH;

r0 = H - LHS_X(X0,Fk,L,IND,N1,N2,N3,tau);
p0 = r0;

for i = 1:999
    
    pp = LHS_X(p0,Fk,L,IND,N1,N2,N3,tau);
    pp1 = p0(:)'*pp(:);
    a = (r0(:)'*r0(:))/pp1; % compute alpha_k
    
    X = X0+a*p0;
    r1 = r0-a*pp;
    
    res = norm(r1(:));
    if res < tolH
        fprintf('CG_X is successful in %3d iterations \n',i)
        break;
    else
        b1 = res^2/(r0(:)'*r0(:)); % compute beta_k
        p1 = r1+b1*p0;
        
        p0 = p1;
        r0 = r1;
        X0 = X;     
    end
    
end