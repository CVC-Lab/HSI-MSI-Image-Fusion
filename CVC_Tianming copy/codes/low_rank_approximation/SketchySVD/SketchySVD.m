function [Q,s,P,cpu] = SketchySVD(A,r,k)
% implementation of SketchySVD in [Streaming low-rank matrix approximation 
% with an application to scientific simulation]

[M,N] = size(A);

s = 2*k+1; % (5.3)

cpu = zeros(1,8);

type = 1; 
% 1 - Gaussian maps
% 2 - Scrambled SRFT maps
% 3 - Sparse sign maps 

switch type
    
    case 1
        
        % form X
        tic;
        T1 = randn(k,M);
        X = T1*A; % k by N
        clear T1
        cpu(1) = toc;
        
        % form Y
        tic;
        T2 = randn(k,N);
        Y = A*T2';
        clear T2
        cpu(2) = toc;
        
        % form Z
        tic;
        T3 = randn(s,M);
        T4 = randn(s,N);
        if M < N
            Z = T3*(A*T4');
        else
            Z = (T3*A)*T4';
        end
        cpu(3) = toc;
        
        % QR of X
        tic;
        [P,~] = qr(X',0);
        cpu(4) = toc;
        
        % QR of Y
        tic;
        [Q,~] = qr(Y,0);
        cpu(5) = toc;
        
        % form C
        tic;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %L = pinv(T3*Q); % k by s
        %R = pinv(T4'*P); % k by s
        %C = L*Z*R'; % k by k
        L = T3*Q;
        clear T3
        R = T4*P;
        clear T4
        C = (L\Z)/R';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cpu(6) = toc;
        
        % SVD of C
        tic;
        [U,S,V] = svd(C);
        s = diag(S);
        cpu(7) = toc;
        
        % final
        tic;
        %Q = Q*U;
        Q = Q*U(:,1:r);
        %P = P*V;
        P = P*V(:,1:r);
        cpu(8) = toc;
        
    case 2
        
        % form X
        tic;
        [~,~,X] = SSRFT(A,k);
        cpu(1) = toc;
        
        % form Y
        tic;
        [~,~,Y] = SSRFT(A',k);
        Y = Y';
        cpu(2) = toc;
        
        % form Z
        tic;
        [d1,ind1,TEMP] = SSRFT(A,s);
        [d2,ind2,Z] = SSRFT(TEMP',s);
        Z = Z';
        %[d2,ind2,TEMP] = SSRFT(A',s);
        %[d1,ind1,Z] = SSRFT(TEMP',s);
        clear TEMP
        cpu(3) = toc;
        
        % QR of X
        tic;
        [P,~] = qr(X',0);
        cpu(4) = toc;
        
        % QR of Y
        tic;
        [Q,~] = qr(Y,0);
        cpu(5) = toc;
        
        % form C
        tic;
        TEMP1 = bsxfun(@times,d1,Q);
        TEMP2 = dct(TEMP1);
        clear TEMP1
        L = TEMP2(ind1,:);
        clear TEMP2
        TEMP1 = bsxfun(@times,d2,P);
        TEMP2 = dct(TEMP1);
        clear TEMP1
        R = TEMP2(ind2,:);
        clear TEMP2
        C = (L\Z)/R';
        cpu(6) = toc;
        
        % SVD of C
        tic;
        [U,S,V] = svd(C);
        s = diag(S);
        cpu(7) = toc;
        
        % final
        tic;
        %Q = Q*U;
        Q = Q*U(:,1:r);
        %P = P*V;
        P = P*V(:,1:r);
        cpu(8) = toc;
        
    case 3
        
        % form X
        tic;
        T1 = SSM(k,M);
        X = T1*A; % k by N
        clear T1
        cpu(1) = toc;
        
        % form Y
        tic;
        T2 = SSM(k,N);
        Y = A*T2';
        clear T2
        cpu(2) = toc;
        
        % form Z
        tic;
        T3 = SSM(s,M);
        T4 = SSM(s,N);
        if M < N
            Z = T3*(A*T4');
        else
            Z = (T3*A)*T4';
        end
        cpu(3) = toc;
        
        % QR of X
        tic;
        [P,~] = qr(X',0);
        cpu(4) = toc;
        
        % QR of Y
        tic;
        [Q,~] = qr(Y,0);
        cpu(5) = toc;
        
        % form C
        tic;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %L = pinv(T3*Q); % k by s
        %R = pinv(T4'*P); % k by s
        %C = L*Z*R'; % k by k
        L = T3*Q;
        clear T3
        R = T4*P;
        clear T4
        C = (L\Z)/R';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cpu(6) = toc;
        
        % SVD of C
        tic;
        [U,S,V] = svd(C);
        s = diag(S);
        cpu(7) = toc;
        
        % final
        tic;
        %Q = Q*U;
        Q = Q*U(:,1:r);
        %P = P*V;
        P = P*V(:,1:r);
        cpu(8) = toc;
        
    otherwise
        error('Test Matrices of Type Not Supported!')
        
end          