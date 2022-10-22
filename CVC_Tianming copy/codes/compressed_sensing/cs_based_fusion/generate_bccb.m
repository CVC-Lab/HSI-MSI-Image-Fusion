function bccb = generate_bccb(nx,ny,psf)
% generate BCCB matrix for 2D convolution with periodic boundary

[nbx,nby] = size(psf);
if nbx == nby
    nb = nbx;
    if mod(nb,2)
        l = (nb+1)/2; % center of the blur kernel
    else
        error('Kernel size should be odd!')
    end
else
    error('Kernel size should be square!')
end

indx = [0:l-1 nx+1-l:nx-1];
indy = [0:l-1 ny+1-l:ny-1];

[ix,jx] = ndgrid((0:nx-1).',(0:nx-1).');
nij = mod(ix-jx,nx); % toeplitz([0:nx-1],[0 fliplr([1 nx-1])])
	
inc = repmat((0:ny-1)*nx,[nb*nx 1]);
inc = inc(:); % row increment

mkx = false(nx); % indicates the nonzero locations on every block
for kk = indx
    mkx = mkx | (nij == kk);
end

len = nb*nb*nx*ny;
lenx = nb*nx*ny;
    
ii = zeros(len,1);
jj = zeros(len,1);
bb = zeros(len,1);

T = zeros(nb*nx,nb);

for i = 0:nb-1     
    
    if i <= l-1
        j = l+i;
    else
        j = l+i-nb;
    end
    
    temp = psf(:,j);
    
    c = zeros(nb,1);
    c(1:l) = temp(l:nb);
    c(l+1:nb) = temp(1:l-1);
    
    C1 = zeros(nb,l-1);
    C1(:,1) = c;
    for k = 2:l-1
        C1(:,k) = circshift(c,k-1);
    end
        
    C2 = repmat(temp,[1 nx-2*(l-1)]);
    
    c = zeros(nb,1);
    c(1) = temp(nb);
    c(2:nb) = temp(1:nb-1);
    
    C3 = zeros(nb,l-1);
    C3(:,1) = c;
    for k = 2:l-1
        C3(:,k) = circshift(c,k-1);
    end
    
    T(:,i+1) = [C1(:);C2(:);C3(:)];

end

for kk = 0:nb-1
    
    it = ix(mkx) + indy(kk+1)*nx;
    ii(kk*lenx+1:(kk+1)*lenx) = mod(repmat(it,[ny 1]) + inc,nx*ny);
    
    jt = jx(mkx);
    jj(kk*lenx+1:(kk+1)*lenx) = repmat(jt,[ny 1]) + inc;

    bb(kk*lenx+1:(kk+1)*lenx) = repmat(T(:,kk+1),[ny 1]);
  
end

bccb = sparse(ii+1,jj+1,bb); 