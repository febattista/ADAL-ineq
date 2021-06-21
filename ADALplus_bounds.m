function [X, y, Z, S, primal, dual, lb_pp, iter, secs, secs_pp, secs_lb] = ADALplus_bounds(A, b, C, mleq, L, max_iter, tol, timelimit, sigma)
% ADAL: Alternating Direction Augmented Lagrangian method for SDP
%   
%       min <C, X> 
%   s.t.    A*X(:) <= b; 
%           A*X(:) = b;
%           X>=L, X psd
%
% A, b, C: A is m by n*n, b is m-vector, C is n by n
% mleq : number of <= constraints, we assume that they are from 1,...,mleq 

% some formal checking
[m, n2] = size(A); n = size(C, 1);
assert(length(b)==m, 'mismatch rows of A and length of b');
assert(n*n==n2, 'mismatch columns of A and size of C');
l_size = size(L);
assert((isempty(L) || l_size(1) == n), 'mismatch dimension on bounds L and U');
normb = norm(b); normC = norm(C(:));
tStart = tic;
secs_pp = 0;

if(nargin <= 5)
    max_iter=1000000;
end
if(nargin <= 6)
    tol=1e-5;
end
if(nargin <= 7)
    timelimit = 300000;
end

I = ones(m, 1);
poszeros = mleq+1:m;
I(poszeros) = 0;
% form A*A' (and cholesky, if not part of input)
if size(A,1) == m
  At = A';
else
  At = A; A = At';
end

%AAT = A * At + sparse(diag(I)); inefficient!!

AAT = A * At;

for i=1:mleq
    AAT(i,i) = AAT(i,i) + 1;
end
[R, p] = chol(AAT);
Rt = R';    % only for speed-up
secs = toc(tStart);
fprintf(' secs after chol:   %12.5f \n', secs);
if p>0 
    fprintf(' rows of A lin. dep. p = %4.0d, m = %4.0d\n',p,m);
    error(' nothing done');
end

% initialize \bar{X} = (X, x), dimension of X: (n) by (n)
X = zeros(n);
% slack variables of primal, dimension of x: (mleq) by (1)
% we pad x to dimension (m) by (1)
x = zeros(m, 1);         

% initialize \bar{Z} = (Z, z), dimension of Z: (n) by (n)
Z = zeros(n);
% surplus variables of dual, dimension of z: (mleq) by (1)
% we pad z to dimension (m) by (1)
z = zeros(m, 1);

% initialize S, dimension of S: (n) by (n) ... multipliers of X>=L
S = zeros(n);

% starting y
y = zeros(m, 1);

% inf_idx = find(isinf(L));
idx = find(~isinf(L));

lb_pp = -Inf;
secs_lb = 0;
model = post_processing_init(A,At,b,C,L,mleq);

done = 0;  % stopping condition not satisfied
iter = 0;  % iteration count
g = b - A*X(:) - x; % primal residue

if(nargin <= 8)
    G1 = Z - C;
    g1 = z;
    normG1 = norm( G1,'fro') + norm(g1);
    relp = norm(g)^2/(1+normb); 
    sigma = relp*(1+normC)/(normG1^2);
    fprintf('starting sigma: %10.7f \n', sigma);
end

%SIGMA BOX
sigma_min = 1e-4;
sigma_max = 1e+5;

fprintf('    it     time     dObj          pObj         dFeas    pFeas     X>=L     compXS    sigma\n');


% main loop
while done == 0 % while not done
    
    % update y 
    Mtmp = X/sigma - C + Z + S;    
    rs1 = b/sigma - A*Mtmp(:) - x/sigma - z; % -s;  
    tmp  = Rt\rs1;   
    y = R\tmp; 
    
    % determine new S
    Aty = reshape(At*y, n, n);
    T = C - Aty - Z - X/sigma + L/sigma;
    S = max(T, 0);
    
    % project W
    Aty = reshape(At*y, n, n);
    % Compute \bar{W} = (W, w), dimension of W: (n) by (n)
    W = Aty - C + X/sigma + S;
    % dimension of w: (mleq) by (1)
    w = x/sigma + y; % + s; 
    w = w(1:mleq);
    [Wp, wp, Wn, wn] = project_W(W, w);
    % computing new Z
    Z = -Wn; z(1:mleq) = -wn;
    % computing new X
    X = sigma * Wp; x(1:mleq) = sigma*wp;
    
    g = b - A*X(:) - x; % primal residue
    
    G = C - Z - Aty - S; % dual residue
    gg = - y - z; % - s;
    gg = gg(1:mleq);
    normX = norm(X(:));
    normx = norm(x);
    X1 = X - max(L, X);
    normX1 = norm(X1(:));
    err_1 = norm(g)/(1+normb);
    err_2 = (sqrt(norm(G(:))^2 + norm(gg)^2))/(1+normC);
    err_3 = (normX1)/(1+normX);
    XL = X(idx) - L(idx);
    normXL = norm(XL(:));
    XLt=XL.';
    trSX = S(idx).'*XLt(:);
    err_4 = abs(trSX)/(1+normXL + norm(S(:)));
    % fprintf( '%8.3f %8.3f  %8.3f %8.3f \n', trSX, normX, norm(S(:)), normx);
    
    err_0 = [err_1, err_2, err_3, err_4];
    primal = C(:)'*X(:); dual = b'*y + L(idx)'*S(idx);
    
    iter = iter + 1;
    secs = toc(tStart);

    done = (max(err_0)<tol) | (iter>= max_iter) | (secs >= timelimit); %done=1 if max{rp,rD}<tol or iter>maxiter

    % print something
    if (mod(iter, 100)==0) %modulus after division iter/10==0 -> stampa ogni 10 iterazioni
      fprintf( '%6.0d %8.2f %13.5e %13.5e %8.3f %8.3f  %8.3f %8.3f %9.6f \n', iter, ...
       secs, dual, primal, log10(err_0(2)), log10(err_0(1)), log10(err_0(3)),...
       log10(err_4), sigma);
    end
    % sigma update
    ratio = (sqrt(norm(X(:))^2 + norm(x)^2))/(sqrt(norm(Z(:))^2 + norm(z)^2));
    % weight
    w = 2^(-(iter-1)/100);
    %w = 1;
    %w = 0.02;
    sigma = (1-w)*sigma + w*max(sigma_min,min(ratio,sigma_max)) ;
    
    if (mod(iter,200)==0) || done
        tic;
        new_lb_pp = post_processing(model,Z);
        if round(new_lb_pp,2) > round(lb_pp,2) 
            lb_pp = new_lb_pp;
            secs_lb = toc(tStart);
        end
        secs_pp = secs_pp + toc;
        
        fprintf('------------------------ Valid bound: %13.5e @ %8.2f secs ------------------------------\n', lb_pp, secs_lb);
    end    
    
end

if max(err_0) < tol
    fprintf(' required accuracy reached after %6.0d iterations,',iter)
else
    fprintf(' Time limit or Maximum number of iteration exceeded.')
end

fprintf('    total time: %8.2f \n', secs);

end
