function LB = primLB(dual, At,C,b,X,y,Z,Z2,mleq,mu,Xbar)
% adapted from Kim-Chuan Toh's code (see below)
% October 2019

n = size(C,1);
if (nargin < 10); mu = 1.1; end
if (nargin < 11);  Xbar = mu*max(eig(full(X))); end

%fprintf('MAX EIG CURR X %10.5e\n', max(eig(full(X))));
Aty = reshape(At*y,n,n);
Znew = C - Aty - Z2;
znew = - y; % - s;
znew = znew(1:mleq);
%fprintf(' Construct Znew from y and S');

LB0 = dual;
pert = 0; 

eigtmp = eig(full(Znew)); 
idx = find(eigtmp < 0); 

numneg = length(idx); 
if (numneg) 
  mineig = min(eigtmp(idx)); 
  %fprintf('\n numneg_ev in Znew = %3.0d,  mineigZnew = %- 3.2e',numneg,mineig);
  pert = pert + Xbar*sum(eigtmp(idx)); 
end
idx = find(znew < 0); 

numneg = length(idx); 
if (numneg) 
  mineig = min(znew(idx)); 
  %fprintf('\n numneg_ev in Znew = %3.0d,  mineigZnew = %- 3.2e',numneg,mineig);
  pert = pert + Xbar*sum(znew(idx)); 
end

LB = LB0 + pert;

%fprintf('\n dual obj = %-10.9e  \n valid LB = %-10.9e\n',LB0,LB); 
%%*****************************************************************


% $$$ %%*****************************************************************
% $$$ %% primLB: compute a lower bound for the exact primal
% $$$ %%         optimal value. 
% $$$ %%*****************************************************************
% $$$ 
% $$$ function LB = primLB(blk,At,C,b,X,y,Z2,mu); 
% $$$ 
% $$$   if (nargin < 8); mu = 1.1; end
% $$$   Aty = sdpnalAtyfun(blk,At,y);
% $$$   Znew = ops(C,'-',Aty); 
% $$$   if ~isempty(Z2)
% $$$      Znew = ops(Znew,'-',Z2); 
% $$$   end
% $$$ 
% $$$   LB0 = b'*y; 
% $$$   pert = 0; 
% $$$   for p = 1:size(blk,1)
% $$$      pblk = blk(p,:);
% $$$      if strcmp(pblk{1},'s')
% $$$         eigtmp = eig(full(Znew{p})); 
% $$$         idx = find(eigtmp < 0); 
% $$$         Xbar = mu*max(eig(full(X{p}))); 
% $$$      elseif strcmp(pblk{1},'l')
% $$$         eigtmp = Znew{p};
% $$$         idx = find(eigtmp < 0); 
% $$$         Xbar = mu*max(X{p}); 
% $$$      end      
% $$$      numneg = length(idx); 
% $$$      if (numneg) 
% $$$         mineig = min(eigtmp(idx)); 
% $$$         fprintf('\n numneg = %3.0d,  mineigZnew = %- 3.2e',numneg,mineig);
% $$$         pert = pert + Xbar*sum(eigtmp(idx)); 
% $$$       end
% $$$       fprintf('\n Xbar = %f\n', Xbar);
% $$$   end
% $$$   LB = LB0 + pert; 
% $$$   fprintf('\n dual obj = %-10.9e  \n valid LB = %-10.9e\n',LB0,LB); 
% $$$ %%*****************************************************************