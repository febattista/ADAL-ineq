function [lb] = dual_bound(model, Z)
n = size(model.C,1);
% % compute eigenvalue decomposition of Z
Z = 1/2*(Z + Z');
[vv,lamvMatrix] = eig(Z);
ZSdp  = vv*max(zeros(n,n),lamvMatrix)*vv';
ZSdp = 1/2*(ZSdp + ZSdp');
%ZSdp = round(ZSdp, 3);

% RHS
rhs = ZSdp-model.C;
rhs = rhs(model.idx);
model.rhs = rhs;

params.outputflag = 0;
% params.resultfile = 'test.lp';
params.FeasibilityTol = 1e-5;

result = gurobi(model, params);

if strcmp(result.status,'OPTIMAL')
    % LB found
    p = model.C-ZSdp;
    lb = result.objval + model.L(:)'*p(:);
else
    % LB NOT found
    lb = -inf;
end

end
