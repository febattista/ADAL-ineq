function [model] = dual_bound_init(A,At,b,C,L,mleq)

sense = strings(size(L));
sense(:, :) = '>';
sense(isinf(L)) = '=';
L(isinf(L)) = 0;

% LHS Model
sizeAt = size(At);
Agurobi = At;
Agurobi(:, mleq+1:end) = -Agurobi(:, mleq+1:end);
idx = any(Agurobi, 2);
Agurobi = Agurobi(idx, :);
model.A = Agurobi;
model.idx = idx;
model.L = L;

model.C = C;

% Objective Fun
obj = [A(1:mleq,:)*L(:) - b(1:mleq); b(mleq+1:end) - A(mleq+1:end, :)*L(:)];
model.obj = obj;

% Variables
model.sense = convertStringsToChars(strjoin(sense(idx), ''));
model.lb = [zeros(1, mleq), -Inf(1, sizeAt(2) - mleq)];
model.vtype = 'C';
model.modelsense = 'max';

end
