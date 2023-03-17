%%%%%%%%%%% TO GENERATE AND SOLVE MODELS
% This code generates random (feasible) SDPs using the rand_sdps function.
% Since rand_sdps generates models suited for ADAL, this code then translate 
% it for SDPNAL+
%%%%%%%%%%%

%%% SDPNAL Options
% restoredefaultpath;
addpath(genpath(strcat(pwd, '/SDPNAL+v1.0')),path);
OPTIONS = SDPNALplus_parameters;

work_dir = fullfile('Rand_SDP');

% rand_sdps parameters 
n = [200];
m = [10000];
mleq = [.25, .5, .75];
rand_seed = [234, 345, 456, 567];
p = 3;

for i=1:length(n)
    for j=1:length(m)
       for k=1:length(mleq)
            for r=1:length(rand_seed)
                fprintf('%d \t %d \t %d \t %d \n', n(i), m(j), mleq(k), rand_seed(r));
                fprintf('ADAL\n');
                ineq = m(j) * mleq(k);
                dim = (n(i)*(n(i)+1))/2;
                [A, b, C] = rand_sdps(n(i), m(j), ineq, p, rand_seed(r));
                L = zeros(n(i));
                % save(fullfile(work_dir, strcat('adal_rand_', num2str(n(i)), '_' ,num2str(m(j)), '_' ,num2str(mleq(k)*100), '_' ,num2str(rand_seed(r)))), ...
                %     'A', 'b', 'C', 'ineq', 'L');
                % ADAL
                [X, y, Z, primal, iter, secs, status] = ADALplus_bounds(A, b, C, ineq, L, 10000, 1e-5, 3600, 1000, 1000);
                fprintf('SDPNAL\n');
                Bt = sparse(dim, ineq);
                for ii=1:ineq
                    Bt(:, ii) = svec({'s', n(i)}, reshape(A(ii, :), n(i), n(i)));
                end
                u = b(1:ineq);

                if m(j) - ineq > 0
                    At = sparse(dim, m(j) - ineq);
                    for ii=1:(m(j)-ineq)
                        At(:, ii) = svec({'s', n(i)}, reshape(A(ii+ineq, :), n(i), n(i)));
                    end
                    b = b(ineq+1:end); 
                end
                s = n(i);
                % save(fullfile(work_dir, strcat('sdpnal_rand_', num2str(n(i)), '_' ,num2str(m(j)), '_' ,num2str(mleq(k)*100), '_' ,num2str(rand_seed(r)))), ...
                %    'At', 'b', 'Bt', 'u', 'C', 's');
               	% SDPNALPLUS
               	[obj,X,s,y,S,Z,ybar,v,info,runhist] = sdpnalplus({'s', n},{At},{C},b, 0,[],{Bt},[],u, OPTIONS);
           end
        end
    end
end