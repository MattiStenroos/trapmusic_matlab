function [ind_max,mu_max,mus] = trapscan_presetori(C_meas,L_scan,n_iter)
%TRAPSCAN_PRESETORI performs TRAP MUSIC source scan using pre-set source orientations
%
%[ind_max,mu_max,mus] = TRAPSCAN_PRESETORI(Covmat,L_scan,n_iter)
%
%   C_meas =  measurement covariance matrix, [n_sens x n_sens]
%   L_scan =  lead field matrix used for scanning,  [n_sens x n_scan]
%   n_iter =  how many TRAP iterations are carried out; > =  number of expected
%              sources
%
%   ind_max = indices to strongest topographies ~ source locations, [n_iter x 1]
%   mu_max  = scanning-function value for those positions, [n_iter x 1]
%   mus     = scanning-function values for the whole scanning space
%
% Based on  
% Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
% MEG and EEG source localization. NeuroImage 167(2018):73--83.
% https://doi.org/10.1016/j.neuroimage.2017.11.013
% For further information, please see the paper. I also kindly ask you to 
% cite the paper, if you use the approach and/or this implementation.
% If you do not have access to the paper, please send a request by email.
%
% trapmusic_matlab/trapscan_presetori.m
% trapmusic_matlab is licensed under BSD 3-Clause License.
% Copyright (c) 2020, Matti Stenroos.
% All rights reserved.
% The software comes without any warranty.
%
% v200421 Matti Stenroos, matti.stenroos@aalto.fi


%number of sensors, number of source locations to scan
[n_sens,n_scan] = size(L_scan);

%output variables
ind_max = zeros(n_iter,1);
mu_max = zeros(n_iter,1);
mus = zeros(n_scan,n_iter);

%other arrays
B = zeros(n_sens,n_iter);

%SVD & space of the measurement covariance matrix
[Utemp,~,~] = svd(C_meas);
Uso = Utemp(:,1:n_iter);

%the subscace basis and lead field matrix for k:th iteration
L_this = L_scan;
Uk = Uso;
for ITER = 1:n_iter
    %subspace projection, removing previously found
    %topographies 
    if ITER>1
        %apply out-projection to forward model
        L_this = Qk*L_scan;
        [Us,~,~] = svd(Qk*Uso,'econ');
        %TRAP truncation
        Uk = Us(:,1:(n_iter-ITER+1));
    end    
    %scan over all test sources, matrix form.
    %in principle we would use a subspace projector Ps = Uk*Uk', but as
    %only norms are needed, it is faster to use Uk directly.
    L_this_normsq = sum(L_this.*L_this,1);
    PsL_normsq = sum((Uk'*L_this).^2,1);
    
    mus(:,ITER) = PsL_normsq./L_this_normsq;
    %with poorly-visible sources, numerical behavior might lead to
    %re-finding the same source again (despite out-projection) -> remove
    mus(ind_max(1:ITER-1),ITER)=0;
    [mu_max(ITER), ind_max(ITER)] = max(mus(:,ITER));
  
    %make the next out-projector
    if ITER<n_iter
        B(:,ITER) = L_scan(:,ind_max(ITER));
        l_found = B(:,1:ITER);
        
        % Next, out-projecting. Typically it would go like this:
        % Qk = eye(n_sens)-l_found/(l_found'*l_found)*l_found';
        % But, this formula gives a 'near-singular' warning in the case,
        % when the topographies of the already-found sources are (nearly)
        % linearly dependent. This could happed due to two or more (nearly)
        % identical source topographies or due to trying to separate more
        % sources than the forward model supports. To avoid the Matlab warning,
        % let us analyze the condition and inform about ill conditioning.
        % We could also use 'pinv', but then we would not know about the problem.
        
        %check for the conditioning
        [U,S] = svd(l_found'*l_found,'econ');
        s = diag(S);
        tol = 10*eps(s(1))*ITER; %a bit lower tolerance than in 'pinv'
        keep = sum(s>tol);
        if keep<ITER
            %truncated pseudoinverse and out-projection
            fprintf('trapscan_presetori: Ill conditioning in out-projection at iteration %d; truncating.\n',ITER);
            fprintf('                    You are likely trying to find more sources than the model supports.\n');
            Qk = eye(n_sens) - l_found*U(:,1:keep)*diag(1./s(1:keep))*U(:,1:keep)'*l_found';
        else
            %normal pseudoinverse and out-projection
            Qk = eye(n_sens) - l_found*U*diag(1./s)*U'*l_found';
        end         
    end
end