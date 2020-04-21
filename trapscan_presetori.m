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
% For further information, please see the paper. We also kindly ask you to 
% cite the paper, if you use the approach / this implementation.
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
        Qk = eye(n_sens)-l_found/(l_found'*l_found)*l_found';
         % The above formula gives a 'near-singular' warning in the case,
        % when there is not enough space left for rejection. This means
        % that you have used the degrees-of-freedom of your lead field
        % matrix i.e. you cannot project out more dimensions i.e. you are
        % trying to extract too large number of sources.
        %
        % If you prefer to not see the warning, comment the above out and
        % use the row below instead. But, I'd rather receive some kind of
        % warning, when the estimation is likely to go wrong...
%         Qk = eye(n_sens)-l_found*pinv(l_found);
    end
end