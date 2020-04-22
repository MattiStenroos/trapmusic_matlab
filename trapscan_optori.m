function [ind_max,mu_max,eta_max,mus] = trapscan_optori(C_meas,L_scan,n_iter,dim_L)
%TRAPSCAN_OPTORI performs TRAP MUSIC source scan, fitting optimal source orientations
%
%[maxind,mumax,mus] = TRAPSCAN_OPTORI(C_meas,L_scan,n_iter,dim_L)
%
%   C_meas  = measurement covariance matrix, [n_sens x n_sens]
%   L_scan  = lead field matrix used for scanning,  [n_sens x (dim_l*n_scan)]
%   n_iter  = how many TRAP iterations are carried out; >= number of expected
%             sources
%   dim_L   = optional, dimension of the source space per position;
%            default 3, corresponding to orthogonal source triplets
%
%   ind_max = indices to strongest topographies ~ source locations, [NITER x 1]
%   mu_max  = scanning-function value for those positions, [NITER x 1]
%   eta_max = estimated source orientations for those positions, [NITER x Ldim]
%   mus    = scanning-function values for the whole scanning space
%
% Based on  
% Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC (TRAP-MUSIC) for
% MEG and EEG source localization. NeuroImage 167(2018):73--83.
% https://doi.org/10.1016/j.neuroimage.2017.11.013
% For further information, please see the paper. I also kindly ask you to 
% cite the paper, if you use the approach and/or this implementation.
% If you do not have access to the paper, please send a request by email.
%
% trapmusic_matlab/trapscan_optori.m
% trapmusic_matlab is licensed under BSD 3-Clause License.
% Copyright (c) 2020, Matti Stenroos.
% All rights reserved.
% The software comes without any warranty.
%
% v200422 Matti Stenroos, matti.stenroos@aalto.fi

if nargin<4
    dim_L = 3;
end

%number of sensors, number of source locations to scan
n_sens = size(L_scan,1);
n_scan = size(L_scan,2)/dim_L;

%output variables
ind_max = zeros(n_iter,1);
mu_max = zeros(n_iter,1);
eta_max = zeros(n_iter,dim_L);
mus = zeros(n_scan,n_iter);

%other arrays
% etas = zeros(Nscan,Ldim);
B = zeros(n_sens,n_iter);

%SVD & space of the original covariance matrix
[Utemp,~,~] = svd(C_meas);
Uso = Utemp(:,1:n_iter);

%The subspace basis and lead field matrix for k:th iteration
Uk = Uso;
L_this = L_scan;
for ITER = 1:n_iter
    %Subspace projection, removing previously found
    %topographies
    if ITER>1
        %apply out-projection to forward model
        L_this = Qk*L_scan;
        [Us,~,~] = svd(Qk*Uso,'econ');
        %TRAP truncation
        Uk = Us(:,1:(n_iter-ITER+1));
    end
    
    %Projector to this subspace
    %Ps = Uk*Uk';
    UkL_this = Uk'*L_this;

    %Scan over all test sources
    for I = 1:n_scan
        %if a source has already been found for this location, skip
        if any(ind_max(1:ITER-1)==I),continue;end
        %local lead field matrix for this source location
        L = L_this(:,(dim_L*I-dim_L+1):dim_L*I);
        UkL = UkL_this(:,(dim_L*I-dim_L+1):dim_L*I);
        %find the optimal orientation
        dtest = eig(UkL'*UkL,L'*L,'chol');
        maxcompindtest = 1;
        for J = 2:dim_L
            if dtest(J)>dtest(maxcompindtest)
                maxcompindtest = J;
            end
        end
        mus(I,ITER) = dtest(maxcompindtest);
    end
    %find the location index & orientation with the largest mu value
    [mm,mi] = max(mus(:,ITER));
    %generate these variables & get the orientation
    L = L_this(:,(dim_L*mi-dim_L+1):dim_L*mi);
    UkL = UkL_this(:,(dim_L*mi-dim_L+1):dim_L*mi);
    [Utest,dtest] = eig(UkL'*UkL,L'*L,'chol','vector');
    [~,maxcompindtest] = max(dtest);
    meta = Utest(:,maxcompindtest);
    
    meta = meta/norm(meta);
    mu_max(ITER) = mm;
    ind_max(ITER) = mi;
    eta_max(ITER,:) = meta;
    
    %make the next out-projector
    if ITER<n_iter
        L = L_scan(:,(dim_L*mi-dim_L+1):dim_L*mi);
        B(:,ITER) = L*meta;
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
            fprintf('trapscan_optori: Ill conditioning in out-projection at iteration %d.\n',ITER);
            fprintf('                 You are likely trying to find more sources than the model supports.\n');
            Qk = eye(n_sens) - l_found*U(:,1:keep)*diag(1./s(1:keep))*U(:,1:keep)'*l_found';
        else
            %normal pseudoinverse and out-projection
            Qk = eye(n_sens) - l_found*U*diag(1./s)*U'*l_found';
        end         
    end
end