% This is a toy example for verifying the implementation of the 
% TRAP-MUSIC algorithm (Makela, Stenroos, Sarvas, Ilmoniemi. Truncated RAP-MUSIC
% (TRAP-MUSIC) for MEG and EEG source localization. NeuroImage 167(2018):73--83. 
% https://doi.org/10.1016/j.neuroimage.2017.11.013
%
% The data simulation and scanning are in this example done using the same
% source model, same source discretization, same forward model and ideal
% noise. As the simulation omits all real-world errors and tests of
% robustness, this script serves only as a verification of the
% implementation and example for using the TRAP MUSIC. Please do not use
% this kind of over-simplified simulation for any serious assessment or
% method comparison.
%
% trapmusic_matlab/example_trapmusic.m
% trapmusic_matlab is licensed under BSD 3-Clause License.
% Copyright (c) 2020, Matti Stenroos.
% All rights reserved.
% The software comes without any warranty.
%
% v200424 Matti Stenroos, matti.stenroos@aalto.fi

%% Prepare a forward model
% Make a toy forward model that has 999 sources topographies and 60 sensors.
% With default parameters, this is actually quite a difficult toy model,
% having ~7 degrees-of-freedom.
clear
n_sens = 60;
n_sourcespace = 999;
alpha = (1:n_sens)/n_sens*2*pi;
phase_offset = rand(n_sourcespace,1)*2*pi;
omega_multip = 0.8+rand(n_sourcespace,1)*0.4;
L = zeros(n_sens,n_sourcespace);
for I=1:n_sens
    L(I,:) = sin(omega_multip.*(phase_offset+alpha(I)));
end

% Check how nasty the model turned out
s = svd(L*L');
relcond = s(1)./s;
dof = find(relcond>1e6,1,'first');
fprintf('\nThe forward model has approx. %d degrees of freedom.\n',dof);

%% Simulation 1, pre-set source orientation
% Assume that each topography (column) in L represents a source point with
% (assumed) known orientation. In this case, each source location has 
% only one possible topography, corresponding to "fixed orientation"
% in EEG/MEG language.

n_truesources = 5; %how many sources there are
n_iter = n_truesources + 2; %how many sources we guess there to be
n_rep = 10; %how many different runs are done
pSNR = 1; %power-SNR, pSNR=trace(C_idealmeas)/trace(C_noise)
C_source = 0.2*ones(n_truesources) + 0.8*eye(n_truesources); %source covariance

rng('default');
fprintf('\nSimulation: pSNR %.1f, %d sources, %d runs, pre-set orientation\n',pSNR,n_truesources,n_rep);
for I=1:n_rep
    % Make a multi-source set
    sourceinds_true = sort(randperm(n_sourcespace,n_truesources));
    % Generate measurement covariance matrix
    L_this = L(:,sourceinds_true);
    C0 = L_this*C_source*L_this';
    C_noise = trace(C0)/(pSNR*n_sens)*eye(n_sens);
    %pSNRtest = trace(C0)/trace(C_noise);
    C_meas = C0 + C_noise;
    %do a TRAP scan with pre-set orientations
    [sourceinds_trap, mu_max] = trapscan_presetori(C_meas, L, n_iter);

    %check how it went
    [~, subind_false] = setdiff(sourceinds_trap,sourceinds_true);
    subind_true = setdiff(1:n_iter,subind_false);
    n_false = numel(subind_false);
    mu_truemin = min(mu_max(subind_true));
    mu_falsemax = max(mu_max(subind_false));
    fprintf('%2d: found %d/%d sources, min(mu_true) = %.2f, max(mu_false) = %.2f\n',...
        I, n_iter - n_false, n_truesources, mu_truemin, mu_falsemax); 
end

%% Simulation 2, unknown / optimized orientation
% Now, assume that L has 333 source locations and each source location has
% three possible (orthogonal) topographies. For L, this situation
% corresponds to "free orientation" or "vector source". The TRAP MUSIC
% algorithm assumes that each source location has a constant unknown
% orientation, and searches for the orientation that most strongly projects
% to the signal space. This corresponds to the typical formulations with
% optimal-orientation scalar beamformers.

n_truesources = 5; %how many sources there are
n_iter = n_truesources + 2; %how many sources we guess there to be
n_rep = 10; %how many different runs are done
pSNR = 1; %power-SNR, pSNR=trace(C_idealmeas)/trace(C_noise)
C_source = 0.2*ones(n_truesources) + 0.8*eye(n_truesources); %source covariance

n_sourcespace = size(L,2)/3;
rng('default');
fprintf('\nSimulation: pSNR %.1f, %d sources, %d runs, optimized orientation\n',pSNR,n_truesources,n_rep);
for I=1:n_rep
    % Make a multi-source set
    sourceinds_true = sort(randperm(n_sourcespace,n_truesources));
    oritemp = rand(n_truesources,3)-.5;
    sourceoris_true = (oritemp./sqrt(sum(oritemp.^2,2)));  
    % Extract oriented sources & make forward mapping for them
    L_this = zeros(n_sens, n_truesources);
    for J=1:n_truesources
        L_local = L(:,3*sourceinds_true(J)+(-2:0));
        L_this(:,J)=L_local*sourceoris_true(J,:)';
    end
    % Generate measurement covariance matrix
    C0 = L_this*C_source*L_this';
    C_noise = trace(C0)/(pSNR*n_sens)*eye(n_sens);
    %pSNRtest = trace(C0)/trace(C_noise);
    C_meas = C0 + C_noise;
    %do a TRAP scan with optimized orientations
    [sourceinds_trap, mu_max, eta_mumax] = trapscan_optori(C_meas, L, n_iter);
    %check how it went
    [~, subind_false] = setdiff(sourceinds_trap,sourceinds_true);
    subind_true = setdiff(1:n_iter,subind_false);
    n_false = numel(subind_false);
    mu_truemin = min(mu_max(subind_true));
    mu_falsemax = max(mu_max(subind_false));
    
    [ind_match, ia, ib] = intersect(sourceinds_true,sourceinds_trap);
    oris_found = eta_mumax(ib,:);
    oris_ref = sourceoris_true(ia,:);
    oris_diff = acosd(round(abs(sum((oris_ref./sqrt(sum(oris_ref.^2,2))).*(oris_found./sqrt(sum(oris_found.^2,2))),2)),6));

    fprintf('%2d: found %d/%d sources, min(mu_true) = %.2f, max(mu_false) = %.2f\n',...
        I, n_iter - n_false, n_truesources, mu_truemin, mu_falsemax); 
    fprintf('    orientation errors: min %.2f deg, max %.2f deg.\n',...
        min(oris_diff),max(oris_diff)); 
    
end
