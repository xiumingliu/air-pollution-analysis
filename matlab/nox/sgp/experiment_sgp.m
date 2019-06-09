%% Load data
clear all;
load('data.mat');

% Log-normal transform (only use data from sensor 1)
nox = nox(:, 1);
log_nox = log(nox(:, 1));

%% Configuration

% Noise level
epsilon = .1;

% Number of observations (weeks)
numObs = 50;

% Number of inducing variables (weeks)
numInd = 1; 

%% 24 hours predictions for year 2014
tic
parfor iteration = 1:364
    x_tst_begin = time(time == '2014-01-01') + (iteration - 1);
    x_tst_end = x_tst_begin + hours(23);
    x_tst_weekday = weekday(x_tst_begin);
    x_tst = [time(find(time == x_tst_begin)):1/24:time(find(time == x_tst_end))]';
    y_tst = nox(find(time == x_tst_begin):find(time == x_tst_end));
    log_y_tst = log_nox(find(time == x_tst_begin):find(time == x_tst_end));

    % Training data (N weeks)
    x_trn_begin = datetime('2009-01-01 00:00:00') + (x_tst_weekday + 2);
    x_trn_end = x_tst_begin - hours(1);
    x_trn = [time(find(time == x_trn_begin)):1/24:time(find(time == x_trn_end))]';
    y_trn = nox(find(time == x_trn_begin):find(time == x_trn_end));
    log_y_trn = log_nox(find(time == x_trn_begin):find(time == x_trn_end));

    % Observations (M weeks)
    x_obs_begin = datetime(x_tst_begin - calweeks(numObs)); 
    x_obs_end = x_tst_begin - hours(1);
    x_obs = [time(find(time == x_obs_begin)):1/24:time(find(time == x_obs_end))]';
    y_obs = nox(find(time == x_obs_begin):find(time == x_obs_end));
    log_y_obs = log_nox(find(time == x_obs_begin):find(time == x_obs_end));

    % Inducing variables
    index_ind = sort(datasample(1:numObs*7*24, numInd*7*24, 'Replace', false));
    x_ind = x_obs(index_ind);
    y_ind = y_obs(index_ind);

    %% GP learning

    % Sample mean and covariance functions
    log_COV = xcov(log_y_trn, 'biased');

    %% GP prediction

    tic
    log_m_tst_prior = func_mean(x_tst, log_y_trn);
    log_K_tst_prior = func_cov(x_tst, x_tst, log_COV);

    log_m_obs = func_mean(x_obs, log_y_trn);
    log_K_obs = func_cov(x_obs, x_obs, log_COV);

    log_K_tst_obs = func_cov(x_tst, x_obs, log_COV);
    log_K_obs_tst = log_K_tst_obs';

    log_K_ind = func_cov(x_ind, x_ind, log_COV);
    log_K_ind_tst = func_cov(x_ind, x_tst, log_COV); log_K_tst_ind = log_K_ind_tst';
    log_K_ind_obs = func_cov(x_ind, x_obs, log_COV);log_K_obs_ind = log_K_ind_obs';

    Lambda = diag(diag(log_K_obs - (log_K_obs_ind/(log_K_ind + epsilon*eye(size(log_K_ind))))*log_K_ind_obs)); 
    Sigma = inv(log_K_ind + (log_K_ind_obs/(Lambda + epsilon*eye(size(Lambda))))*log_K_obs_ind + epsilon*eye(size(log_K_ind)));

    % log_m_tst_post = log_m_tst_prior + log_K_tst_obs*inv(log_K_obs + epsilon*eye(size(log_K_obs)))*(log_y_obs - log_m_obs);
    % log_K_tst_post = log_K_tst_prior - log_K_tst_obs*inv(log_K_obs + epsilon*eye(size(log_K_obs)))*log_K_obs_tst;

    log_m_tst_post = log_m_tst_prior + (log_K_tst_ind*Sigma*log_K_ind_obs/(Lambda + epsilon*eye(size(Lambda))))*(log_y_obs - log_m_obs);
    log_K_tst_post = log_K_tst_prior - (log_K_tst_ind/(log_K_ind + epsilon*eye(size(log_K_ind))))*log_K_ind_tst + log_K_tst_ind*Sigma*log_K_ind_tst;

    % Transform back to log-normal
    m_tst_post = exp(log_m_tst_post + 0.5*diag(log_K_tst_post));
    K_tst_post = zeros(24, 24);
    for row = 1:24
        for col = 1:24
            K_tst_post(row, col) = exp(log_m_tst_post(row) + log_m_tst_post(col) + .5*(log_K_tst_post(row, row) + log_K_tst_post(col, col)))*(exp(log_K_tst_post(row, col)) - 1);
        end
    end
    toc
    
    errors(:, iteration) = m_tst_post - y_tst;
    display(iteration);
    
end
toc

fname = string(['results\errors-' num2str(numInd) 'weeksinducing-' num2str(numObs) 'weeksobserving']);
    
save(fname, 'errors');

histogram(errors); 
RMSE = sqrt(mean(mean(errors.^2)));