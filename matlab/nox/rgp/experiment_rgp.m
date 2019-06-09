%% Load data
clear all;
load('data.mat');

% Log-normal transform (only use data from sensor 1)
nox = nox(:, 1);
log_nox = log(nox(:, 1));

%% Configuration

% Noise level
epsilon = .1;

% Number of observations per batch
numObs = 35;

% Number of batches
numBat = 2;

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

    % Observations (numObs*numBat weeks)
    x_obs_begin = datetime(x_tst_begin - calweeks(numObs*numBat)); 
    x_obs_end = x_tst_begin - hours(1);
    x_obs = [time(find(time == x_obs_begin)):1/24:time(find(time == x_obs_end))]';
    y_obs = nox(find(time == x_obs_begin):find(time == x_obs_end));
    log_y_obs = log_nox(find(time == x_obs_begin):find(time == x_obs_end));

    %% GP learning

    % Sample mean and covariance functions
    log_COV = xcov(log_y_trn, log_y_trn, 'biased');

    %% GP prediction

    tic
    % Initial Prior
    log_m_tst_prior = func_mean(x_tst, log_y_trn);
    log_K_tst_prior = func_cov(x_tst, x_tst, log_COV);

    this_log_m_tst_prior = log_m_tst_prior;
    this_log_K_tst_prior = log_K_tst_prior;

    index_obs = 1:numObs*numBat*7*24;
    for i = 1:numBat
    %     this_x_obs = x_obs(((i-1)*numObs*7*24+1):(i*numObs*7*24));
    %     this_log_y_obs = log_y_obs(((i-1)*numObs*7*24+1):(i*numObs*7*24));

        this_x_obs = x_obs(i:numBat:end);
        this_log_y_obs = log_y_obs(i:numBat:end);

    %     this_index_obs = datasample(index_obs, numObs*7*24, 'Replace', false);
    %     index_obs = setdiff(index_obs, this_index_obs);
    %     this_x_obs = x_obs(this_index_obs);
    %     this_log_y_obs = log_y_obs(this_index_obs);

        this_log_m_obs = func_mean(this_x_obs, log_y_trn);
        this_log_K_obs = func_cov(this_x_obs, this_x_obs, log_COV);

        this_log_K_tst_obs = func_cov(x_tst, this_x_obs, log_COV);
        this_log_K_obs_tst = this_log_K_tst_obs';

        this_H = this_log_K_obs_tst/(log_K_tst_prior + epsilon*eye(size(log_K_tst_prior)));
        this_s = this_log_m_obs - this_H*log_m_tst_prior;
        this_G = (this_log_K_obs - this_H*this_log_K_tst_obs) + this_H*this_log_K_tst_prior*this_H';

        % Update the posterior
        log_m_tst_post = this_log_m_tst_prior + (this_log_K_tst_prior*this_H'/(this_G + epsilon*eye(size(this_G))))*(this_log_y_obs - (this_H*this_log_m_tst_prior + this_s));
        log_K_tst_post = this_log_K_tst_prior - (this_log_K_tst_prior*this_H'/(this_G + epsilon*eye(size(this_G))))*this_H*this_log_K_tst_prior;

        if i < numBat
            % Update the prior for next batch
            this_log_m_tst_prior = log_m_tst_post;
            this_log_K_tst_prior = log_K_tst_post;
        end

        disp(['Batch number ' num2str(i)])
    end
    toc

%     clear this_*

    % Transform back to log-normal
    m_tst_post = exp(log_m_tst_post + 0.5*diag(log_K_tst_post));
    K_tst_post = zeros(24, 24);
    for row = 1:24
        for col = 1:24
            K_tst_post(row, col) = exp(log_m_tst_post(row) + log_m_tst_post(col) + .5*(log_K_tst_post(row, row) + log_K_tst_post(col, col)))*(exp(log_K_tst_post(row, col)) - 1);
        end
    end
    
    errors(:, iteration) = m_tst_post - y_tst;
    display(iteration);

end
toc

fname = string(['results\errors-' num2str(numObs) 'weeksperbatch-' num2str(numBat) 'batches']);

save(fname, 'errors');

histogram(errors); 
RMSE = sqrt(mean(mean(errors.^2)));
