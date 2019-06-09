%% Load data
clear all;
load('data.mat');

% Log-normal transform (only use data from sensor 1)
nox = nox(:, 1);
log_nox = log(nox(:, 1));

%% Configuration

% Noise level
epsilon = .1;

% Number of observations
numObs = 50;

% Testing data (24 hours)
x_tst_begin = datetime('2014-03-12 00:00:00');
x_tst_end = datetime('2014-03-12 23:00:00');
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

log_m_tst_post = log_m_tst_prior + (log_K_tst_obs/(log_K_obs + epsilon*eye(size(log_K_obs))))*(log_y_obs - log_m_obs);
log_K_tst_post = log_K_tst_prior - (log_K_tst_obs/(log_K_obs + epsilon*eye(size(log_K_obs))))*log_K_obs_tst;

% Transform back to log-normal
m_tst_post = exp(log_m_tst_post + 0.5*diag(log_K_tst_post));
K_tst_post = zeros(24, 24);
for row = 1:24
    for col = 1:24
        K_tst_post(row, col) = exp(log_m_tst_post(row) + log_m_tst_post(col) + .5*(log_K_tst_post(row, row) + log_K_tst_post(col, col)))*(exp(log_K_tst_post(row, col)) - 1);
    end
end
toc

%% Plot the result

% Plot results
figure;
hold on;
grid on;
f = [exp(log_m_tst_post+2*sqrt(diag(log_K_tst_post)));...
    flipdim(exp(log_m_tst_post-2*sqrt(diag(log_K_tst_post))),1)]; 
h1 = fill([x_tst; flipdim(x_tst,1)], f, [7 7 7]/8);
h2 = plot(x_tst, y_tst, 'xk', x_tst, m_tst_post, '-+r', 'LineWidth', 3, 'MarkerSize', 16); 
ylim([0 1000]); xlim([datetime(x_tst_begin) datetime(x_tst_end)]);
legend(h2, 'True values', 'Predictive mean', 'Location', 'northwest');
xlabel('time'); ylabel('NOx concentrations (\mug/m^3)'); 
set(gca,'FontSize',26,'fontweight','bold');

axes('Position',[.5 .65 .39 .25])
box on
hold on;
grid on;
h3 = plot(x_obs, y_obs, 'o', 'Color', [.6 .6 .6], 'LineWidth', 1, 'MarkerSize', 3);
h4 = plot(x_tst, y_tst, 'xk', x_tst, m_tst_post, '-+r', 'LineWidth', 3, 'MarkerSize', 3);  
ylim([0 1000]); xlim([x_obs_begin x_tst_end]);
set(gca,'FontSize',20,'fontweight','bold');

figure; 
imagesc(K_tst_post); colorbar; 
set(gca,'FontSize',26,'fontweight','bold');