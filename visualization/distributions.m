clear all; 
close all;

load('C:\Users\Administrator\Documents\GitHub\air-pollution-analysis\data\data.mat')

%% Distributions of no2, nox, o3, pm10, pm2_5
index_valid_no2 = isfinite(no2);
index_valid_nox = isfinite(nox);
index_valid_o3 = isfinite(o3);
index_valid_pm10 = isfinite(pm10);
index_valid_pm2_5 = isfinite(pm2_5);