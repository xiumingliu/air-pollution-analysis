clear all; 
close all;

load('C:\Users\Administrator\Documents\GitHub\air-pollution-analysis\data\data.mat')

isworkingdays = ~isweekend(time) & ~ismember(time, holidays_2006_2015);
isholidays = isweekend(time) | ismember(time, holidays_2006_2015);

%%
figure;
scatter(temperature(time.Hour == 12 & isworkingdays), log(no2(time.Hour == 12 & isworkingdays)));
xlabel('Temperature');
ylabel('NO2');

figure;
scatter(temperature(time.Hour == 12 & isworkingdays), log(nox(time.Hour == 12 & isworkingdays)));
xlabel('Temperature');
ylabel('NOx');

figure;
scatter(temperature(time.Hour == 12 & isworkingdays), log(o3(time.Hour == 12 & isworkingdays)));
xlabel('Temperature');
ylabel('O3');

figure;
scatter(temperature(time.Hour == 12 & isworkingdays), log(pm10(time.Hour == 12 & isworkingdays)));
xlabel('Temperature');
ylabel('PM10');

figure;
scatter(temperature(time.Hour == 12 & isworkingdays), log(pm2_5(time.Hour == 12 & isworkingdays)));
xlabel('Temperature');
ylabel('PM2.5');

%%
figure;
scatter(windspeed(time.Hour == 12 & isworkingdays), log(no2(time.Hour == 12 & isworkingdays)));
xlabel('Wind speed');
ylabel('NO2');

figure;
scatter(windspeed(time.Hour == 12 & isworkingdays), log(nox(time.Hour == 12 & isworkingdays)));
xlabel('Wind speed');
ylabel('NOx');

figure;
scatter(windspeed(time.Hour == 12 & isworkingdays), log(o3(time.Hour == 12 & isworkingdays)));
xlabel('Wind speed');
ylabel('O3');

figure;
scatter(windspeed(time.Hour == 12 & isworkingdays), log(pm10(time.Hour == 12 & isworkingdays)));
xlabel('Wind speed');
ylabel('PM10');

figure;
scatter(windspeed(time.Hour == 12 & isworkingdays), log(pm2_5(time.Hour == 12 & isworkingdays)));
xlabel('Wind speed');
ylabel('PM2.5');

%%
figure;
scatter(rh(time.Hour == 12 & isworkingdays), log(no2(time.Hour == 12 & isworkingdays)));
xlabel('Relative humidity');
ylabel('NO2');

figure;
scatter(rh(time.Hour == 12 & isworkingdays), log(nox(time.Hour == 12 & isworkingdays)));
xlabel('Relative humidity');
ylabel('NOx');

figure;
scatter(rh(time.Hour == 12 & isworkingdays), log(o3(time.Hour == 12 & isworkingdays)));
xlabel('Relative humidity');
ylabel('O3');

figure;
scatter(rh(time.Hour == 12 & isworkingdays), log(pm10(time.Hour == 12 & isworkingdays)));
xlabel('Relative humidity');
ylabel('PM10');

figure;
scatter(rh(time.Hour == 12 & isworkingdays), log(pm2_5(time.Hour == 12 & isworkingdays)));
xlabel('Relative humidity');
ylabel('PM2.5');