clear all; 
close all;

load('C:\Users\Administrator\Documents\GitHub\air-pollution-analysis\data\data.mat')

isworkingdays = ~isweekend(time) & ~ismember(time, holidays_2006_2015);
isholidays = isweekend(time) | ismember(time, holidays_2006_2015);

%% 
figure;
histogram(no2(time.Hour == 12 & isworkingdays),'Normalization','pdf');
xlabel('NO$_2$ ($\mu$g/m$^3$), working days, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(nox(time.Hour == 12 & isworkingdays),'Normalization','pdf');
xlabel('NO$_x$ ($\mu$g/m$^3$), working days, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(o3(time.Hour == 12 & isworkingdays),'Normalization','pdf');
xlabel('O$_3$ ($\mu$g/m$^3$), working days, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(pm10(time.Hour == 12 & isworkingdays),'Normalization','pdf');
xlabel('PM$_10$ ($\mu$g/m$^3$), working days, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(pm2_5(time.Hour == 12 & isworkingdays),'Normalization','pdf');
xlabel('PM$_{2.5}$ ($\mu$g/m$^3$), working days, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

%%
figure;
histogram(temperature(time.Month == 7 & time.Hour == 12),'Normalization','pdf');
xlabel('Temperature ($^\circ$C), July, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(rh(time.Month == 7 & time.Hour == 12),'Normalization','pdf');
xlabel('Relative humidity ($\%$), July, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(pressure(time.Month == 7 & time.Hour == 12),'Normalization','pdf');
xlabel('Pressure (hPa), July, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(radiation(time.Month == 7 & time.Hour == 12),'Normalization','pdf');
xlabel('Radiation (W/m$^2$), July, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
histogram(windspeed(time.Month == 7 & time.Hour == 12),'Normalization','pdf');
xlabel('Wind speed (m/s), July, 12:00 noon', 'Interpreter', 'latex');
ylabel('Probability density', 'Interpreter', 'latex');

figure;
polarhistogram(winddirection(time.Month == 7 & time.Hour == 12),'Normalization','pdf');
title('Wind direction, July, 12:00 noon', 'Interpreter', 'latex');