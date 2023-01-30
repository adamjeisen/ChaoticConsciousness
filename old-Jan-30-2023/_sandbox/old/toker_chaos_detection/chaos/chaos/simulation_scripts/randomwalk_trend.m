function y = randomwalk_trend(N,level,b)

% Simulate a random walk (a stochastic, non-stationary signal)

% Inputs
% N - number of time-points to simulate
% b - slope of the linear trend
% multiplied randn by 0.01 to generate data
y=zeros(1,N);
y(1)=randn;
for i = 2:N
    y(i) = y(i-1)+b+randn;
end
y=y';
y=y+randn(N,1)*level*std(detrend(y));