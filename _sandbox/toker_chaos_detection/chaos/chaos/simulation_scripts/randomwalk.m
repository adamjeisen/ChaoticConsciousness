function y = randomwalk(N,level)

% Simulate a random walk (a stochastic, non-stationary signal)

% Inputs
% N - number of time-points to simulate

y=zeros(1,N);
y(1)=randn;
for i = 2:N
    y(i) = y(i-1)+randn;
end
y=y';
% Add white measurement noise
y=y+randn(N,1)*level*std(y);