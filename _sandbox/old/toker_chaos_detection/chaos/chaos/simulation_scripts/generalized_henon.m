function y = generalized_henon(N, level, a, b)

% Simulate the generalized Henon map described in Richter (2002),
% "The generalized Henon maps: examples for higher-dimensional chaos"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signal,
% relative to the standard deviation of the signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% y
%
% a, b - parameters of the map. For hyperchaotic dynamics, as in the paper,
% set a=1.76, b=0.1

% Random initial conditions
y(1:2) = abs(randn(1,2));

% Simulate
for i = 3:N-1
    y(i+1) = a-y(i-1)^2-b*y(i-2);
end

% Add white noise
y=y+randn(1,N)*level*std(y);