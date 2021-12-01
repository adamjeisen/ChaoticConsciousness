function y = freitas(n,level)

% Simulate the nonlinear stochastic map described in Freitas et al (2009), 
% "Failure in distinguishing colored noise from chaos using the 'noise
% titration' technique"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signal,
% relative to the standard deviation of the signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% y

% random initial conditions
y(1:2)=rand(1,2);
v=rand(1,n);

% Simulate
for i=3:n
    y(i)=3*v(i-1)+4*v(i-2)*(1-v(i-1));
end

% add white noise
y=y'+randn(n,1)*level*std(y);