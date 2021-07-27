function [x,y,a]=henon(N,level,a,b)

% Simulate the Henon map described in Henon (1976), "A two-dimensional
% mapping with a strange attractor"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signals,
% relative to the standard deviation of those signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% each component of the Henon map 
%
% a, b - parameters of the map. For periodic dynamics, as in the paper,
% set a=1.25, b=0.3

%
% Henon M (1976):A two-dimensional mapping with a strange attractor. 
% Communications in Mathematical Physics 50: 69-77

% Random initial conditions
x0=0.1*randn;
y0=0.1*randn;
x(1,1)=1-a*x0^2+b*y0;
y(1,1)=b*x0;

% Simulate
for i=2:N
    x(i,1)=1-a*x(i-1,1)^2+y(i-1,1);
    y(i,1)=b*x(i-1,1);
end

a=x+y;

% Add white noise
x=x+randn(N,1)*level*std(x);
y=y+randn(N,1)*level*std(y);
a=a+randn(N,1)*level*std(a);