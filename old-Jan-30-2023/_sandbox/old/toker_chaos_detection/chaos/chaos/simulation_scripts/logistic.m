function y=logistic(N,level,r,y0)

% Simulate the logistic map described in May (1976), "Simple mathematical
% models with very complicated dynamics"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signal,
% relative to the standard deviation of the signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% the logistic map 
%
% r - parameter of the map. For the periodic dynamics reported in Table 2,
% set r = 3.5, and for the chaotic dynamics reported in Table 2, set r=4

% random initial condition
if nargin<4
y0=rand;
end
y(1,1)=r*y0*(1-y0);

%N=N+1000;

% Simulate
for i=2:N
    y(i,1)=r*y(i-1,1)*(1-y(i-1,1));
end


%y=y(1001:end); % discard initial settling period
