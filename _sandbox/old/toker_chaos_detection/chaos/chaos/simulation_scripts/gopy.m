function [x,theta,y]=gopy(N,level, sigma)

% Simulate the GOPY system described in Grebogi et al (1984), "Strange
% attractors that are not chaotic"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signals,
% relative to the standard deviation of those signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% each component of the GOPY system 

% sigma - parameter of the GOPY system. For strange non-chaotic dynamics,
% set sigma = 1.5

% golden ratio
w = (sqrt(5)-1)/2;


% Random initial conditions
x(1,1)=randn;%438;%randn;
theta(1,1)=0;


% Simulate
for i=2:N
    x(i,1)=2*sigma*tanh(x(i-1,1))*cos(2*pi*theta(i-1,1));

    theta(i,1)=mod((theta(i-1,1)+w),1);
end

% Take a linear combination of the two variables
y = x./(4*sigma)+theta./10; 
%y=x./(4*sigma)+theta;

% Add normal white noise
x=x+randn(N,1)*level*std(x);
theta=theta+randn(N,1)*level*std(theta);
y=y+randn(N,1)*level*std(y);