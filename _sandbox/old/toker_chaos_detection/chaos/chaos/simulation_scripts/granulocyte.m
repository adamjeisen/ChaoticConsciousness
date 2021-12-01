function y=granulocyte(N,level,a,b,c,s)

% Simulate the circulating granulocyte levels model described in Mackey and
% Glass (1977), "Oscillation and chaos in physiological control systems"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signal,
% relative to the standard deviation of the signal (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% the simulated signal
%
% a, b, c, s - parameters of the model. For the periodic dynamics simulated
% in Table 1, set a=.2, b=.1, c=10, and s=10. For the chaotic dynamics
% simulated in Table 2 (modeling observed white blood cell levels in
% granulocytic leukemia), set a=.2, b=.1, c=10, and s=30 (s controls the
% delay time between granulocyte production in bone marrow and its release
% into the bloodstream, a process which is delayed in granulocytic
% leukemia)

% Farmer, CHAOTIC ATTRACTORS OF AN INFINITE-DIMENSIONAL DYNAMICAL SYSTEM,
% 1982


N=N+100;

% random initial conditions
y0=0.1*randn(s,1).*ones(s,1);
y(1,1)=y0(s)+a*y0(1)/(1+y0(1)^c)-b*y0(s);
for i=2:length(y0)
    y(i,1)=y(i-1)+a*y0(i)/(1+y0(i)^c)-b*y(i-1);
end

% Simulate
for i=s+1:N
    y(i,1)=y(i-1)+a*y(i-s)/(1+y(i-s)^c)-b*y(i-1);
end

% Add normal white noise
y=y+randn(N,1)*level*std(y);

y=y(101:end); % discard initial settling period