function y = noise_sine_map(N,level, mu)

% Simulate the noise-driven sine map described in Freitas et al (2009), 
% "Failure in distinguishing colored noise from chaos using the 'noise
% titration' technique"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signal,
% relative to the standard deviation of the signal (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% the noise-driven sine map 
%
% mu - parameter of the map. For the dynamics simulated in the paper,
% mu=2.4

% random initial condition
y(1)=rand;

% Simulate
for i=2:N
    
    % random Bernoulli process
    q=rand;
    if q<=.01
        Y=1;
    else
        Y=0;
    end
    
    y(i)=mu*sin(y(i-1))+Y*(4*rand-2);
end

% add white noise
y=y'+randn(N,1)*level*std(y);