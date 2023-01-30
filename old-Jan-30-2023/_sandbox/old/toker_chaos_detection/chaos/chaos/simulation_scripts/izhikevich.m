function y = izhikevich(N,level,a,b,c,d,I)

% Simulate the Izhikevich spiking neuron model described in Izhikevich
% (2003), "Simple model of spiking neurons" and Izhikevich (2004), "Which
% model to use for cortical spiking neurons?"

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signal,
% relative to the standard deviation of the signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% the spiking neuron model)
%
% a, b, c, d, I - parameters of the model. For chaotic spiking, set a=.2, 
% b=2, c=-56, d=-16, I=-99 

% random initial condition
V=randn*80;  
u=b*V;
y=[];  uu=[];

% integration step size
tau = 0.25; 

% timepoints
tspan = 0:tau:(N+999)*tau;
for t=tspan
    V = V + tau*(0.04*V^2+5*V+140-u+I);
    u = u + tau*a*(b*V-u);
    if V > 30
        y(end+1)=30;
        V = c;
        u = u + d;
    else
        y(end+1)=V;
    end
    uu(end+1)=u;
end
y=y(21:end); % discard initial settling period

% add white noise
y=y+randn(size(y))*level*std(y);
