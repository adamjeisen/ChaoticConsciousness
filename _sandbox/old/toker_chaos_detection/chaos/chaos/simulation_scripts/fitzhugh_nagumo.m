
function [v,w] = fitzhugh_nagumo(N, level, a, b, c, I, ISI)

% Simulate the Rossler system described in Rossler (1976), "An equation for
% continuous chaos" using a fourth-order Runge Kutta method

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signals,
% relative to the standard deviation of those signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% each component of the Rossler system
%
% a, b, c - parameters of the Rossler system. For the chaotic
% simulation of Table 4, a=.2, b=.2, and c=5.7

% w - the frequency of the Rossler system oscillations (in the paper, w=1)

% random initial conditions
v(1)=rand;
w(1)=rand;

% time points, with an integration step of 0.01
h=.01;   %step size
t=0: h:  ((N-1)*h+  150000*h);
I_ext = zeros(1,length(t));
I_ext(ISI:ISI:length(I_ext))=I;

% ordinary differential equations
f=@(t,v,w) -v*(v-1)*(v-a)-w+I_ext(t);
g=@(t,v,w) c*(v-b*w);


% Simulate
for i=1:(length(t)-1)
    k1=f(t(i),v(i),w(i));
    l1=g(t(i),v(i),w(i));
  
    k2=f(t(i)+h/2,(v(i)+0.5*k1*h),(w(i)+(0.5*l1*h)));
    l2=g(t(i)+h/2,(v(i)+0.5*k1*h),(w(i)+(0.5*l1*h)));

    k3=f(t(i)+h/2,(v(i)+0.5*k2*h),(w(i)+(0.5*l2*h)));
    l3=g(t(i)+h/2,(v(i)+0.5*k2*h),(w(i)+(0.5*l2*h)));

    k4=f(t(i)+h,(v(i)+k3*h),(w(i)+l3*h));
    l4=g(t(i)+h,(v(i)+k3*h),(w(i)+l3*h));

    v(i+1) = v(i) + h*(k1 +2*k2  +2*k3   +k4)/6;
    w(i+1) = w(i) + h*(l1  +2*l2   +2*l3    +l4)/6;
end

% discard initial settling period
v=v(150001:end);
w=w(150001:end);

% take linear combination of x and y components of system
b=v+w;

% add white noise
l=length(v);
v=v+randn(1,l)*level*std(v);
w=w+randn(1,l)*level*std(w);
b=b+randn(1,l)*level*std(b);
