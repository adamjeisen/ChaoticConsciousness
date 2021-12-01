function [x,y,z,a] = lorenz(N, level, sigma, rho, beta,dynamic_noise)

% Simulate the Lorenz system described in Lorenz (1963), "Deterministic
% nonperiodic flow" using a fourth-order Runge Kutta method. The
% data are downsampled by a factor of 5,000 so that there are significant 
% numbers of cyclces in the simulated data (otherwise even thousands of
% time-points will only yield a few cycles)

% Inputs
% N - number of time-points to simulate
%
% level - the amplitude of white noise to add to the final signals,
% relative to the standard deviation of those signals (e.g. level=0.2 will
% add white noise, the amplitude of which is 20% the standard deviation of
% each component of the Lorenz system 
%
% sigma, rho, beta - parameters of the Lorenz system. For the chaotic
% simulation of Table 2, sigma=10, rho=30, beta=(8/3)

% random initial conditions
x(1)=rand;
y(1)=rand;
z(1)=rand;

% time points
n=N;
h=.01;   %step size
t=0: h:  ((n-1)*h+  150000*h);

% ordinary differential equations
f=@(t,x,y,z) sigma*(y-x); 
g=@(t,x,y,z) x*rho-x.*z-y;
p=@(t,x,y,z) x.*y-beta*z;

% simulate using the fourth-order Runge-Kutta method
for i=1:(length(t)-1) %loop
    k1=f(t(i),x(i),y(i),z(i));
    l1=g(t(i),x(i),y(i),z(i));
    m1=p(t(i),x(i),y(i),z(i));
      k2=f(t(i)+h/2,(x(i)+0.5*k1*h),(y(i)+(0.5*l1*h)),(z(i)+(0.5*m1*h)));     
      l2=g(t(i)+h/2,(x(i)+0.5*k1*h),(y(i)+(0.5*l1*h)),(z(i)+(0.5*m1*h)));
      m2=p(t(i)+h/2,(x(i)+0.5*k1*h),(y(i)+(0.5*l1*h)),(z(i)+(0.5*m1*h)));
      k3=f(t(i)+h/2,(x(i)+0.5*k2*h),(y(i)+(0.5*l2*h)),(z(i)+(0.5*m2*h)));
      l3=g(t(i)+h/2,(x(i)+0.5*k2*h),(y(i)+(0.5*l2*h)),(z(i)+(0.5*m2*h)));
      m3=p(t(i)+h/2,(x(i)+0.5*k2*h),(y(i)+(0.5*l2*h)),(z(i)+(0.5*m2*h)));
      k4=f(t(i)+h,(x(i)+k3*h),(y(i)+l3*h),(z(i)+m3*h));
      l4=g(t(i)+h,(x(i)+k3*h),(y(i)+l3*h),(z(i)+m3*h));
      m4=p(t(i)+h,(x(i)+k3*h),(y(i)+l3*h),(z(i)+m3*h));
      x(i+1) = x(i) + h*(k1 +2*k2  +2*k3   +k4)/6+dynamic_noise*randn; %final equations
      y(i+1) = y(i) + h*(l1  +2*l2   +2*l3    +l4)/6;
      z(i+1) = z(i) + h*(m1+2*m2 +2*m3  +m4)/6;
end

% get rid of initial settling period
x=x(150001:end);
z=z(150001:end);
y=y(150001:end);


% % downsample
% x=downsample(x,5);
% y=downsample(y,5);
% z=downsample(z,5);

% take linear combination of x and y components (this is the signal
% analyzed in our paper)
a=x+y;

% add white noise
l=length(x);
x=x+randn(1,l)*level*std(x);
y=y+randn(1,l)*level*std(y);
z=z+randn(1,l)*level*std(z);
a=a+randn(1,l)*level*std(a);