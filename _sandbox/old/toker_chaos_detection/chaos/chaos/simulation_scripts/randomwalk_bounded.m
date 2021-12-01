function y = randomwalk_bounded(N,level, k,a0,a1,a2,sig)

% Simulate a bounded walk, following Nicolau 2002, "Stationary Processes
% That Look like Random Walks: The Bounded Random Walk Process in
% Discrete and Continuous Time." The resulting signal will be globally
% stationary but locally non-stationary
% bounded random walk
% k = 100
% a0=-15
% a1 = 3
% a2 = 3
% sig = 0.4

y=zeros(1,N);
y(1)=100;
for i = 2:N
    y(i) = y(i-1)+exp(a0)*(exp(-a1*(y(i-1)-k))-exp(a2*(y(i-1)-k)))+sig*randn;
end
y=y';
% Add white measurement noise
y=y+randn(N,1)*level*std(y);