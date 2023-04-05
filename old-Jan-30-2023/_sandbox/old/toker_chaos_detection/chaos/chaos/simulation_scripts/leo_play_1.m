addpath 'C:\Users\Leo\Projects\ChaoticConsciousness\ChaoticConsciousness\_sandbox\toker_chaos_detection\chaos\chaos';
addpath(genpath('C:\Users\Leo\Projects\utils\Etalo-main'))
addpath(genpath('C:\Users\Leo\Projects\utils\spikeAnalysis'))
etl = Etalo();

N = 1000;
level = .01;
mu = 2.4;
p = 0.01;
theta = 0.1;



FILENAME = '\\millerdata.mit.edu\common\datasets\anesthesia\mat\propofolPuffTone\Mary-Anesthesia-20160809-01.mat';
%load(FILENAME)
START_TIME = round(size(lfp,1)*0.8);
WINDOW = 40000;
mean_lfp = mean(lfp(START_TIME:START_TIME+WINDOW,:).^2,2);
%rand_lfp_unit = lfp(START_TIME:START_TIME+WINDOW,10);



%y = random_ARMA(N, level, p,theta);

%output = chaos_mainfun(mean_lfp);