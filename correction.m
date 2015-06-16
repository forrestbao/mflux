% define the coefficient matrix and constraint
clear
clc
%% get values from user inputs
ratio_glucose = 1;
ratio_galactose = 0;
ratio_fructose = 0;
ratio_glycerol = 0;
ratio_gluconate = 0;
ratio_pyruvate = 0;
ratio_citrate = 0;
ratio_xylose = 0;
ratio_malate = 0;
ratio_succinate = 0;
ratio_glutamate = 0;
%% flux reported in 13CMFA paper, not useful for website,ignore this
realflux = [100.0,1.1,7.6,7.6,96.7,84.5,97.2,106.7,0.0,97.7,18.0,9.1,8.9,5.6,3.5,5.6,86.4,86.4,86.0,73.1,70.0,70.0,70.0,0.0,79.7,0.0,0.0,23.7,0]';

%% flux vector predicted by ML
f = [100.0000 ,-2.7159,15.2254,17.7016,110.9973,91.8578,137.7961,91.1558, -0.7373, 94.1518, 24.1126,21.2310,2.8816,11.0324,10.1986,11.0324,79.4203, 79.4203,67.9442,67.8806,79.3567,79.3567, 64.0876, 11.4761, 70.0392,  -1.2424, 0.0059,  23.2159, 26.7451]';

%% average value of all fluxes from literature survey
fa = [88.6	40.1	56.3	56.2	137.3	130.6	120.3	98.9	16.7	46.6	26.2	13.7	11.6	8.2	6.2	8.2	64.6	64.6	60.3	53.4	54.0	54.5	46.8	4.0	20.5	5.9	3.5	25.8	14.1]';
%% upper bound and lower bound from the range of literature survey
ub = [100,99.5,99.3,99.3,216.6,196.2,232,213.1,135,151.4,113.7,94.1,41.2,47.5,71,47.5,189,189,189,194,194,194,181.5,55,148,193.2,151,149.8,104.2043714]';
lb = [0,-99.9,-51.5,-51.5,-13.5,-23.3,-36,-7.9,-144,0,0,-33,-94.4,-2,-6.6,-2,0,-0.1,-0.1,0,-105,-106,-144.3,0,0,0,-100,-67.60986805,-13.5]';

%% specific setting on boundary, based on the user's knowledge on bacteria
lb(29) = 0;ub(29)= 0;
ub(24) = 0;
 ub(9) = 0;
 
%% in case some flux missing, check the number of flux available first
dim =size(realflux,1);
f = f(1:dim,1);

%% in case some of ML results are out of flux range, adjust the initial value to be
%% in the range and used for quadratic programming
init0 = f;
for i = 1:dim
    if f(i) <= lb(i) 
        init0 (i) = lb(i);
    end
    if f(i) >= ub(i)
          init0 (i) = ub(i);
    end 
end

%% Objective function: to minimize sum Xn^2 - 2*Vn*Xn (29 fluxes in total)
% coefficient of objective function at second order
H = diag (ones(1,29));

%% linear constraints
% inequality
Aineq = zeros(12,29);
Aineq(1,1) = 1; Aineq(1,2) = -1; Aineq(1,10) = -1;    
Aineq(2,2) = 1;Aineq(2,3) = -1;  Aineq(2,16) = 2; 
Aineq(3,3) = 1; Aineq(3,4) = 1; Aineq(3,5) = -1;Aineq(3,14) = 1; Aineq(3,25) = 1;
Aineq(4,5) = 1; Aineq(4,6) = -1; 
Aineq(5,6) = 1; Aineq(5,7) = -1; Aineq(5,28) = -1; 
Aineq(6,7) = 1; Aineq(6,8) = -1; Aineq(6,25) = 1;Aineq(6,27) = -1; Aineq(6,29) = 1;  
Aineq(7,8) = 1; Aineq(7,9) = -1; Aineq(7,17) = -1;Aineq(7,24) = -1; Aineq(7,26) = -1; 
Aineq(8,13) = 1; Aineq(8,14) = -1;  
Aineq(9,16) = 1; Aineq(9,15) = -1; 
Aineq(10,19) = 1; Aineq(10,20) = -1;
Aineq(11,23) = 1; Aineq(11,17) = -1;Aineq(11,28) = 1;
Aineq(12,21) = -1; Aineq(12,22) = 1;

bineq = zeros(12,1);  
bineq(2,1)= 100 * ratio_fructose; 
bineq(6,1)= 100 * ratio_pyruvate; bineq(10,1) = 100 * ratio_glutamate;  

Aineq = -Aineq; % A*X <= B  

% equlity
Aeq = zeros(10,29);
Aeq(1,1) = 1; 
Aeq(2,3) = 1; Aeq(2,4) = -1; 
Aeq(3,11) = 1; Aeq(3,12) = -1; Aeq(3,13) = -1; 
Aeq(4,14) = 1; Aeq(4,16) = -1;  
Aeq(5,10) = 1; Aeq(5,11) = -1; Aeq(5,25) = -1; 
Aeq(6,18) = 1; Aeq(6,17) = -1;  
Aeq(7,15) = 1; Aeq(7,12) = -1; Aeq(7,14) = 1; 
Aeq(8,24) = 1; Aeq(8,18) = -1; Aeq(8,19) = 1; 
Aeq(9,22) = -1; Aeq(9,23) = 1; Aeq(9,24) = -1;Aeq(9,29) = 1; 
 Aeq(10,20) = 1; Aeq(10,24) = 1;Aeq(10,21) = -1;
beq = zeros(10,1);
beq(1,1) = 100 * (ratio_glucose + ratio_galactose);
beq(2,1) = -100 * ratio_glycerol;
beq(5,1) = -100 * ratio_gluconate;
beq(6,1) = 100 * ratio_citrate;
beq(7,1) = 100 * ratio_xylose;
beq(9,1) = 100 * ratio_malate;
beq(10,1)= -100 * ratio_succinate;

% optimset('Algorithm','interior-point-convex','MaxIter',100,'Display','off');
% adjust flux to calculate
% fa = fa(1:28);
H = H(1:dim,1:dim);
Aineq = Aineq(:,1:dim);
Aeq = Aeq(:,1:dim);
lb = lb(1:dim);
ub = ub(1:dim);
options = optimset('LargeScale','off','Algorithm','interior-point','MaxIter',1000,'Display','off');
[x,fval,exitflag,output,lambda] = quadprog(H,-f,Aineq,bineq,Aeq,beq,lb,ub,init0,options);
%check the value exitflag, if 1 -- get optimized solution, otherwise, the
%optimiztion may get failed

%% for validation early, not useful for website
result_ML = sum((realflux -f).^2);
result_cor = sum((realflux -x).^2);

[x1,fval,exitflag2,output1,lambda1] = quadprog(H,-fa,Aineq,bineq,Aeq,beq,lb,ub,fa,options);
result_avgcor  = sum((realflux -x1).^2);
x_test = (x+x1)/2;
result_avg = sum((realflux -fa').^2);
result3 = sum((realflux -x_test).^2);

