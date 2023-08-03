% File: Evolutionary_Game.m
% 
% Description: 
% 1. This program is used to implement the evolutionary game analysis
% process in the paper, mainly including: defining model parameters and
% income matrix, constructing and solving dynamic equations for
% replication, simulation analysis and visualization.
% 2. This program does not require additional input, and the parameters are
% consistent with the paper, so it can be run directly. 
% 3. Please save the differential function to the same directory as the
% main program to ensure that the program runs correctly. 
% 
% Copyright (c) 2023 Jiaming Wang. All rights reserved.
% Citation：
% Jiaming Wang, Ling Jia, Pan He, Peng Wang, Lei Huang,
% Engaging stakeholders in collaborative control of air pollution: A tripartite evolutionary game of enterprises, public and government,
% Journal of Cleaner Production, Volume 418, 2023, 138074, ISSN 0959-6526
% https://doi.org/10.1016/j.jclepro.2023.138074

%% Main Program
clear
currentPath = pwd;
disp(currentPath);
userpath(currentPath);

disp('Defining parameters 参数定义中')

% The meaning of each parameter is consistent with Table 1.
syms    x       PA      Pe      CA          eA        Rc       alpha      beta       gamma   ...
  ...%企业治理 正常利润 外部性 完全治理成本 努力程度  碳减排总收益 补贴系数    惩罚系数   创新能力
        y       CH              CB1        delta      CB0     epsilon      Rp        zeta       eta         theta       lambda...
  ...%公众参与 健康损失-负       参与成本  渠道难度系数 防护成本  补贴系数    环境收益   专业公众  促进企业创新  促进政府收益  心理损失系数
        z       CG1      CE     gov         G0        G1      kappa       ;
     %政府监管 行政成本 环境损害 外部性分成 声誉损失  声誉收益   分权系数    
disp('Completed 已完成')
% Note:In the paper, in order to facilitate the interpretation of the
% actual meaning, CB0 is replaced by PB0-CB0 to avoid the situation where
% -CB0 appears alone in the inequality.

% A for Enterprise 企业 
P = PA + (1-eA)*(1-gov)*Pe; %Total revenue of Enterprise企业总收益
% B for Public 公众
% C for Government 政府

%% Payoff matrix 支付矩阵

disp('Defining payoff matrix 支付矩阵构建中')
% Probability z of government regulation 
% Probability x of corporate pollution control 
% Probability y of public participation

% (1)When the government regulates(z), the benefit matrix of the three stakeholders is as follows 政府监管z，三方的收益矩阵
% (1-1)企业治污x 公众参与y
    A_PayCCC = (1+eta)*PA + (1-eA)*(1-gov)*Pe - eA*(1-gamma)*CA + eA*Rc + alpha*P ;             %企业收益：正常利润+外部性利润-治污投入+碳减排收益+政府补贴
    B_PayCCC = (1-eA)*CH + zeta*Rp - (1-epsilon)*delta*CB1 ;                                    %公众收益：健康损害 + 环境效益 + 政府补贴 - 参与成本
    C_PayCCC = (1-kappa)*(eA-1)*CE + kappa*(1-eA)*gov*Pe + (1+theta)*G1- CG1 - alpha*P - epsilon*delta*CB1 ; %政府收益：环境损害降低+外部性收益+增益声誉收益-行政成本-企业补贴-公众补贴
% (1-2)企业不治污(1-x) 公众参与y
    A_PayDCC = PA + (1-gov)*Pe - beta*P ;                                                     %企业收益：正常利润+外部性利润-政府处罚
    B_PayDCC = (1+lambda)*CH + zeta*Rp - (1-epsilon)*delta*CB1 ;                              %公众收益：健康损害+心理损害+环境效益 + 政府补贴 - 参与成本
    C_PayDCC = -(1-kappa)*CE + kappa*gov*Pe + (1+theta)*G1 - CG1 + beta*P- epsilon*delta*CB1; %政府收益：环境损害+外部性收益+增益声誉收益-行政成本+企业处罚-公众补贴
% (1-3)企业治污x 公众不参与(1-y)
    A_PayCDC = PA + (1-eA)*(1-gov)*Pe - eA*(1-gamma)*CA + eA*Rc + alpha*P ;     %企业收益：正常利润+外部性利润-治污投入+碳减排收益+政府补贴
    B_PayCDC = (1-eA)*CH - CB0 ;                                                %公众收益：健康损害-个人防护成本
    C_PayCDC = (1-kappa)*(eA-1)*CE + kappa*(1-eA)*gov*Pe + G1 - CG1 - alpha*P; %政府收益：环境损害降低+外部性收益+声誉收益-行政成本-企业补贴
% (1-4)企业不治污(1-x) 公众不参与(1-y)
    A_PayDDC = PA + (1-gov)*Pe - beta*P ;                               %企业收益：正常利润+外部性利润-政府处罚
    B_PayDDC = CH - CB0 ;                                               %公众收益：健康损害-个人防护成本
    C_PayDDC = -(1-kappa)*CE + kappa*gov*Pe + G1 - CG1 + beta*P;        %政府收益：环境损害+外部性收益+声誉收益-行政成本+企业处罚

% (2)When the government not to regulates(1-z), the benefit matrix of the three stakeholders is as follows 政府不监管时(1-z)，三方收益矩阵
% (2-1)企业治污x 公众参与y
    A_PayCCD = (1+eta)*PA + (1-eA)*(1-gov)*Pe - eA*(1-gamma)*CA + eA*Rc ;   %企业收益：正常利润+外部性利润-治污投入+碳减排收益
    B_PayCCD = (1-eA)*CH + zeta*Rp - delta*CB1 ;                            %公众收益：健康损害 + 环境效益 - 参与成本
    C_PayCCD = (1-kappa)*(eA-1)*CE + kappa*(1-eA)*gov*Pe - (1+theta)*G0;    %政府收益：环境损害降低+外部性收益-政府声誉加剧损失
% (2-2)企业不治污(1-x) 公众参与y
    A_PayDCD = PA + (1-gov)*Pe ;                                 %企业收益：正常利润+外部性利润
    B_PayDCD = (1+lambda)*CH + zeta*Rp - delta*CB1 ;             %公众收益：健康损害+心理损害+环境效益 - 参与成本
    C_PayDCD = -(1-kappa)*CE + kappa*gov*Pe - (1+theta)*G0;      %政府收益：环境损害+外部性收益-政府声誉加剧损失
% (2-3)企业治污x 公众不参与(1-y)
    A_PayCDD = PA + (1-eA)*(1-gov)*Pe - eA*(1-gamma)*CA + eA*Rc ; %企业收益：正常利润+外部性利润-治污投入+碳减排收益
    B_PayCDD = (1-eA)*CH - CB0 ;                                  %公众收益：健康损害-个人防护成本
    C_PayCDD = (1-kappa)*(eA-1)*CE + kappa*(1-eA)*gov*Pe -G0;     %政府收益：环境损害降低+外部性收益-政府声誉损失
% (2-4)企业不治污(1-x) 公众不参与(1-y)
    A_PayDDD = PA + (1-gov)*Pe ;                  %企业收益：正常利润+外部性利润
    B_PayDDD = CH - CB0 ;                         %公众收益：健康损害-个人防护成本
    C_PayDDD = -(1-kappa)*CE + kappa*gov*Pe -G0;  %政府收益：环境损害+外部性收益-政府声誉损失

A_Payoff = [A_PayCCC;A_PayDCC;A_PayCDC;A_PayDDC;A_PayCCD;A_PayDCD;A_PayCDD;A_PayDDD];
B_Payoff = [B_PayCCC;B_PayDCC;B_PayCDC;B_PayDDC;B_PayCCD;B_PayDCD;B_PayCDD;B_PayDDD];
C_Payoff = [C_PayCCC;C_PayDCC;C_PayCDC;C_PayDDC;C_PayCCD;C_PayDCD;C_PayCDD;C_PayDDD];

writematrix(string([A_Payoff, B_Payoff, C_Payoff]),'Outputs关键过程输出.xlsx','WriteMode','overwritesheet','Sheet','Payoff Matrix支付矩阵')
disp('Completed 已完成')

%% Replicator equation 复制方程
% Main steps: 
% (1) calculate the average expectation according to the payoff matrix; 
% (2) find the replication equation and simplify it; 
% (3) get the replication dynamic equation set together and solve the Equant; 
% (4) construct the Jacobian matrix for stability analysis.
% 主要步骤：
% （1）根据收益矩阵，计算平均期望      （2）求复制方程，并化简 
% （3）联立得复制动态方程组，求解均衡点 （4）构建雅克比矩阵，稳定性分析

disp('Calculating replication equation 复制动态方程构建中')
% 【A-Enterprise企业】
    EA1 = y*z*A_PayCCC + (1-y)*z*A_PayCDC + y*(1-z)*A_PayCCD + (1-y)*(1-z)*A_PayCDD; % 企业治污收益.公众参与/不参与 政府监管/不监管，四种组合
    EA2 = y*z*A_PayDCC + (1-y)*z*A_PayDDC + y*(1-z)*A_PayDCD + (1-y)*(1-z)*A_PayDDD; % 企业不治污收益.四种组合
    EA = x*EA1 + (1-x)*EA2 ;
    FX = x*(1-x)*(collect(EA1-EA2,x)); 
    FX1 = simplify(FX);
    % The following steps are to replace it with the most intuitive expression 此后步骤为了换成最直观的表达式
    Tep_X = factor(FX1);
    FX2   = Tep_X(1)*Tep_X(2)*(collect(Tep_X(3),[z y PA Pe eA alpha beta])) %需要自己设置collect所需合并的同类项
   
% 【B-Public公众】
    EB1 = x*z*B_PayCCC + (1-x)*z*B_PayDCC + x*(1-z)*B_PayCCD + (1-x)*(1-z)*B_PayDCD; % 公众参与收益.企业治污/不治污 政府监管/不监管，四种组合
    EB2 = x*z*B_PayCDC + (1-x)*z*B_PayDDC + x*(1-z)*B_PayCDD + (1-x)*(1-z)*B_PayDDD; % 公众不参与收益.四种组合
    EB = y*EB1 + (1-y)*EB2 ;
    FY = y*(1-y)*(collect(EB1-EB2,y));  %dy/dt = y(EB1-EB) = y(1-y)(EB1-EB2)
    FY1 = simplify(FY);
    FY2 = FY1

% 【C-政府】
    EC1 = x*y*C_PayCCC + (1-x)*y*C_PayDCC + x*(1-y)*C_PayCDC + (1-x)*(1-y)*C_PayDDC; % 政府监管.企业治污/不治污 公众参与/不参与，四种组合
    EC2 = x*y*C_PayCCD + (1-x)*y*C_PayDCD + x*(1-y)*C_PayCDD + (1-x)*(1-y)*C_PayDDD; % 政府不监管.四种组合
    EC = z*EC1 + (1-z)*EC2 ;
    FZ = z*(1-z)*(EC1-EC2);  %dy/dt = z(EC1-EC) = z(1-z)(EC1-EC2)
    FZ1 = simplify(FZ);
    Tep_Z = factor(FZ1);
    FZ2 = Tep_Z(1)*Tep_Z(2)*(collect(Tep_Z(3),[Pe PA eA gov x y kappa])) %需要自己设置collect所需合并的同类项
    %M_C = [simplify(diff(FY,'x',1)); simplify(diff(FY,'y',1)); simplify(diff(FX,'z',1))]
    writematrix(string([FX2;FY2;FZ2]),'Outputs关键过程输出.xlsx','WriteMode','overwritesheet','Sheet','Replicator equation复制动态方程组')
   disp('Completed, the replication dynamic equations are: FX FY FZ')
   disp('已完成，三方复制动态方程分别为：FX FY FZ')
Replication_dynamics = [FX FY FZ];
Replication_dynamics2 = [FX2 FY2 FZ2];

% The system replication dynamic equations are established to solve the Equant.
% 建立系统复制动态方程组，求解均衡点
disp('The solution of the equation system is: 方程组的解为：')
sol = solve(Replication_dynamics,[x y z]);
sol2 = solve(Replication_dynamics2,[x y z]);
sol2.x = simplify(sol2.x);
sol2.y = simplify(sol2.y);
sol2.z = collect(simplify(sol2.z),[Pe eA gov PA]);
disp('The number of Equant of the system is: 则本系统均衡点个数为：')
Num = size(sol.x,1);
writematrix(string([sol2.x(1:Num) sol2.y(1:Num) sol2.z(1:Num)]),'Outputs关键过程输出.xlsx','WriteMode','overwritesheet','Sheet','Equant expression均衡点表达式')
readmatrix("Outputs关键过程输出.xlsx","Sheet",'Equant expression均衡点表达式')
disp(Num)

% Constructing Jacobian Matrix构建雅克比矩阵
    J = jacobian([FX FY FZ],[x y z]);
    J1 = simplify(J);
    J1(1,1) = simplify(J(1,1));
    J1(3,3) = simplify(J(3,3));
    disp('Jacobian Matrix: 本系统雅克比矩阵为：')
    disp(J1)
    writematrix(string(J1),'Outputs关键过程输出.xlsx','WriteMode','overwritesheet','Sheet','Jacobian Matrix雅克比矩阵')


% Stability analysis稳定性分析
% Substitute the Equant into the Jacobi matrix to solve the expression of   eigenvalue.
% 分别将均衡点代入雅克比矩阵，求解特征值的表达式
for i = 1:Num
    Sol_tep = [sol2.x(i) sol2.y(i) sol2.z(i)];
    x = Sol_tep(1);
    y = Sol_tep(2);
    z = Sol_tep(3);
    disp(i)
    J_tep = subs(J);
    E_tep = eig(J_tep);
    if i == 1
        E_all = simplify(E_tep);
    elseif i < 9
        E_all = [E_all,simplify(E_tep)];
    else
        break
        E_all = [E_all,E_tep];
    end
end
disp('In the three-player evolutionary game, there are 8 pure strategy equilibrium points, and the corresponding eigenvalue expressions are as follows:')
disp('三方博弈中，8个纯策略均衡点，对应的特征值表达式如下:')
disp(E_all(:,1:8)) 
% According to Lyapunov (1992), the equilibrium point is asymptotically
% stable only when all eigenvalues are negative.
% 对于每个均衡点，其所有特征值均 < 0 时，该点为ESS

writematrix(string((E_all)),'Outputs关键过程输出.xlsx','WriteMode','overwritesheet','Sheet','Eigenvalues expression特征值表达式')

%% Simulation analysis仿真分析

% This step is to pre-judge whether A(0,0,0) is ESS under the initial conditions of the four scenarios.
    Para_A =[x,      PA,      Pe,       CA,       eA,      Rc,         alpha,    beta,       gamma];
    Para_B =[y,      CH,      CB1,      delta,    CB0,     epsilon,     Rp,      zeta,        eta,        theta,      lambda];
    Para_C =[z,      CG1,     CE,       gov,      G0,      kappa,       G1,    ]; 
% ESS（0,0,0）
    [x,      PA,      Pe,       CA,       eA,      Rc,         alpha,   beta,       gamma,   ...
     y,      CH,      CB1,      delta,    CB0,     epsilon,    Rp,      zeta,        eta,        theta,      lambda ...
     z,      CG1,     CE,       gov,       G0,     kappa,       G1,     ]=...
deal( ...
      0.2,      40,      20,       35,        0.1,      10,         0,       0,       0.1, ...
...%企业治理 正常利润   外部性 完全治理成本 努力程度 碳减排总收益  补贴系数    惩罚系数    创新能力    
     0.2,     -30,      30,       0.8,       18,       0,         25,       0.1,       0.1,          0.1,        0.1, ...
...%公众参与 健康损失-负 参与成本  渠道难度系数 防护成本  补贴系数   环境收益   专业公众   企业创新增益  政府声誉倍增  心理损失系数
     0.2,      25,      30,       0.5,       10,       0.8,        5 ...     
...%政府监管 行政成本   环境损害 外部性分成 声誉损失  分权系数     声誉收益   
     );

% Judging the stability of the equilibrium point According to the above
% parameters, substitute each/the first 8 eigenvalues, judge the positive
% and negative respectively, and output the stability of each point {ESS; saddle point; unstable} 
% ESS: evolutionary stable state
% 判断均衡点的稳定性
% 根据以上参数，代入每个/前8个特征值，分别判断正负性，并输出每个点的稳定性{ESS；鞍点；不稳定}
Stable = strings(1,Num); % Asymptotically stable
for j = 1:8
    E_simu_tep = subs(E_all(:,j));
    if j == 1
        E_all_simu = E_simu_tep;
    else
        E_all_simu = [E_all_simu,E_simu_tep];
    end
    % Judging the stability of each equilibrium point
    % 判断每个均衡点的稳定性
    for k = sign(E_simu_tep)
        if sum(k) == -3
            Stable(j) = 'ESS';
        elseif sum(k) == 3
            Stable(j) = 'Unstable 不稳定';
        else
            Stable(j) = 'saddle point 鞍点';
        end
    end
end
disp('After substituting the values, the eigenvalues of the first 8 equilibrium points are as follows:')
disp('代入数值后，前8个均衡点的特征值如下：')
disp(E_all_simu(:,1:8))
disp('Under the current parameters, the stability of each point is as follows:')
disp('目前参数下，每个点的稳定性如下：')
disp(Stable)

%% Simulation analysis: Initial state for Scenario 1~4   A(0,0,0)
%  When A is an asymptotically stable point, the corresponding eigenvalues
%  must be less than 0. According to Table 3, the initial conditions are as follows:
% lambda_A1<0       eA*(Rc- Pe(1-gov)- CA*(1-gamma)) < 0  
% lambda_A2<0       (CB0 + CH*lambda) + (Rp*zeta- CB1*delta) < 0
% lambda_A3<0       G0 - CG1 + G1 + beta*(PA + Pe*(1-eA)*(1-gov)) - Pe*gov*kappa*eA < 0
%  Note: Rc < CA

% The initial simulation parameters we set first need to meet the basic
% parameter relationship of the game model in Table 1, and also meet the
% condition that point A (0,0,0) is an asymptotically stable point.

          %  x       PA      Pe         CA        eA        Rc         alpha    beta      gamma 
A_input = [0.2,      40,      20,       35,        0.1,      10,         0,       0,       0.1];
          %企业治理 正常利润  外部性  完全治理成本  努力程度 碳减排总收益  补贴系数  惩罚系数   创新能力    

          %  y       CH         CB1     delta      CB0     epsilon      Rp        zeta       eta         theta       lambda
B_input = [0.2,     -30,      30,       0.8,       18,       0,         25,       0.1,       0.1,          0.1,        0.1];
          %公众参与 健康损失-负 参与成本 渠道难度系数 防护成本  补贴系数   环境收益   专业公众   企业创新增益 政府声誉倍增  心理损失系数

          %  z       CG1      CE         gov        G0        G1      kappa 
C_input = [0.2,      25,      30,       0.5,       10,       0.8,        5];
          %政府监管 行政成本  环境损害-无 外部性分成 声誉损失 分权系数-无 声誉收益

for i=0.1:0.4:0.9
    for j=0.1:0.4:0.9
        for k=0.1:0.4:0.9
            [T,Y] = ode45(@(t,X) differential(t,X,A_input,B_input,C_input),[0 15],[i j k]);
            figure(1)
            grid on
            p1 = plot(T,Y(:,1),'r:','LineWidth',1);
            p2 = plot(T,Y(:,2),'g:','LineWidth',1);
            p3 = plot(T,Y(:,3),'b:','LineWidth',1);
%           plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1);
            hold on
        end
    end
end
ylim([0 1])
xlim([0 10]);xticks([0 5 10]);
ylabel('Frequency')
xlabel('Time')
legend([p1 p2 p3],{'Enterprises','Public','Government'})

%% Scenario 4, Stage 1  A(0,0,0) → D(0,0,1)
% Key condition：
% Pe*gov*kappa*eA- G1   <  G0 + beta*(PA + Pe*(eA - 1)*(gov - 1)) - CG1
% 
% After simplification, it is condition 5 in Figure 2: 
% CG1  < G0 + G1 + beta*(PA + Pe*(eA - 1)*(gov - 1))

          %  x       PA      Pe         CA        eA 0.1        Rc      alpha    beta 0      gamma 
A_input = [0.2,      40,      20,       35,        0.2,      10,         0,       0.05,       0.1];
          %企业治理 正常利润  外部性  完全治理成本 努力程度0.1 碳减排总收益  补贴系数  惩罚系数0    创新能力    
          
          %  y       CH        CB1     delta      CB0     epsilon      Rp        zeta       eta         theta       lambda
B_input = [0.2,     -30,      30,       0.8,       18,       0,         25,       0.1,       0.1,          0.1,        0.1];         
          %公众参与 健康损失-负 参与成本 渠道难度系数 防护成本  补贴系数   环境收益   专业公众   企业创新增益 政府声誉倍增  心理损失系数

          %  z       CG1      CE        gov        G0        G1      kappa 
C_input = [0.2,      25,      30,       0.5,       20,       0.8,        5];
          %政府监管 行政成本  环境损害-无 外部性分成 声誉损失10 分权系数-无 声誉收益   政府绩效损失

for i=0.1:0.4:0.9
    for j=0.1:0.4:0.9
        for k=0.1:0.4:0.9
            [T,Y] = ode45(@(t,X) differential(t,X,A_input,B_input,C_input),[0 15],[i j k]);
            figure(2)
            grid on
            p1 = plot(T,Y(:,1),'r:','LineWidth',1);
            p2 = plot(T,Y(:,2),'g:','LineWidth',1);
            p3 = plot(T,Y(:,3),'b:','LineWidth',1);
            hold on
        end
    end
end
ylim([0 1])
xlim([0 10]);xticks([0 5 10]);
ylabel('Frequency')
xlabel('Time')
legend([p1 p2 p3],{'Enterprises','Public','Government'},'FontSize',10)

%% Scenario 4, Stage 2  A(0,0,0) → D(0,0,1) → F(1,0,1)

% D(0,0,1) → F(1,0,1)
% Key condition：(alpha+beta)*(PA + Pe*(1- eA)*(1- gov)) + eA*Rc - eA*CA*(1 - gamma)  >  Pe*eA*(1-gov) 
% After simplification, it is condition 12 in Figure 2: 
% G0 + G1 + beta*(PA + Pe*(eA - 1)*(gov - 1)) < CG1

A_input = [0.2,      40,      20,       35,        0.4,      15,         0.06,       0.2,       0.1];
B_input = [0.2,     -30,      30,       0.8,       18,       0,         25,       0.1,       0.1,          0.1,        0.1];
C_input = [0.2,      25,      30,       0.5,       20,       0.8,        10];

for i=0.1:0.4:0.9
    for j=0.1:0.4:0.9
        for k=0.1:0.4:0.9
            [T,Y] = ode45(@(t,X) differential(t,X,A_input,B_input,C_input),[0 15],[i j k]);
            figure(3)
            grid on
            p1 = plot(T,Y(:,1),'r:','LineWidth',1);
            p2 = plot(T,Y(:,2),'g:','LineWidth',1);
            p3 = plot(T,Y(:,3),'b:','LineWidth',1);
            hold on
        end
    end
end
ylim([0 1])
xlim([0 10]);xticks([0 5 10]);
ylabel('Frequency')
xlabel('Time')
legend([p1 p2 p3],{'Enterprises','Public','Government'},"FontSize",10)

%% Scenario 4, Stage 3  A(0,0,0) → D(0,0,1) → F(1,0,1) → H(1,1,1)

% F(1,0,1) → H(1,1,1)
% Key condition：Rp*zeta - CB1*delta*(1-epsilon) > PB0-CB0
% After simplification, it is condition 11 in Figure 2: 
% Rp*zeta - CB1*delta*(1-epsilon) > PB0-CB0

A_input = [0.2,      40,      20,       35,        0.5,      20,         0.1,       0.2,       0.1];
B_input = [0.2,     -30,      30,       0.7,       18,       0.1,         25,       0.1,       0.1,          0.1,        0.1];
C_input = [0.2,      25,      30,       0.5,       25,       0.8,        15];

for i=0.1:0.4:0.9
    for j=0.1:0.4:0.9
        for k=0.1:0.4:0.9
            [T,Y] = ode45(@(t,X) differential(t,X,A_input,B_input,C_input),[0 15],[i j k]);
            figure(4)
            grid on
            p1 = plot(T,Y(:,1),'r:','LineWidth',1);
            p2 = plot(T,Y(:,2),'g:','LineWidth',1);
            p3 = plot(T,Y(:,3),'b:','LineWidth',1);
            hold on
        end
    end
end
ylim([0 1])
xlim([0 10]);xticks([0 5 10]);
ylabel('Frequency')
xlabel('Time')
legend([p1 p2 p3],{'Enterprises','Public','Government'})

%% Scenario 4, Stage 4  A(0,0,0) → D(0,0,1) → F(1,0,1) → H(1,1,1) → E(1,1,0)

% H(1,1,1) → E(1,1,0)
% Key condition：(G1+G0)*(theta + 1)*(theta + 1) < CG1 + alpha*(PA + Pe*(eA - 1)*(gov - 1)) + CB1*delta*epsilon
% It is condition 7 in Figure 2.

A_input = [0.2,      40,      10,       35,        0.7,      25,          0,       0.2,       0.3];
B_input = [0.2,     -30,      30,       0.5,       18,        0,         25,       0.2,       0.1,          0.1,        0.1];
C_input = [0.2,      25,      30,       0.5,       15,       0.8,        5];

for i=0.1:0.4:0.9
    for j=0.1:0.4:0.9
        for k=0.1:0.4:0.9
            [T,Y] = ode45(@(t,X) differential(t,X,A_input,B_input,C_input),[0 15],[i j k]);
            figure(5)
            grid on
            p1 = plot(T,Y(:,1),'r:','LineWidth',1);
            p2 = plot(T,Y(:,2),'g:','LineWidth',1);
            p3 = plot(T,Y(:,3),'b:','LineWidth',1);
            hold on
        end
    end
end
ylim([0 1.001])
xlim([0 10]);xticks([0 5 10]);
ylabel('Frequency')
xlabel('Time')
legend([p1 p2 p3],{'Enterprises','Public','Government'})

%% Key parameter simulation analysis. Scenario 1.
% Section 3.2.2 Key parameter simulation analysis.
% 
% Scenario 1
% Two parameters:
% Enterprises innovation capability & collaborative emission reduction benefits

A_input = [0.2,      40,      20,       35,        0.1,      10,         0,       0,       0.1];
B_input = [0.2,     -30,      30,       0.8,       18,       0,         25,       0.2,       0.1,          0.1,        0.1];
C_input = [0.2,      25,      30,       0.5,       10,       0.8,        5];

% Figure 5a
% Innovation of green technology【创新能力】
for i=0.3:0.5:0.8         %企业治污概率
    for j=0.3:0.6:0.9     %公众参与概率
        for k=0.3:0.6:0.9 %政府监管概率
            color = 0;
            for m=0:0.5:1
                color = color + 1;
                % Enterprises innovation capability【创新能力】
                A_input(9) = m;
                [T,Y] = ode45(@(t,X) differential2(t,X,A_input,B_input,C_input),[0 20],[i j k]);
                figure(6) %绘图
                grid on
                switch color
                    case 1
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[1 0 0]);
                    case 2
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0.2 1 0.2]);
                    case 3
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0 0 1]);
                end
                hold on
            end
        end
    end
end
lgd = legend('\gamma=0','\gamma=0.5','\gamma=1');
title(lgd,'Innovation of green technology')
xlim([0 1]);ylim([0 1]);zlim([0 1]);%坐标轴范围
view([0.3 0.2 0.15])
xlabel('Enterprise pollution control (x)')
ylabel('Public participation (y)')
zlabel('Government regulation(z)')


% Figure 5b
% Synergistic emission reduction benefits【协同减排收益】
for i=0.3:0.5:0.8         %企业治污概率
    for j=0.3:0.6:0.9     %公众参与概率
        for k=0.3:0.6:0.9 %政府监管概率
            color = 0;
            for m=10:15:40
                color = color + 1;
                % Collaborative emission reduction benefits【协同减排收益】
                A_input(6) = m;
                [T,Y] = ode45(@(t,X) differential2(t,X,A_input,B_input,C_input),[0 20],[i j k]);
                figure(7) %绘图
                grid on
                switch color
                    case 1
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[1 0 0]);
                    case 2
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0.2 1 0.2]);
                    case 3
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0 0 1]);
                end
                hold on
            end
        end
    end
end
lgd = legend('R_c=10','R_c=25','R_c=40');
title(lgd,'Synergistic emission reduction benefits')
xlim([0 1]);ylim([0 1]);zlim([0 1]);%坐标轴范围
view([0.3 0.2 0.15])
xlabel('Enterprise pollution control (x)')
ylabel('Public participation (y)')
zlabel('Government regulation(z)')

%% Key parameter simulation analysis. Stage 1 of Scenario 3.

%       A = [x,       PA,      Pe,     CA,        eA,        Rc,      alpha,      beta,     gamma];
         %企业治理 正常利润     外部性 完全治理成本 努力程度 碳减排总收益  补贴系数    惩罚系数    创新能力0.1
A_input = [0.2,      40,      20,       35,        0.2,      10,         0,       0.1,       0.1];

%       B = [y,      CH,       CB1,    delta,      CB0,    epsilon,     Rp,      zeta,        eta,        theta,      lambda;
         %公众参与 健康损失-负 参与成本 渠道难度系数 防护成本  补贴系数   环境收益     专业公众  企业创新增益    政府声誉增益 心理损失系数
B_input = [0.2,     -30,      30,       0.8,       18,       0,         25,       0.2,       0.1,          0.1,        0.1];

%       C = [z,       CG1,     CE无,    gov,       G0     kappa无     G1];
         %政府监管 行政成本   环境损害  外部性分成 声誉损失 分权系数  声誉收益 
C_input = [0.2,      25,      30,       0.5,       20,       0.8,        5];

% Figure 5c
% Complexity of participation channels【渠道难度系数】
for i=0.2        %企业治污概率
    for j=0.3:0.5:0.8     %公众参与概率
        for k=0.7%:0.5:0.7 %政府监管概率
            color = 0;
            for m=0.2:0.3:0.8
                color = color + 1;
                %【渠道难度系数】
                B_input(4) = m;
                [T,Y] = ode45(@(t,X) differential2(t,X,A_input,B_input,C_input),[0 20],[i j k]);
                figure(9) %绘图
                grid on
                switch color
                    case 1
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[1 0 0]);
                    case 2
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0.2 1 0.2]);
                    case 3
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0 0 1]);
                end
                hold on
            end
        end
    end
end
lgd = legend('Delta = 0.2','Delta = 0.5','Delta = 0.8');
title(lgd,'complexity of participation channels')
xlim([0 1]);ylim([0 1]);zlim([0 1]);%坐标轴范围
view([0.1 0.6 0.15])
xlabel('Enterprise pollution control (x)')
ylabel('Public participation (y)')
zlabel('Government regulation(z)')


% Figure 5d 
% Subsidy for public participation
for i=0.2:0.3:0.5         %企业治污概率
    for j=0.3:0.5:0.8     %公众参与概率
        for k=0.7%2:0.5:0.7 %政府监管概率
            color = 0;
            for m=[0 0.08 0.1]
                color = color + 1;
                %【公众补贴系数】
                B_input(6) = m;
                [T,Y] = ode45(@(t,X) differential2(t,X,A_input,B_input,C_input),[0 20],[i j k]);
                figure(10) %绘图
                grid on
                switch color
                    case 1
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[1 0 0]);
                    case 2
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0.2 1 0.2]);
                    case 3
                        plot3(Y(:,1),Y(:,2),Y(:,3),'LineWidth',1,'Color',[0 0 1]);
                end
                hold on
            end
        end
    end
end
lgd = legend('Epsilon = 0','Epsilon = 0.1','Epsilon = 0.2');
title(lgd,'Subsidy for public participation')
xlim([0 1]);ylim([0 1]);zlim([0 1]);%坐标轴范围
view([0.1 0.6 0.15])
xlabel('Enterprise pollution control (x)')
ylabel('Public participation (y)')
zlabel('Government regulation(z)')