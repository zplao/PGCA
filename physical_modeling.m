clc; clear all; close all;

%% 从实测加速度反推路径分量
% 1. 加载实测数据（示例）
load('data/rawdata/G3_1200rpm.mat'); % 含a_measured, Fs, t
a_measured_ori = Data(:, 3);

Fs = 32000;  % 采样频率（与仿真参数一致）
dt = 1/Fs;  % 采样频率（与仿真参数一致）
low_cutoff = 400;         % 带通下限
high_cutoff = 6000;       % 带通上限
order = 4;                % 滤波器阶数
cutoff = 1200;              % 截止频率（Hz），也可设为 3000

% ========== 低通滤波器设计 ==========
Wn = cutoff / (Fs/2);       % 归一化截止频率
[b, a] = butter(order, Wn, 'low');  % Butterworth 低通滤波器
a_measured = filtfilt(b, a, a_measured_ori);    % 零相位滤波，推荐
% a_measured = a_measured_ori;

%% calculation of the period
rho=7850;                          %density 密度
m=3/1000;                        %modulus 模量
z1=28; z2=40; z3=34;z4=34;         %teeth number 齿数
B1=16/1000;B3=16/1000;B4=16/1000;  B2=16/1000;           %face width 表面宽度
rint1=24/1000; rint2=24/1000;rint3=24/1000;  %radius of the shaft (hub radius)  轴半径(中心半径)
alpha0=20*pi/180;                  %pressure angle 压力角
Torque=40;                         %input torque ；转矩
r1=m*z1/2; r3=m*z3/2;              %radius of the reference circle
r2=m*z2/2;r4=m*z4/2;
rb1=r1*cos(alpha0);   rb3=r3*cos(alpha0);              %radius of the base circle  基圆半径
rb2=r2*cos(alpha0);   rb4=r4*cos(alpha0);
Rot_period=1200;                     %number of the rotational period 旋转周期个数
% Tm_num=100;                        %data points in a meshing period 
% zhouqi_shu=z1*Rot_period;          %number of the meshing period  啮合部分数
% step_num=zhouqi_shu*(Tm_num);      %total data points of the simulation  模拟总数据点

%% Parameter of the lumped mass
m1=pi*(r1^2-rint1^2)*B1*rho;                       %齿轮1的质量
J1=1/2*m1*(rint1^2+r1^2);                          %齿轮1的转动惯量
m3=pi*(r2^2-rint2^2)*B2*rho;                       %齿轮3的质量
J3=1/2*m3*(rint2^2+r2^2);                          %齿轮3的质量转动惯量
m2=m1+m3;                                          %齿轮2的质量，2是另外一对1，3的连接物
J2=J1+J3;   
m4=1;m5=3; m6=2; m7=23.5;                          %m4-m6是被动部分，m7是target
M=diag([m1,J1,m2,J2,m3,J3,m4,m4,m5,m5,m6,m6,m7]);           %质量矩阵

Nspeed=1200;           %input speed [rpm]
w1=Nspeed*2*pi/60;     %speed of gear1 [rad/s]
w2=w1*z1/z2;           %speed of gear1 [rad/s]
T=2*pi/(w1*z1);        %meshing period
% dt=T/Tm_num;           %time step
fm=Nspeed/60*z1;       %meshing frequency
% Tm_num=T/dt;                        %data points in a meshing period 
Tm_num=57.142857142857146;
zhouqi_shu=z1*Rot_period;          %number of the meshing period  啮合部分数
% step_num=zhouqi_shu*fix(Tm_num);      %total data points of the simulation  模拟总数据点
step_num=1915200;      %total data points of the simulation  模拟总数据点

%% Time varying meshing stiffness and Time varying meshing damping 时变啮合刚度和时变啮合阻尼
K_health=load('K_health.txt');
K_ext=interp1(linspace(0,1,length(K_health)),K_health(1:60),linspace(0,1,Tm_num)); 
% interp1 一维数据插值（表查找）
KM=repmat(K_ext,1,zhouqi_shu);
% B = repmat(A,m,n)，将矩阵 A 复制 m×n 块，即把 A 作为 B 的元素，B 由 m×n 个 A 平铺而成
CM=2*0.02*sqrt(KM/(r1^2/J1+r2^2/J2));

%% Project vector of the meshing element 啮合部分投影矢量
V=[1,rb1,1,-rb2];V2=[1,rb3,1,-rb4];                %两个啮合面V，V2的啮合矩阵投影
unit_VV=zeros(length(M));
unit_VV(1:4,1:4)=V'*V;                              
unit_VV2=zeros(length(M));
unit_VV2(3:6,3:6)=V2'*V2;
unit_VV=unit_VV+unit_VV2;                          %放入矩阵

%% Stiffness and damping parameter of the spring-damping element
k14=1e7; k25=1e7;k36=1e7; k47=1e8;k57=1e8; k67=1e8; k07=1e8;
c14=1e4; c25=1e4;c36=1e4; c47=1e3; c57=1e3;c67=1e3; c07=1e3;

k47r=1e7;k57r=1e7; k67r=1e7;
c47r=1e4; c57r=1e4;c67r=1e4;

k14r=1e6; k25r=1e6;k36r=1e6;
c14r=1e5; c25r=1e5;c36r=1e5;

%% Matirx assembling of the whole system
K=zeros(length(M));
K([1,7],[1,7])=K([1,7],[1,7])+[k14,-k14;-k14,k14];
K([1,8],[1,8])=K([1,8],[1,8])+[k14r,-k14r;-k14r,k14r];
K([3,9],[3,9])=K([3,9],[3,9])+[k25,-k25;-k25,k25];
K([3,10],[3,10])=K([3,10],[3,10])+[k25r,-k25r;-k25r,k25r];
K([5,11],[5,11])=K([5,11],[5,11])+[k36,-k36;-k36,k36];
K([5,12],[5,12])=K([5,12],[5,12])+[k36r,-k36r;-k36r,k36r];
K([7,13],[7,13])=K([7,13],[7,13])+[k47,-k47;-k47,k47];
K([8,13],[8,13])=K([8,13],[8,13])+[k47r,-k47r;-k47r,k47r];
K([9,13],[9,13])=K([9,13],[9,13])+[k57,-k57;-k57,k57];
K([10,13],[10,13])=K([10,13],[10,13])+[k57r,-k57r;-k57r,k57r];
K([11,13],[11,13])=K([11,13],[11,13])+[k67,-k67;-k67,k67];
K([12,13],[12,13])=K([12,13],[12,13])+[k67r,-k67r;-k67r,k67r];
K(13,13)=K(13,13)+k07;

C=zeros(length(M));
C([1,7],[1,7])=C([1,7],[1,7])+[c14,-c14;-c14,c14];
C([1,8],[1,8])=C([1,8],[1,8])+[c14r,-c14r;-c14r,c14r];
C([3,9],[3,9])=C([3,9],[3,9])+[c25,-c25;-c25,c25];
C([3,10],[3,10])=C([3,10],[3,10])+[c25r,-c25r;-c25r,c25r];
C([5,11],[5,11])=C([5,11],[5,11])+[c36,-c36;-c36,c36];
C([5,12],[5,12])=C([5,12],[5,12])+[c36r,-c36r;-c36r,c36r];
C([7,13],[7,13])=C([7,13],[7,13])+[c47,-c47;-c47,c47];
C([8,13],[8,13])=C([8,13],[8,13])+[c47r,-c47r;-c47r,c47r];
C([9,13],[9,13])=C([9,13],[9,13])+[c57,-c57;-c57,c57];
C([10,13],[10,13])=C([10,13],[10,13])+[c57r,-c57r;-c57r,c57r];
C([11,13],[11,13])=C([11,13],[11,13])+[c67,-c67;-c67,c67];
C([12,13],[12,13])=C([12,13],[12,13])+[c67r,-c67r;-c67r,c67r];
C(13,13)=K(13,13)+c07;

%% Natural frequency of the whole system (coupled system)
K_mean=K+mean(K_health)*unit_VV;                    
D_whole=eig(K_mean/M);
Freq_whole=sqrt(sort(real(abs(D_whole))))/(2*pi);
%% Numerical iteration in time domain
[allnode,allnodeXd,allnodeXdd]=deal(zeros(length(M),step_num));
[X,Xd,Xdd]=deal(zeros(length(M),1));  

F0=zeros(length(M),1); 
F0(2)=F0(2)+Torque;  %Torque 输入转矩
F0(4)= F0(4)+rb2*Torque/rb1;

for i=1:step_num
    K_t=K+KM(i)*unit_VV;
    C_t=C+CM(i)*unit_VV;
    [X,Xd,Xdd]=newmark(M,C_t,K_t,F0,dt,0.5,0.25,X,Xd,Xdd);
    allnode(:,i)=X;
    allnodeXd(:,i)=Xd;
    allnodeXdd(:,i)=Xdd;
end
time=linspace(0,zhouqi_shu*T,step_num);

% steady7=allnodeXdd(10,end-(Rot_period-4)*z1*Tm_num+1:end);
steady6=allnodeXdd(9,end-(Rot_period-4)*z1*Tm_num+1:end);
steady5=allnodeXdd(8,end-(Rot_period-4)*z1*Tm_num+1:end);
steady4=allnodeXdd(7,end-(Rot_period-4)*z1*Tm_num+1:end);
steady_time=time(end-(Rot_period-4)*z1*Tm_num+1:end);
%% vibration response in frequency domain
% [~,A10]=Num_fft(steady7,Fs);  % ~ 通常用于忽略不需要的函数返回值。
[~,A9]=Num_fft(steady6,Fs);
[~,A8]=Num_fft(steady5,Fs);
[~,A7]=Num_fft(steady4,Fs);   %A3,A1为频率响应
%% Matirx assembling of the decoupled system (passive part) 被动部分的解耦系统的矩阵装配
Md = diag([m4, m4, m5, m5, m6, m6, m7]);  % 6个轴承自由度的质量矩阵

Kd=zeros(length(Md));
Kd([1,7],[1,7])=Kd([1,7],[1,7])+[k47,-k47;-k47,k47];
Kd([2,7],[2,7])=Kd([2,7],[2,7])+[k47r,-k47r;-k47r,k47r];
Kd([3,7],[3,7])=Kd([3,7],[3,7])+[k57,-k57;-k57,k57];
Kd([4,7],[4,7])=Kd([4,7],[4,7])+[k57r,-k57r;-k57r,k57r];
Kd([5,7],[5,7])=Kd([5,7],[5,7])+[k67,-k67;-k67,k67];
Kd([6,7],[6,7])=Kd([6,7],[6,7])+[k67r,-k67r;-k67r,k67r];
Kd(7,7)=Kd(7,7)+k07;

Cd=zeros(length(Md));
Cd([1,7],[1,7])=Cd([1,7],[1,7])+[c47,-c47;-c47,c47];
Cd([2,7],[2,7])=Cd([2,7],[2,7])+[c47r,-c47r;-c47r,c47r];
Cd([3,7],[3,7])=Cd([3,7],[3,7])+[c57,-c57;-c57,c57];
Cd([4,7],[4,7])=Cd([4,7],[4,7])+[c57r,-c57r;-c57r,c57r];
Cd([5,7],[5,7])=Cd([5,7],[5,7])+[c67,-c67;-c67,c67];
Cd([6,7],[6,7])=Cd([6,7],[6,7])+[c67r,-c67r;-c67r,c67r];
Cd(4,4)=Cd(4,4)+c07;

% 2. 截取稳态段（匹配仿真参数）
% N_steady = Nspeed/60 * z1 * Tm_num * Rot_period; % 根据转速和齿数计算
N_steady = (Rot_period-4)*z1*Tm_num;
a_measured_1 = a_measured(end - N_steady + 1 : end)'; % N_steady_samples需匹配仿真时长
% a_measured_steady = mapminmax(a_measured_1, -1, 1);
a_measured_steady = a_measured_1;

% 3. 计算频响函数矩阵（需预先定义Md, Kd, Cd）
% 使用实测信号替换仿真信号 A10 (m7的加速度)
[Freq_ex, A10_measured] = Num_fft(a_measured_steady, Fs);

% 更新力反演循环（修改Omega_f和A10为实测数据）
Omega_f = Freq_ex * 2 * pi;
A_Fb_measured = zeros(6, length(Omega_f));
Contribution_measured = zeros(6, length(Omega_f));

% 计算频响函数
Hd = zeros(7, 7, length(Omega_f));  % 6轴承 + m7
for n = 1:length(Omega_f)
    Hd(:,:,n) = Omega_f(n)^2 * inv(-Md * Omega_f(n)^2 + Cd * 1i * Omega_f(n) + Kd);
end

% 使用正则化方法（Tikhonov）反演6个轴承力：
for n = 1:length(Omega_f)
    H = Hd(7, 1:6, n);  % 从6个轴承到m7的频响
    A_Fb_measured(:, n) = Tikhonov(H.', A10_measured(n));
    Contribution_measured(:, n) = H.' .* A_Fb_measured(:, n);
end

% 计算因果矩阵依据
DOF_a=[1,3,5];             %主动部分x位移,1,3,5
DOF_p=[7,8,9,10,11,12];             %被动部分x位移,6,7,8
for n=1:length(Omega_f)
    %% FRF of the passive part
    % Hd=Omega_f(n)^2*inv(-Md*Omega_f(n)^2+Cd*1i*Omega_f(n)+Kd);   
    %% FRF of the whole system
    H=Omega_f(n)^2*pinv(-M*Omega_f(n)^2+C*1i*Omega_f(n)+K_mean);   
    Hctp=H(10,DOF_p);      %提取p对t的传递函数下同
    Hcta=H(10,DOF_a);      %提取对t的传递函数下同
    Hcaa=H(DOF_a,DOF_a);   %提取a对a的矩阵下同
    Hcpp=H(DOF_p,DOF_p);   %提取p对P的矩阵下同
    Hcpa=H(DOF_p,DOF_a);   %提取p对a的矩阵下同
    Hcap=H(DOF_a,DOF_p);   %提取a对p的矩阵下同
end

% 5. 时域重构
% 绝对路径贡献
Contribution_TD = real(ifft(Contribution_measured, [], 2)) * length(Omega_f);

% 验证重构精度
A_reconstructed = sum(Contribution_TD, 1);

% 绘图
figure;
subplot(8,1,1)
plot(steady_time, a_measured_steady, 'k'); 
xlim([0.2, 0.6]); 
title('原始信号')

subplot(8,1,2)
plot(steady_time(1:2:end), Contribution_TD(1,:), 'r');
xlim([0.2, 0.6]); 
title('Path 1')

subplot(8,1,3)
plot(steady_time(1:2:end), Contribution_TD(2,:), 'g');
xlim([0.2, 0.6]); 
title('Path 2')

subplot(8,1,4)
plot(steady_time(1:2:end), Contribution_TD(3,:), 'b');
xlim([0.2, 0.6]); 
title('Path 3')

subplot(8,1,5)
plot(steady_time(1:2:end), Contribution_TD(4,:), 'b');
xlim([0.2, 0.6]); 
title('Path 4')

subplot(8,1,6)
plot(steady_time(1:2:end), Contribution_TD(5,:), 'b');
xlim([0.2, 0.6]); 
title('Path 5')

subplot(8,1,7)
plot(steady_time(1:2:end), Contribution_TD(6,:), 'b');
xlim([0.2, 0.6]); 
title('Path 6')

subplot(8,1,8)
plot(steady_time(1:2:end), A_reconstructed, 'm');
xlim([0.2, 0.6]); 
title('重构信号');

% 频域路径分解
figure;
subplot(8,1,1)
plot(Freq_ex, abs(A10_measured), 'k', 'LineWidth', 1.5);
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('原始信号')

subplot(8,1,2)
[Freq,T_FD_1]=Num_fft(Contribution_TD(1, :),Fs/2);
plot(Freq, abs(T_FD_1), 'r');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('Path 1')

subplot(8,1,3)
[Freq,T_FD_2]=Num_fft(Contribution_TD(2, 1:320000),Fs/2);
plot(Freq, abs(T_FD_2), 'g');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('Path 2')

subplot(8,1,4)
[Freq,T_FD_3]=Num_fft(Contribution_TD(3, 1:320000),Fs/2);
plot(Freq, abs(T_FD_3), 'r');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('Path 3')

subplot(8,1,5)
[Freq,T_FD_4]=Num_fft(Contribution_TD(4, 1:320000),Fs/2);
plot(Freq, abs(T_FD_4), 'r');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('Path 4')

subplot(8,1,6)
[Freq,T_FD_5]=Num_fft(Contribution_TD(5, 1:320000),Fs/2);
plot(Freq, abs(T_FD_5), 'r');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('Path 5')

subplot(8,1,7)
[Freq,T_FD_6]=Num_fft(Contribution_TD(6, 1:320000),Fs/2);
plot(Freq, abs(T_FD_6), 'r');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('Path 6')

subplot(8,1,8)
[Freq,T_FD_7]=Num_fft(A_reconstructed,Fs/2);
plot(Freq, abs(T_FD_7), 'm');
xlim([0, 2000]); 
xlabel('频率 (Hz)'); ylabel('幅值');
title('重构信号')
