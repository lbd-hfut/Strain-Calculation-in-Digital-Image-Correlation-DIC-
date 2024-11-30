clear
clc
close all

load star_displacement.mat
% N = 256;  % 位移场的大小
% [x, y] = meshgrid(1:N, 1:N);  % 创建二维网格
% u = sin(2 * pi * x / N);  % 水平位移场
% v = cos(2 * pi * y / N);  % 垂直位移场

% 1. 将位移场转换为频域（2D FFT）
U_freq = fft2(u);  % 对u进行2D傅里叶变换
V_freq = fft2(v);  % 对v进行2D傅里叶变换

% 2. 在频域中进行操作（例如滤波），此例中我们保持原样

% 3. 从频域场重构位移场（2D IFFT）
u_reconstructed = ifft2(U_freq);  % 对u进行2D逆傅里叶变换
v_reconstructed = ifft2(V_freq);  % 对v进行2D逆傅里叶变换

% 4. 可视化原始位移场和重构位移场
figure(1);
subplot(2, 2, 1);
imagesc(u);  % 绘制原始水平位移场
colorbar;
title('Original Horizontal Displacement (u)');

subplot(2, 2, 2);
imagesc(u_reconstructed);  % 绘制重构后的水平位移场
colorbar;
title('Reconstructed Horizontal Displacement (u)');

subplot(2, 2, 3);
imagesc(v);  % 绘制原始垂直位移场
colorbar;
title('Original Vertical Displacement (v)');

subplot(2, 2, 4);
imagesc(v_reconstructed);  % 绘制重构后的垂直位移场
colorbar;
title('Reconstructed Vertical Displacement (v)');


% 幅度谱
U_mag = abs(fftshift(U_freq));  % 对 U_freq 进行幅度计算并移动零频到中心
V_mag = abs(fftshift(V_freq));  % 对 V_freq 进行幅度计算并移动零频到中心

% 相位谱
U_phase = angle(fftshift(U_freq));  % 对 U_freq 进行相位计算并移动零频到中心
V_phase = angle(fftshift(V_freq));  % 对 V_freq 进行相位计算并移动零频到中心

% 可视化
figure(2);
% U_freq 的幅度谱
subplot(2, 2, 1);
imagesc(log(1 + U_mag));  % 使用 log 变换来增强显示（避免动态范围过大）
colorbar;
title('Magnitude Spectrum of U_freq');

% U_freq 的相位谱
subplot(2, 2, 2);
imagesc(U_phase);  % 相位通常是 -pi 到 pi，直接显示
colorbar;
title('Phase Spectrum of U_freq');

% V_freq 的幅度谱
subplot(2, 2, 3);
imagesc(log(1 + V_mag));  % 使用 log 变换来增强显示（避免动态范围过大）
colorbar;
title('Magnitude Spectrum of V_freq');

% V_freq 的相位谱
subplot(2, 2, 4);
imagesc(V_phase);  % 相位通常是 -pi 到 pi，直接显示
colorbar;
title('Phase Spectrum of V_freq');
