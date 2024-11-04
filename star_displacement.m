% Star Displacement Field Plotting
% Clear workspace and figures
clear; close all; clc;

% 1. Parameter Settings
H = 256; % Image height
L = 1024; % Image length
pmax = 120; % Controls the number of stripes on the far left
pmin = 10;  % Controls the number of stripes on the far right

% 2. Create Grid
x = 1:L;
y = 1:H;
[X, Y] = meshgrid(x, y);

% 3. Calculate Wavelength Distribution
pwave = pmin + X * (pmax - pmin) / L;

% 4. Calculate v1 matrix
v1 = 0.5 * cos((Y - H/2) * 2 * pi ./ pwave);

% 5. Calculate the Minimum and Maximum Values of the Matrix
min_val = min(v1(:));
max_val = max(v1(:));

% 6. Normalize Matrix to the Range [-1, 1]
v = 2 * (v1 - min_val) / (max_val - min_val) - 1;
u = zeros(H, L);

[m,n] = size(u);
mean_value = 0; std_value = 0.2; 
white_noise_matrix_u = mean_value + std_value * randn(m, n);
white_noise_matrix_v = mean_value + std_value * randn(m, n);

u = u + white_noise_matrix_u;
v = v + white_noise_matrix_v;

% 7. Plot Image
figure;
imshow(v, 'Colormap', jet);
title('Normalized Star Displacement Field');
colorbar; caxis('auto');
axis on;

% 8. Save the Displacement Result
save('star_displacement.mat', 'u', 'v');

% 9. Calculate partial derivatives
v_y = -pi*L*sin( (2*pi*L* (Y-H/2) ) ./ (pmin*L+ (pmax-pmin)*X ))./(pmin*L+ (pmax-pmin)*X )*2;

v_x = (pmax-pmin)/2 * sin( (2*pi*L*(Y-H/2))./(pmin*L+ (pmax-pmin)*X)).*(pmin*L+ (pmax-pmin)*X).^(-2).* (2*pi*L*(Y-H/2))*2;

figure;
subplot(311)
imshow(zeros(size(v_x)), 'Colormap', jet);
title('ex');
colorbar; caxis('auto');
axis on;

subplot(312)
imshow(v_x, 'Colormap', jet);
title('exy');
colorbar; caxis('auto');
axis on;

subplot(313)
imshow(v_y, 'Colormap', jet);
title('ey');
colorbar; caxis('auto');
axis on;
