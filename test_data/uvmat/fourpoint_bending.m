% Four-Point Bending Test Pure Bending Region Displacement Field Plotting
% Clear workspace and figures
clear; close all; clc;

% 1. Parameter settings
L = 1.0;          % Beam length (meters)
a = 0.25;         % Distance from loading point to support point (meters)
b = 2;          % Distance between loading points (meters)
P = 1000;         % Applied force (Newtons)
E = 210e9;        % Young's modulus (Pascals)
I = 1.0e-6;       % Moment of inertia of the cross-section (m^4)

% 2. Define the pure bending region
x_start = a;          % Starting point of the pure bending region
x_end = a + b;        % End point of the pure bending region
x = linspace(x_start, x_end, 1024);  % x-coordinates within the pure bending region

% Define y-coordinates (assuming beam thickness is 10cm, symmetrically distributed)
y_max = 1;         % Maximum y displacement (meters)
y = linspace(-y_max, y_max, 256);  % y-coordinates along the beam thickness direction

[X, Y] = meshgrid(x, y); 

% 3. Calculate transverse displacement w(x)
M = P * a;            % Pure bending moment
v = (M / (2 * E * I)) * (X - (a+b/2)).^2;   % v displacement formula
v = v * 1e+3; C1 = max(v(:)); C2 = min(v(:));
v = (v - C2)/(C1 - C2) * 2 - 1;

% 4. Calculate longitudinal displacement u(x, y)
u = (M * Y / (E * I)) .* (a + b/2 - X);     % u displacement formula
u = u * 1e+3; B1 = max(u(:)); B2 = min(u(:));
u = (u - B2)/(B1 - B2) * 2 - 1;


[m,n] = size(u);
mean_value = 0; std_value = 0.0; 
white_noise_matrix_u = mean_value + std_value * randn(m, n);
white_noise_matrix_v = mean_value + std_value * randn(m, n);

u = u + white_noise_matrix_u;
v = v + white_noise_matrix_v;

% 5. Plot transverse displacement w(x)
figure(1);
imshow(v, 'Colormap', jet);
title('Longitudinal Displacement Field v in the Pure Bending Region', 'FontSize', 14);
colorbar; caxis('auto');
grid on; axis on;

% 6. Plot longitudinal displacement u(x, y) in 3D
figure(2);
imshow(u, 'Colormap', jet);
title('Transverse Displacement Field u in the Pure Bending Region', 'FontSize', 14);
colorbar; caxis('auto');
grid on; axis on;

% 7. Save the displacement result
save('fpb_displacement.mat', 'u', 'v');

% 8. Calculate partial derivatives
u_x = -M*Y/(E*I)*1e+3;
u_y = M/(E*I)*(a+b/2-X)*1e+3;

v_x = M/(E*I)*(a+b/2-X)*1e+3;
v_y = zeros(size(v_x));

figure;
subplot(311)
imshow(u_x(5:251,5:1019), 'Colormap', jet);
title('ex');
colorbar; caxis('auto');
axis on;

exy = (v_x/2+u_y/2);
subplot(312)
imshow(exy(5:251,5:1019), 'Colormap', jet);
title('exy');
colorbar; caxis('auto');
axis on;

subplot(313)
imshow(v_y(5:251,5:1019), 'Colormap', jet);
title('ey');
colorbar; caxis('auto');
axis on;