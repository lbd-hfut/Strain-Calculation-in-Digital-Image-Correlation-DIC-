% Clear workspace and figures
clear; close all; clc;

% Load displacement 
% load fpb_displacement.mat
load star_displacement.mat
[H, L] = size(u);

% Load and preprocess the image
ref_image = imread('000.bmp');
ref_gray = ref_image;
RG_ROI = ref_gray(400:400+H-1, 50:50+L-1);
RG_copy = flipud(RG_ROI); % Flip vertically
RG_ROI = RG_copy;


% Create the new sampling points
x_list = linspace(-1, 1, L);
y_list = linspace(-1, 1, H);
[X, Y] = meshgrid(x_list, y_list);
displacement_field_u = u;
displacement_field_v = v;
X_new = X - displacement_field_u / L / 2;
Y_new = Y - displacement_field_v / H / 2;

% Perform bilinear interpolation
DG_ROI = interp2(X, Y, double(RG_ROI), X_new, Y_new, 'linear');

% ROI = zeros(H,L); b = 5;
% ROI(b+1:H-b, b+1:L-b) = 255;
ROI = ones(H,L)*255;

% Save the images
subfolder = 'restructed_image';
if ~exist(subfolder, 'dir')
    mkdir(subfolder);
end
RG_path = fullfile(subfolder, 'RG.bmp');
DG_path = fullfile(subfolder, 'DG.bmp');
ROI_path = fullfile(subfolder, 'ROI.bmp');
imwrite(uint8(RG_ROI), RG_path);
imwrite(uint8(DG_ROI), DG_path);
imwrite(uint8(ROI), ROI_path);
