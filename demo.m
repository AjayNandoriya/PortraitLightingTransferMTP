addpath(genpath('.'));
% Siggraph 2017 MTP demo script
source_img = im2double(imread('img/images/mean_002_1_15.jpg'));
ref_img = im2double(imread('img/images/mean_003_1_15.jpg'));

tic;
out_img = sig17mtp(source_img,ref_img);
toc;

figure(1);
subplot(131);imshow(source_img,[]);
subplot(132);imshow(ref_img,[]);
subplot(133);imshow(out_img,[]);
