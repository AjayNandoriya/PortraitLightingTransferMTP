clear all;clc;
scale=0.25;
LAB_FLAG=false;
ONLY_L=false;
WHOLE_IMG=false;
INVERT=false;


Niter=100;
wn= 1;wp= 1;wc = 1;

w_de = wc;
wn =wn/w_de;
wp = wp/w_de;
wc = wc/w_de;

outdir = 'sig_17_mtp_results';


% source.pid = id;
% source.lid = l(k);
% % source.fname= fullfile('\\DS2015XS\Kilimanjaro\Dropbox_MIT\MERL_facial',pidname{source.pid},sprintf('refl1_%03d_15.png',source.lid));
% % source.fname= fullfile('\\DS2015XS\Kilimanjaro\Dropbox_MIT\MERL_facial',pidname{source.pid},'blend_015.png');
% source.fname = in_name{k};
% source.fname_normals = fullfile('C:\Users\qcri\Documents\Ajay\Face\dataset\mean_img_rotate',sprintf('normals_%03d.png',pid(source.pid)));
% source.fname_masks= fullfile('C:\Users\qcri\Documents\Ajay\Face\dataset\mean_img_rotate',sprintf('masks_%03d.png',pid(source.pid)));
% source.fname_pos= fullfile('C:\Users\qcri\Documents\Ajay\Face\dataset\mean_img_rotate',sprintf('depth_%03d.png',pid(source.pid)));
% ref.pid = 4;
% ref.lid = l(k);
% ref.fname = fullfile('\\DS2015XS\Kilimanjaro\Dropbox_MIT\MERL_facial',pidname{ref.pid},sprintf('refl1_%03d_15.png',ref.lid));
% ref.fname_normals = fullfile('C:\Users\qcri\Documents\Ajay\Face\dataset\mean_img_rotate',sprintf('normals_%03d.png',pid(ref.pid)));
% ref.fname_masks= fullfile('C:\Users\qcri\Documents\Ajay\Face\dataset\mean_img_rotate',sprintf('masks_%03d.png',pid(ref.pid)));
% ref.fname_pos= fullfile('C:\Users\qcri\Documents\Ajay\Face\dataset\mean_img_rotate',sprintf('depth_%03d.png',pid(ref.pid)));

source.pid = 9;
source.lid = 0;
source.fname= fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('mean_%03d_1_15.jpg',source.pid));
source.fname_normals = fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('normals_%03d.png',source.pid));
source.fname_masks= fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('masks_%03d.png',source.pid));
source.fname_pos= fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('depth_%03d.png',source.pid));

ref.pid = 8;
ref.lid = 0;
ref.fname = fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('mean_%03d_1_15.jpg',ref.pid));
ref.fname_normals = fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('normals_%03d.png',ref.pid));
ref.fname_masks= fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('masks_%03d.png',ref.pid));
ref.fname_pos= fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('depth_%03d.png',ref.pid));

source.gt_fname= fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp\img',sprintf('mean_%03d_to_%03d.PNG',source.pid,ref.pid));
source.img = permute(im2double(imread(source.fname)),[1 2 3]);
source.img = imfilter(source.img,fspecial('gaussian',5),'symmetric');
source.ori_img = source.img;
source.gt_img = im2double(imread(source.gt_fname));


ptsOriginal  = detectSURFFeatures(rgb2gray(source.ori_img));
ptsDistorted = detectSURFFeatures(rgb2gray(source.gt_img));
[featuresOriginal,  validPtsOriginal]  = extractFeatures(rgb2gray(source.ori_img),  ptsOriginal);
[featuresDistorted, validPtsDistorted] = extractFeatures(rgb2gray(source.gt_img), ptsDistorted);
indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));
[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'similarity');
outputView = imref2d(size(source.ori_img));
source.gt_img= imwarp(source.gt_img,tform,'OutputView',outputView);
% figure, imshowpair(source.ori_img,recovered,'montage')

% source.normal_img = permute(im2double(imread(source.fname_normals)),[2 1 3]);
source.normal_img = im2double(imread(source.fname_normals));
source.normal_img = imresize(source.normal_img,size(source.img(:,:,1)));
source.normal_img = source.normal_img*2-1;
% source.mask_img = permute(im2double(imread(source.fname_masks)),[2 1 3]);
source.mask_img = imresize(im2double(imread(source.fname_masks)),size(source.img(:,:,1)));
source.mask_img = imerode(source.mask_img(:,:,1),strel('disk',3));
source.pos_img = im2double(imread(source.fname_pos));
source.pos_img = imresize(source.pos_img(end:-1:1,:,:),size(source.img(:,:,1)));





ref.img = permute(im2double(imread(ref.fname)),[1 2 3]);
% ref.normal_img = permute(im2double(imread(ref.fname_normals)),[2 1 3]);
ref.normal_img = im2double(imread(ref.fname_normals));
ref.normal_img = imresize(ref.normal_img,size(ref.img(:,:,1)));
ref.normal_img = ref.normal_img*2-1;
% ref.mask_img = permute(im2double(imread(ref.fname_masks)),[2 1 3]);
ref.mask_img = imresize(im2double(imread(ref.fname_masks)),size(ref.img(:,:,1)));
ref.mask_img = imerode(ref.mask_img(:,:,1),strel('disk',3));
ref.pos_img = im2double(imread(ref.fname_pos));
ref.pos_img = imresize(ref.pos_img(end:-1:1,:,:),size(ref.img(:,:,1)));


% resize
source.img = imresize(source.img,scale);
if(LAB_FLAG)
    source.img = rgb2lab(source.img);
end
if(INVERT)
    source.img = 1-(source.img);
end
source.normal_img = imresize(source.normal_img,scale);
source.mask_img = imresize(source.mask_img,scale);
source.pos_img = imresize(source.pos_img,scale);
% source.gt_img = imresize(source.gt_img,scale);
ref.img = imresize(ref.img,scale);
if(LAB_FLAG)
    ref.img = rgb2lab(ref.img);
end
if(INVERT)
    ref.img = 1-(ref.img);
end
ref.normal_img = imresize(ref.normal_img,scale);
ref.mask_img = imresize(ref.mask_img,scale);
ref.pos_img = imresize(ref.pos_img,scale);


if(WHOLE_IMG)
    source.normal_img = blending(source.normal_img,zeros(size(source.normal_img)),source.mask_img);
    source.pos_img = blending(source.pos_img,zeros(size(source.normal_img)),source.mask_img);
    source.mask_img = true(size(source.mask_img));
    ref.normal_img = blending(ref.normal_img,zeros(size(ref.normal_img)),ref.mask_img);
    ref.normal_img = blending(ref.pos_img,zeros(size(ref.pos_img)),ref.mask_img);
    ref.mask_img = true(size(ref.mask_img));
end
% source.mask_img = true(size(source.mask_img));
% ref.mask_img = true(size(ref.mask_img));


[source.indy,source.indx] = find(source.mask_img);
if(ONLY_L)
    source.cimg = reshape(source.img(:,:,1),[],1);
else
    source.cimg = reshape(source.img,[],3);
end
source.nimg = reshape(source.normal_img,[],3);
source.pimg = reshape(source.pos_img,[],3);
source.ind  = sub2ind(size(source.mask_img),source.indy,source.indx);
source.s = [wc*source.cimg(source.ind,:) source.nimg(source.ind,:)*wn source.pimg(source.ind,1:2)*wp];
% source.s = [wc*source.cimg(source.ind,:) source.pimg(source.ind,1:2)*wp];

[ref.indy,ref.indx] = find(ref.mask_img);
if(ONLY_L)
    ref.cimg = reshape(ref.img(:,:,1),[],1);
    ColorDim=1;
else
    ref.cimg = reshape(ref.img,[],3);
    ColorDim=3;
end
ref.nimg = reshape(ref.normal_img,[],3);
ref.pimg = reshape(ref.pos_img,[],3);
ind  = sub2ind(size(ref.mask_img),ref.indy,ref.indx);
ref.s = [wc*ref.cimg(ind,:) ref.nimg(ind,:)*wn ref.pimg(ind,1:2)*wp];
% ref.s = [wc*ref.cimg(ind,:) ref.pimg(ind,1:2)*wp];




Ndim = size(source.s,2);
alpha= 1;
Si = source.s;
Ls = size(Si,1);
Sr = ref.s;

tic;
sigma=0.1;
Si_extra = Si;
for i=1:5
    Si_extra(i*Ls+(1:Ls),:) =Si;
    Si_extra(i*Ls+(1:Ls),1:ColorDim)=Si(:,1:ColorDim)+randn(Ls,ColorDim)*sigma;
end

So = Si_extra;

R{1}=eye(Ndim);
% R{1}=[eye(Ndim);((Ndim-1)*ones(Ndim)-Ndim*eye(Ndim))/Ndim];
for kiter=2:Niter
    R{kiter} = R{1}*orth(randn(Ndim));
end

So = pdf_transfer(Si_extra', Sr', R, 0.2);
So = So';

output = source.cimg;
output(source.ind,:)=So(1:Ls,1:ColorDim);
output = reshape(output,size(source.img(:,:,1:ColorDim)));
if(ONLY_L)
    output(:,:,2:3)=source.img(:,:,2:3);
end
if(INVERT)
    source.img = 1-(source.img);
    output= 1-(output);
    ref.img = 1-(ref.img);
end
if(LAB_FLAG)
    source.img= lab2rgb(source.img);
    ref.img= lab2rgb(ref.img);
    output = lab2rgb(output);
end

output_r = regrain(source.img,output,[0.125]);

t=toc;
fprintf('processing time = %f sec\n',t);



subplot(221);imshow(source.img,[]);title('source')
% subplot(232);imshow(source.normal_img/2+0.5,[]);title('normal')
subplot(222);imshow(output_r,[]);title('output')
subplot(223);imshow(ref.img,[]);title('ref')
% subplot(235);imshow(ref.normal_img/2+0.5,[]);title('normal')
subplot(224);imshow(source.gt_img,[]);title('Ground truth')

% subplot(231);imshow(source.img);title('source')
% subplot(232);imshow(source.normal_img/2+0.5,[]);title('normal')
% subplot(233);imshow(output);title('mask')
% subplot(234);imshow(ref.img);title('ref')
% subplot(235);imshow(ref.normal_img/2+0.5,[]);title('normal')
% subplot(236);imshow(source.gt_img);title('Ground truth')

%%
% img_out = output_r(1:320,1:270,:);
% clear img_out2
% for c=1:size(source.gt_img,3)
%     
%     imgt = source.gt_img(:,1:end-5,c);
%     h = imhist(imgt);
%     imgt = img_out(:,:,c)./max(max(img_out(:,:,c)));
%     imgt = histeq(imgt,h);
%     img_out2(:,:,c)=imgt;
%     
% end

diff = source.ori_img -imresize(source.img,size(source.ori_img(:,:,1)));
output =imresize(output,size(source.ori_img(:,:,1)))+diff;
output_r =imresize(output_r,size(source.ori_img(:,:,1)))+diff;
subplot(222);imshow(output_r,[]);title('output')
mask = imresize(source.mask_img,size(source.ori_img(:,:,1)));
mask = repmat(mask,[1 1 3]);
gtimg = source.gt_img;
% gtimg = [zeros(size(gtimg,1),19,3) gtimg];
imwrite(output_r.*mask,fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp',sprintf('%dto%d_ours_masked.png',source.pid,ref.pid)));
imwrite(gtimg.*mask,fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp',sprintf('%dto%d_paper_masked.png',source.pid,ref.pid)));
imwrite(output_r,fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp',sprintf('%dto%d_ours.png',source.pid,ref.pid)));
imwrite(gtimg,fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp',sprintf('%dto%d_paper.png',source.pid,ref.pid)));

subplot(211);imshowpair(gtimg.*mask,output_r.*mask,'montage');

om = output(1:size(gtimg,1),1:size(gtimg,2),:).*mask(1:size(gtimg,1),1:size(gtimg,2),:);
gm = gtimg.*mask(1:size(gtimg,1),1:size(gtimg,2),:);
om = imfilter(om,fspecial('gaussian',7),'symmetric');
p=fit(om(om>0.1),gm(om>0.1),'poly3');

output = reshape(feval(p,output),size(output));
output_r = reshape(feval(p,output_r),size(output));
subplot(212);imshowpair(gtimg.*mask,output_r.*mask,'montage');
imwrite(output_r.*mask,fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp',sprintf('%dto%d_ours_masked2.png',source.pid,ref.pid)));
imwrite(output_r,fullfile('C:\Users\qcri\Documents\Ajay\Face\SOA\sig_2017_mtp',sprintf('%dto%d_ours2.png',source.pid,ref.pid)));
