function out_img = sig17mtp(source_img,ref_img)
% INPUT:
% source_img : Input Img to change lighting Condition, 3-channel double
% image
% ref_img : target lighting Condition Image, 3-channel double image
% OUTPUT:
% out_img : relighted Image, 3-channel double image


% Hyper Parameters
Niter=100;
wn= 1;wp= 1;wc = 1;
scale = 2;
addpath('../3DMM_edges-master');

% rescale
source.ori_img = source_img;

source_img = imresize(source_img,1./scale);
ref_img = imresize(ref_img,1./scale);
source.diff = source.ori_img - imresize(source_img,size(source.ori_img(:,:,1)));

source.img = reshape(source_img,[],3);
ref.img = reshape(ref_img,[],3);

% Find Face Normals & 2D positions
[source.normal,source.mask,source.posMap]=img2facedata(source_img);
[ref.normal,ref.mask,ref.posMap]=img2facedata(ref_img);

source.normal = reshape(source.normal,[],3);
source.mask = reshape(source.mask(:,:,1),[],1);
source.posMap = reshape(source.posMap,[],3);

ref.normal = reshape(ref.normal,[],3);
ref.mask = reshape(ref.mask(:,:,1),[],1);
ref.posMap = reshape(ref.posMap,[],3);

source.s = [wc*source.img(source.mask,:) source.normal(source.mask,:)*wn source.posMap(source.mask,1:2)*wp];
ref.s = [wc*ref.img(ref.mask,:) ref.normal(ref.mask,:)*wn ref.posMap(ref.mask,1:2)*wp];


% Color-Transfer 
Ndim = size(source.s,2);
Si = source.s;
Ls = size(Si,1);
Sr = ref.s;

sigma=0.1;
Si_extra = Si;
for i=1:5
    Si_extra(i*Ls+(1:Ls),:) =Si;
    Si_extra(i*Ls+(1:Ls),1:3)=Si(:,1:3)+randn(Ls,3)*sigma;
end

R{1}=eye(Ndim);
for kiter=2:Niter
    R{kiter} = R{1}*orth(randn(Ndim));
end

So = pdf_transfer(Si_extra', Sr', R, 0.2);
So = So';

output = source.img;
output(source.mask,:)=So(1:Ls,1:3);
output = reshape(output,size(source_img));
output_r = regrain(source_img,output,[0.125]);
% output_r = output;
out_img = imresize(output_r,size(source.diff(:,:,1))) + source.diff; 


