outdir = 'C:\Users\qcri\Documents\Ajay\Face\SOA\facerelight\sig_17_mtp_results';
flist = dir(fullfile(outdir,'*_gt.png'));

data{1,1}='filename';
data{1,2}='l1_norm';
data{1,3}='l2_norm';
data{1,4}='ssim';

CROPPED=true;
if(CROPPED)
    fid = fopen(fullfile(outdir,'cropped_results.csv'),'w');
else
    fid = fopen(fullfile(outdir,'results.csv'),'w');
end
fprintf(fid,'filename,l1_norm,l2_norm,ssim\n');

probl_id=[];
for k=1:length(flist)
    fprintf('processing %d\n',k);
    out_fname = fullfile(outdir,[flist(k).name(1:end-6) 'output_r.png']);
    gt_fname = fullfile(outdir,[flist(k).name(1:end-6) 'gt.png']);
    [img_out,~,img_mask] =  imread(out_fname);
    img_out = im2double(img_out);
    img_gt =  im2double(imread(gt_fname));
    img_mask = im2double(img_mask);
    
    img_out = img_out.*repmat(img_mask,[1 1 3]);
    img_gt = img_gt.*repmat(img_mask,[1 1 3]);
    
    
%     if(size(img_out,1) ~=size(img_gt,1))
%         probl_id=[k, probl_id];
%         continue;
%     end
    if(CROPPED)
        % cropped
        [y,x]=find(img_mask);
        xmin = min(x); xmax = max(x);
        ymin = min(y); ymax = max(y);
        img_out = img_out(ymin:ymax,xmin:xmax,:);
        img_gt = img_gt(ymin:ymax,xmin:xmax,:);
        img_mask = img_mask(ymin:ymax,xmin:xmax);
        img_mask = img_mask>0;
        clear img_out2
        for c=1:size(img_gt,3)

            imgt = img_gt(:,:,c);
            h = imhist(imgt(img_mask));
            imgt = img_out(:,:,c)./max(max(img_out(:,:,c)));
            imgt(img_mask) = histeq(imgt(img_mask),h);
            img_out2(:,:,c)=imgt;

        end
    end
    
    ssim_val = ssim(rgb2gray(img_out2),rgb2gray(img_gt));
    l1_val = mean(abs(img_out2(:)-img_gt(:)));
    l2_val = mean((img_out2(:)-img_gt(:)).^2);
    data{k+1,1} = flist(k).name(1:end-6);
    data{k+1,2} = l1_val;
    data{k+1,3} = l2_val;
    data{k+1,4} = ssim_val;
    
    fprintf(fid,'%s,%f,%f,%f\n',flist(k).name(1:end-7),l1_val,l2_val,ssim_val);
end
fclose(fid);