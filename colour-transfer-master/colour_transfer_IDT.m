%
%   colour transfer algorithm based on N-Dimensional PDF Transfer 
%
%   IR = colour_transfer_IDT(I_original, I_target, nb_iterations);
%
%  (c) F. Pitie 2007
%
%  see reference:
%     Automated colour grading using colour distribution transfer. (2007) 
%     Computer Vision and Image Understanding.
%
%  To remove the "grainyness" on the results, you should apply the grain 
%  reducer proposed in the paper and implemented in regrain.m:
%
%  IRR = regrain(I_original, IR);
%
function [IR,IR_full] = colour_transfer_IDT(I0, I1, nb_iterations,I0_full)
if nargin < 4
    I0_full = I0;
end

if (ndims(I0)~=3)
    error('pictures must have 3 dimensions');
end

nb_channels = size(I0,3);

%% reshape images as 3xN matrices
for i=1:nb_channels
    D0(i,:) = reshape(I0(:,:,i), 1, size(I0,1)*size(I0,2));
    D1(i,:) = reshape(I1(:,:,i), 1, size(I1,1)*size(I1,2));
    D0_full(i,:) = reshape(I0_full(:,:,i), 1, size(I0_full,1)*size(I0_full,2));
end

%% building a sequence of (almost) random projections
% 

R{1} = [1 0 0; 0 1 0; 0 0 1; 2/3 2/3 -1/3; 2/3 -1/3 2/3; -1/3 2/3 2/3];
for i=2:nb_iterations
      R{i} = R{1} * orth(randn(3,3));
end

%for i=2:nb_iterations, R{i} = R{1}; end
%% pdf transfer
[DR,DR_full] = pdf_transfer2(D0, D1, R, 1,D0_full);

%% reshape the resulting 3xN matrix as an image
IR = I0;
IR_full = I0_full;
for i=1:nb_channels
    IR(:,:,i) = reshape(DR(i,:), size(IR, 1), size(IR, 2));
    IR_full(:,:,i) = reshape(DR_full(i,:), size(IR_full, 1), size(IR_full, 2));
end
