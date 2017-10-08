clear all
clc

I = imread('./Assign1_imgs/zelda512-NoiseV400.jpg');

% Gaussian filter

% sigma = 100;
% ind = -floor(8/2) : floor(8/2);
% [X, Y] = meshgrid(ind, ind);
% h = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
% h = h / sum(h(:));
% 
% I_filtered = imfilter(I, h);
% 
% figure
% imshow(I_filtered)

% Median filter
% 
% fsize = 3;
% ind = floor(fsize / 2);
% tempimg = I;
% I = padarray(I, [ind ind]);
% 
% for i = 1 + ind: size(I, 1) - ind
%     for j = 1 + ind: size(I, 2) - ind
%         box = I(i - ind: i + ind, j - ind: j + ind, 1);
%         box = reshape(box, [1, size(box, 1) * size(box, 1)]);
%         medval = median(box);
%         tempimg(i, j, 1) = medval;
%         
%         box = I(i - ind: i + ind, j - ind: j + ind, 2);
%         box = reshape(box, [1, size(box, 1) * size(box, 1)]);
%         medval = median(box);
%         tempimg(i, j, 2) = medval;
%         
%         box = I(i - ind: i + ind, j - ind: j + ind, 3);
%         box = reshape(box, [1, size(box, 1) * size(box, 1)]);
%         medval = median(box);
%         tempimg(i, j, 3) = medval;
%     end
% end
% imshow(tempimg);
% 
% 
% Highboost filtering (uncomment Gaussian part)
% 
% cons = 3;
% I2 = I + cons .* (I - I_filtered);
% imshow(I2);
% 
% Bilateral filtering
% 
fsize = 3;
sigma = 2;
ind = floor(fsize / 2);

I = im2double(I);

I2 = padarray(I, [ind ind]);
dim = size(I2);
ht = dim(1);
wid = dim(2);

tempimg = I;

for i = 1 + ind: size(I2, 1) - ind
    for j = 1 + ind: size(I2, 2) - ind
        box = I2(i - ind: i + ind, j - ind: j + ind, :);

        [X, Y] = meshgrid(-floor(fsize/2) : floor(fsize/2), -floor(fsize/2) : floor(fsize/2));
        gf = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));

        db = box(:,:,3) - I2(i,j,3);
        da = box(:,:,2) - I2(i,j,2);
        dL = box(:,:,1) - I2(i,j,1);
        
        H = exp(-(dL.^2+da.^2+db.^2)/(2*sigma^2));

        temp(:, :, 1) = (box(:, :, 1) .* H) .* gf;
        temp(:, :, 2) = (box(:, :, 2) .* H) .* gf;
        temp(:, :, 3) = (box(:, :, 3) .* H) .* gf;
        
        temp(1) = sum(sum(temp(:, :, 1)));
        temp(2) = sum(sum(temp(:, :, 2)));
        temp(3) = sum(sum(temp(:, :, 3)));

        norm = gf .* H;
        norm = sum(sum(norm(:)));

        final(3) = temp(3) / norm;
        final(2) = temp(2)/ norm;
        final(1) = temp(1) / norm;
        
        tempimg(i, j, :) = final(:);

    end
end

imshow(tempimg);