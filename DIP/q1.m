im1 = rgb2gray(imread('./Assign1_imgs/hist_equal.jpg'));
im2 = rgb2gray(imread('./Assign1_imgs/monkey.jpg'));

% part a

M = zeros(256,1,'uint8');
cdf1 = cumsum(imhist(im1)) / numel(im1);
cdf2 = cumsum(imhist(im2)) / numel(im2);

for idx = 1 : 256
    [~,ind] = min(abs(cdf1(idx) - cdf2));
    M(idx) = ind-1;
end
out = M(double(im1)+1);
imshow(out);


% part b

% W = [201 201];
% Wp = [201 201]/2;
% Wp = floor(Wp);
% impadded = padarray(im1, Wp);
% imn = im1;
% 
% for i=Wp(1) + 1 : size(im1, 1) + Wp(1)
%     for j=Wp(2) + 1 : size(im1, 2) + Wp(2)
%         counts = imhist(impadded(i - Wp(1) : i + Wp(1), j - Wp(2) : j + Wp(2)));
%         counts2 = cumsum(counts);
%         Wm = floor([201 201] / 2) + 1;
%         window = impadded(i - Wp(1) : i + Wp(1), j - Wp(2) : j + Wp(2));
%         imn(i - Wp(1), j - Wp(2)) = counts2(window(Wm(1), Wm(2)) + 1) / (W(1) * W(2)) * 255;
%     end
% end
% 
% imshow(imn);