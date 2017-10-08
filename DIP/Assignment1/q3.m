I = imread('Assign1_imgs/blur1.jpg');

h = size(I, 1);
w = size(I, 2);
newI = zeros(size(I));

% Ripple
 
for x = 1:size(newI, 1)
    for y = 1:size(newI, 2)
        tx = x + 10 * sin((2 * pi * y) / 120);
        ty = y + 15 * sin((2 * pi * x) / 150);
        ox = min(max(round(tx), 1), h);
        oy = min(max(round(ty), 1), w);
        newI(x, y, 1) = I(ox, oy, 1);
        newI(x, y, 2) = I(ox, oy, 2);
        newI(x, y, 3) = I(ox, oy, 3);
    end
end

imshow(uint8(newI));


% Sphere

rmax = min(size(I,1),size(I,2)) * 0.75;
rho = 2;
xc = round(size(I,1)/2);
yc = round(size(I,2)/2);

    
for x = 1:size(zeros(size(I)), 1)
    for y = 1:size(zeros(size(I)), 2)
        dx = x - xc;
        dy = y - yc;
        r = sqrt(dx^2 + dy^2);
        z = sqrt(rmax^2 - r^2);

        if r > rmax
            tx = x;
            ty = y;
        else
            tx = x - (z * tan((1 - (1/rho)) * asin(dx/sqrt(dx^2 + z^2))));
            ty = y - (z * tan((1 - (1/rho)) * asin(dy/sqrt(dy^2 + z^2))));
        end

        ox = min(max(round(tx), 1), h);
        oy = min(max(round(ty), 1), w);

        newI(x, y, 1) = I(ox, oy, 1);
        newI(x, y, 2) = I(ox, oy, 2);
        newI(x, y, 3) = I(ox, oy, 3);
    end
end

imshow(uint8(newI));