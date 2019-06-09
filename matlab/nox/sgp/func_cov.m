function [ K_x1_x2 ] = func_cov( x1, x2, COV )
%FUNC_COV Summary of this function goes here
%   Detailed explanation goes here

D = datenum(x1) - datenum(x2)';

K_x1_x2 = zeros(size(D));

for col = 1:size(D, 2)
    K_x1_x2(:, col) = COV((ceil(length(COV)/2) + round(D(:, col)*24)));
end

end

