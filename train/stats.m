clear all; close all; clc;
data_dir = './colon';
a = dir([data_dir, '/*.bmp']);
a = a(1:end-1);

means = 0;
stds = 0;
for i = 1 : length(a)
    img = mat2gray(imread(fullfile(data_dir, a(i).name)));
    means = means + sum(sum(img))/(1024*1024);
    disp(i);
end
means = means / length(a);

vars = 0;
for i = 1 : length(a)
    img = mat2gray(imread(fullfile(data_dir, a(i).name)));
    vars = vars + sum(sum((img - means).^2));
    disp(i);
end
vars = vars / (1024*1024*length(a));
