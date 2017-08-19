clc

d1 = load('~/Projects/ActivityRec/datasets/20170708_6.mat');

d1_data = [d1.dataset.data.omega d1.dataset.data.accel d1.dataset.data.omega_norm d1.dataset.data.accel_norm d1.dataset.data.accel_global d1.dataset.data.omega_global d1.dataset.data.omega_global_norm d1.dataset.data.accel_global_norm d1.dataset.data.vio_quaternion d1.dataset.data.vio_rpy d1.dataset.data.velocity d1.dataset.data.position];
test_data = d1_data;

%yfit = trainedModel.predictFcn(test_data);
figure;
plot(d1.dataset.data.accel_global(:,3));
figure;
spec = spectrogram(d1.dataset.data.accel_global(:,3),50,10,[],1000,'yaxis');
[row,col,v]= find(abs(spec)>-1);
my_image = zeros(size(spec));
for i=1:length(row)
    my_image(row(i),col(i)) = 100;
end
figure;
imshow(spec);
figure;
imshow(my_image);

% figure; 
% plot(yfit, '-r');
% hold on;
% plot(d1.dataset.tags, '--b');
% plot(d1.dataset.data.accel);

clear d1_data