clear
clc
load ~/Projects/ActivityRec/experiments/2017_07_11/collection1/dataset.mat;

start_cut = 3100;
end_cut = 13099;
% end_cut - start_cut = 11999

dataset.data.time = dataset.data.time(1:end_cut-start_cut+1,:);
dataset.data.omega = dataset.data.omega(start_cut:end_cut,:);
dataset.data.accel = dataset.data.accel(start_cut:end_cut,:);
dataset.data.omega_norm = dataset.data.omega_norm(start_cut:end_cut,:);
dataset.data.accel_norm = dataset.data.accel_norm(start_cut:end_cut,:);
dataset.data.accel_global = dataset.data.accel_global(start_cut:end_cut,:);
dataset.data.omega_global = dataset.data.omega_global(start_cut:end_cut,:);
dataset.data.omega_global_norm = dataset.data.omega_global_norm(start_cut:end_cut,:);
dataset.data.accel_global_norm = dataset.data.accel_global_norm(start_cut:end_cut,:);
dataset.data.vio_quaternion = dataset.data.vio_quaternion(start_cut:end_cut,:);
dataset.data.vio_rpy = dataset.data.vio_rpy(start_cut:end_cut,:);
dataset.data.omega_bias = dataset.data.omega_bias(start_cut:end_cut,:);
dataset.data.velocity = dataset.data.velocity(start_cut:end_cut,:);
dataset.data.accel_bias = dataset.data.accel_bias(start_cut:end_cut,:);
dataset.data.position = dataset.data.position(start_cut:end_cut,:);


% Jump = 4
% Move = 3
% Sit = 2
% Stand = 1
% Walk = 0
dataset.tags = [repelem(4, 2000)'; repelem(3, 2000)'; repelem(2, 2000)'; repelem(3, 2000)'; repelem(4, 2000)'];

save('~/Projects/ActivityRec/experiments/2017_07_11/collection1/dataset1.mat', 'dataset'); clear
