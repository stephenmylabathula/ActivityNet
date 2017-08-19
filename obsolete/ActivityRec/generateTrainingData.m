clear
clc

d1 = load('~/Projects/ActivityRec/datasets/20170708_1.mat');
d2 = load('~/Projects/ActivityRec/datasets/20170708_2.mat');
d3 = load('~/Projects/ActivityRec/datasets/20170708_3.mat');
d4 = load('~/Projects/ActivityRec/datasets/20170708_4.mat');
d5 = load('~/Projects/ActivityRec/datasets/20170708_5.mat');

d1_data = [d1.dataset.data.omega d1.dataset.data.accel d1.dataset.data.omega_norm d1.dataset.data.accel_norm d1.dataset.data.accel_global d1.dataset.data.omega_global d1.dataset.data.omega_global_norm d1.dataset.data.accel_global_norm d1.dataset.data.vio_quaternion d1.dataset.data.vio_rpy d1.dataset.data.velocity d1.dataset.data.position d1.dataset.tags];
d2_data = [d2.dataset.data.omega d2.dataset.data.accel d2.dataset.data.omega_norm d2.dataset.data.accel_norm d2.dataset.data.accel_global d2.dataset.data.omega_global d2.dataset.data.omega_global_norm d2.dataset.data.accel_global_norm d2.dataset.data.vio_quaternion d2.dataset.data.vio_rpy d2.dataset.data.velocity d2.dataset.data.position d2.dataset.tags];
d3_data = [d3.dataset.data.omega d3.dataset.data.accel d3.dataset.data.omega_norm d3.dataset.data.accel_norm d3.dataset.data.accel_global d3.dataset.data.omega_global d3.dataset.data.omega_global_norm d3.dataset.data.accel_global_norm d3.dataset.data.vio_quaternion d3.dataset.data.vio_rpy d3.dataset.data.velocity d3.dataset.data.position d3.dataset.tags];
d4_data = [d4.dataset.data.omega d4.dataset.data.accel d4.dataset.data.omega_norm d4.dataset.data.accel_norm d4.dataset.data.accel_global d4.dataset.data.omega_global d4.dataset.data.omega_global_norm d4.dataset.data.accel_global_norm d4.dataset.data.vio_quaternion d4.dataset.data.vio_rpy d4.dataset.data.velocity d4.dataset.data.position d4.dataset.tags];
d5_data = [d5.dataset.data.omega d5.dataset.data.accel d5.dataset.data.omega_norm d5.dataset.data.accel_norm d5.dataset.data.accel_global d5.dataset.data.omega_global d5.dataset.data.omega_global_norm d5.dataset.data.accel_global_norm d5.dataset.data.vio_quaternion d5.dataset.data.vio_rpy d5.dataset.data.velocity d5.dataset.data.position d5.dataset.tags];

train_data = [d1_data; d2_data; d3_data; d4_data; d5_data];
clear d1 d1_data d2 d2_data d3 d3_data d4 d4_data d5 d5_data