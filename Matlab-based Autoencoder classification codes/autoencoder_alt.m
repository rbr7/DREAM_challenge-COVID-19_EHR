clear all; close all; clc
%Read the training set containing balanced weightclass
Tknn = readtable('final_df_balanced_data');
%Sort Rows based of status
Tknn = sortrows(Tknn,'status','descend');
%Convert Table to Array
T3 = table2array(Tknn);
%Remove patient ID as feature
T3(:,1) = [];

%Retrive classfication ground truth
YTrain = T3(:,end);
T3(:,end) = [];
Y0 = abs(YTrain-1);
Yout = [Y0 YTrain];

%Z-score normalize the feature data
T3balancedZ = zscore(T3);

% Data preparation for autoencoder
x = [T3balancedZ]';
t = [abs(YTrain-1) YTrain]';
size(x)
size(t)

autoenc1 = trainAutoencoder(x,20, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);


feat1 = encode(autoenc1,x);

softnet = trainSoftmaxLayer(feat1,t,'MaxEpochs',400);

stackednet = stack(autoenc1,softnet);
view(stackednet)
y = stackednet(x);
plotconfusion(Yout',y);
