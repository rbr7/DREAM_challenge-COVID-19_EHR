clear all; close all; clc

% Data preparation for autoencoder
Tknn = readtable('final_df_balanced_data');
Tknn = sortrows(Tknn,'status','descend');
T3 = table2array(Tknn);
T3(:,1) = [];
YTrain = T3(:,end);
T3(:,end) = [];
Y0 = abs(YTrain-1);
Yout = [Y0 YTrain];
T3balancedZ = zscore(T3);
x = [T3balancedZ]';
t = [abs(YTrain-1) YTrain]';
t0 = t;

size(x)
size(t)


avgauctot = [];
for i = 1:50
    disp(i)
    aucs = 0;
    youtts = 0;
    for j = 1:10
net = patternnet(i);
net = train(net,x,t);
Outputs = sim(net,x);
genFunction(net,'autoencodertemptesting','MatrixOnly','yes');
[X1,Y1,T1,AUC1] = perfcurve(t(2,:),Outputs(2,:),1);
aucs = aucs + AUC1;
clear net
    end
    avgauctot = [avgauctot aucs];
    
end
avgauctot = avgauctot/10;



avgauctotsmooth = smooth(avgauctot);
xs = 1:50;
figure
plot(xs,avgauctotsmooth,'LineWidth',2)
grid on
xlabel('# of Hidden Layers');
ylabel('Average AUC ROC');
title('AUC ROC versus # of Hidden Layers');

