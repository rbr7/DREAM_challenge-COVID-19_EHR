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

% Creating autoencoder
net = patternnet(20);
view(net)
auroc = 0;

% Data preparation for validation set
Teval = readtable('final_df_full_eval.csv');
Te = table2array(Teval);
Te(:,1) = [];
YTraineval = Te(:,end);
YTraineval= [abs(YTraineval-1) YTraineval];
Te(:,end) = [];
Tez = zscore(Te);


auctraintot = [];
aucevaltot = [];
AUC2 = 0;

%Autoencoder itertaion for model optimization
while auroc < 0.7 || AUC2 < 0.7
net = train(net,x,t);
Outputs = sim(net,x);
[X1,Y1,T1,AUC1] = perfcurve(t(2,:),Outputs(2,:),1);
disp('training AUC')
disp(AUC1)
auctraintot = [auctraintot AUC1];
genFunction(net,'autoencodertempfunc','MatrixOnly','yes');
Yout = autoencoder0502Fcnv3(Tez');
[X,Y,T,AUC] = perfcurve(YTraineval(:,2)',Yout(2,:),1);
auroc = AUC;
disp('testing AUC')
disp(auroc)
aucevaltot = [aucevaltot auroc];


Yout2 = autoencodertempfunc(T3balancedZ');
[X2,Y2,T2,AUC2] = perfcurve(t(2,:)',Yout2(2,:),1);

end

%Display of AUC ROC of CV versus EV
figure,
plot(auctraintot); hold on;
plot(aucevaltot);
grid on
legend('training','eval')

%Save final autoencoder model
genFunction(net,'autoencoderfinalfuc','MatrixOnly','yes');