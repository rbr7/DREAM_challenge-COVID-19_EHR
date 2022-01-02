clear all; close all; clc
Teval = readtable('final_df_full_eval.csv');

Te = table2array(Teval);
Te(:,1) = [];

YTraineval = Te(:,end);
YTraineval= [abs(YTraineval-1) YTraineval];
Te(:,end) = [];
Tez = zscore(Te);

Yout = autoencoder0502Fcnvfinalv2(Tez');
Yout = Yout';
plotconfusion(YTraineval',Yout')

set(gca,'xticklabel',{'Non-hospitalized' 'Hospitalized' ''})
set(gca,'yticklabel',{'Non-hospitalized' 'Hospitalized' ''})

[X,Y,T,AUC] = perfcurve(YTraineval(:,2),Yout(:,2),1);
auroc = AUC;
disp('testing AUC')

results = Yout(:,2);
csvwrite('zdevalresult.csv',results)

figure,
plot(X,Y,'linewidth',2)
hold on
grid on
plot(0:0.01:1,0:0.01:1)
xlabel('False positive rate') 
ylabel('True positive rate')
title(['ROC for Autoencoder classication (AUC: ' num2str(AUC) ')']) 

figure
[Xpr,Ypr,Tpr,AUCpr] = perfcurve(YTraineval(:,2),Yout(:,2), 1, 'xCrit', 'reca', 'yCrit', 'prec'); 
plot(Xpr,Ypr,'linewidth',2) 
grid on
xlabel('Recall'); ylabel('Precision') 
title(['Precision-recall curve (AUC: ' num2str(AUCpr) ')']) 