%% Data Tuning (normalization)
DF = readtable("mydata4.csv");
ID = DF(:,1);
ID=ID{:,:};
GS = DF(:,end);
GS=GS{:,:};
DFM = DF(:,2:end-1);
S = vartype('numeric');
DFM=DFM{:,S};


DFMWN = knnimpute(DFM);
DFMWN = normc(DFMWN);

% PCA
%X = table2array(sTable);
[coeff, score, ~, ~, explained] = pca(DFMWN,'centered',false);
hold on
bar(explained)
yyaxis right
h = gca;
h.YAxis(2).Limits = [0 100];
h.YAxis(2).Color = h.YAxis(1).Color;
h.YAxis(2).TickLabel = strcat(h.YAxis(2).TickLabel, '%');
id = find(cumsum(explained)>90,1);
scoreTrain90 = score(:,1:id);
% % PCA
% %X = table2array(sTable);
% [coeff, score, ~, ~, explained] = pca(DFAWN, 'centered', false);
% hold on
% bar(explained)
% yyaxis right
% h = gca;
% h.YAxis(2).Limits = [0 100];
% h.YAxis(2).Color = h.YAxis(1).Color;
% h.YAxis(2).TickLabel = strcat(h.YAxis(2).TickLabel, '%');
% id = find(cumsum(explained)>90,1);
% scoreTrain90 = score(:,1:id);
scoreTrain90wC = [scoreTrain90, GS];
%% K-mean clustering and t-SNE plot
 Y = tsne(scoreTrain90wC);
[IDX,C,SUMD,K]=kmeans_opt(scoreTrain90);
figure
[clusters, centroid] = kmeans(scoreTrain90 , K);
 gscatter(Y(:,1),Y(:,2),clusters)
 figure
 gscatter(Y(:,1),Y(:,2),GS)

Labels = cellstr(string(titlesorted)');
bar(sortedcoeff)
set(gca, 'XTickLabel',Labels, 'XTick',1:numel(Labels))

M = [clusters, GS, ID]
[sortedM,idx] = sort(M(:,1));
sortedM = M(idx,:);
%% Evaluation and justification of K silhoutte. 
E = evalclusters(scoreTrain90,'kmeans','silhouette','klist',[1:12])
plot(E)% used to see wheather the data set is suited for clustering and which value for K is suited best. The higher the silhouette value the better the fit. 


