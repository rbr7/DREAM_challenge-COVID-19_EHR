4 steps involved in autoencoder classification
1) data preprocessing from final_df_balanced_data.csv or final_df_full_eval.csv
Data format: 
1 column is patient ID (ID).
last column is hospitalization status (GS).
Measurement data is from sencond colum to end-1 (DFM).
knnimpute fill in missing value with knn method, and the measurements were normalized with Z-score normalization.

2) Autoencoder modeling: Use Autoencoder_Modeling.m
Autoencoder model are generated using optimized paramters and training of model parameters are iterated until high AUC ROC is reached for both CV and EV dataset.
The top autoencoder architecture is saved as "autoencoder0502Fcnvfinalv2.m"

3) Model evaluation of autoencoder model: Use Training_Set_Model_Eval.m and EV_Set_Model_Eval
Similar dataprocessing is used to generate patient feature data, MATLAB then called the autoencoder0502Fcnvfinalv2.m for hosptialization classification
The ground through and classfication generated result are compared with confusion materix, AUC ROC, AUC RF.

Additional .m file
A) Hidden_Layer_Selection.m is used for justification of number of hidden layer used, where AUC ROC are calculated with varying layer depth
B) autoencoder_alt.m is for deep learning network trial of autoencoder.