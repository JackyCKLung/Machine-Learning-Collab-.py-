Environment, using 3050ti 4G to speed up the training process

File structure, 5.Graph_Transformer_Netowrk is where the code of GTN put at. GTN folder is under the root. also, data folder is under the root. Dataset need to be path to the data/data/METR-LA 

V4_MorePara.py is the file with a higher parameters, it may take up to 30 mins to train and plot the graph.

V4_WithMAERMSE.py is the file to find MAE and RMSE of node 0 1 2. 


V3_withGraph.py is the file with a lower parameters, it may take around 15 mins to train and plot the graph. 

V3_withMAE.py is the file with the same parameters as all named as V3, it generate the mean absolute error and also the maximum absolute error. 

hi.py can test whether you can run the code with pytorch and cuda, cuda require nvidia gpu.
