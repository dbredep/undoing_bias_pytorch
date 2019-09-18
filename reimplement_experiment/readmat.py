from scipy.io import loadmat

dic = loadmat('./features/Caltech101.mat')
print (dic.keys())
print (dic['train_labels'][1][0])
print (dic['train_labels'][1][0].shape)