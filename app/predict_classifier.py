from magpie import Magpie
import os

folder = "magpie_data"
labf = open(folder+"askubuntu.labels", 'r')
labels = labf.read()
labels = labels.split('\n')
labels = [l for l in labels if len(l)>1 ]

print("loading model")
magpie = Magpie(keras_model=folder+'/model.h5',
		word2vec_model=folder+'/wordvec', 
		scaler=folder+'/scalervec',
		labels=labels)
#print(labels)


