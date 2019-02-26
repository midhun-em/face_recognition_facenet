import sys
from classifier import training

datadir = '../processed_data'
modeldir = '../models/20170511-185253.pb'
classifier_filename = '../models/emb_array.npy'
print ("Training Start")
obj=training(datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")
