import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import pickle
import csv
#Valence(-5)	Arousal(-4) 
def data(csv):
	modality1_data = []
	modality2_data = []
	modality3_data = []
	modality4_data = []
	label_data = []

	for i,line in enumerate(csv):
		if i >= 1:
			if int(float(line[-8])) in  [2,3,4,5]:
				modality1_data1 = []
				modality2_data1 = []
				modality3_data1 = []
				modality4_data1 = []
				for a in range(2,34,1):#modality3
					modality3_data1.append(float(line[a]))
				for b in range(34,38,1):#emg
					modality2_data1.append(float(line[b]))
				for c in range(38,42,1):#eog
					modality4_data1.append(float(line[c]))
				modality1_data1 = float(line[42])

				modality1_data.append(modality1_data1)
				modality2_data.append(modality2_data1)
				modality3_data.append(modality3_data1)
				modality4_data.append(modality4_data1)

				if 1<=round(float(line[-5])) <=3:
					label_data.append(-1)
				elif 4<=round(float(line[-5])) <=6:
					label_data.append(1)
				elif 7<=round(float(line[-5]))<=9:
					label_data.append(2)

	modality1_data2 = []
	modality2_data2 = []
	modality3_data2 = []
	label_data2 = []
	modality4_data2 = []

	for i in range(0,len(modality1_data),512):
		modality1_data2.append(modality1_data[i:i+512])
		modality2_data2.append(modality2_data[i:i+512])
		modality3_data2.append(modality3_data[i:i+512])
		modality4_data2.append(modality4_data[i:i+512])
		label_data2.append(label_data[i:i+512])

	modality1_data2.pop(-1)
	modality2_data2.pop(-1)
	modality3_data2.pop(-1)
	label_data2.pop(-1)
	modality4_data2.pop(-1)

	label_data3 = np.array(label_data2)
	label_data4 = []
	for i in range(label_data3.shape[0]): 
		label_data4.append(round(np.mean(label_data3[i])))

	csv_len = len(modality1_data2)
	
	return modality1_data2,modality2_data2,modality3_data2,modality4_data2,label_data4,csv_len





def pkl_make(modality11,modality21,modality31,modality41,label1,train_id,val_id,test_id,pkl,epoch):
	print('data over'+ str(epoch))

	modality1_train = np.array(modality11)[train_id].reshape(train_id.shape[0],1,512)
	modality1_val = np.array(modality11)[val_id].reshape(val_id.shape[0],1,512)
	modality1_test = np.array(modality11)[test_id].reshape(test_id.shape[0],1,512)

	modality2_train = np.array(modality21)[train_id].reshape(train_id.shape[0],4,512)
	modality2_val = np.array(modality21)[val_id].reshape(val_id.shape[0],4,512)
	modality2_test = np.array(modality21)[test_id].reshape(test_id.shape[0],4,512)

	modality3_train = np.array(modality31)[train_id].reshape(train_id.shape[0],32,512)
	modality3_val = np.array(modality31)[val_id].reshape(val_id.shape[0],32,512)
	modality3_test = np.array(modality31)[test_id].reshape(test_id.shape[0],32,512)

	modality4_train = np.array(modality41)[train_id].reshape(train_id.shape[0],4,512)
	modality4_val = np.array(modality41)[val_id].reshape(val_id.shape[0],4,512)
	modality4_test = np.array(modality41)[test_id].reshape(test_id.shape[0],4,512)

	id_train = np.arange(train_id.shape[0]).reshape(train_id.shape[0],1,1)
	id_val = np.arange(val_id.shape[0]).reshape(val_id.shape[0],1,1)
	id_test = np.arange(test_id.shape[0]).reshape(test_id.shape[0],1,1)

	label_train = np.array(label1)[train_id].reshape(train_id.shape[0],1,1)
	label_val = np.array(label1)[val_id].reshape(val_id.shape[0],1,1)
	label_test = np.array(label1)[test_id].reshape(test_id.shape[0],1,1)
	print('array over'+ str(epoch))
	print('reshape over'+ str(epoch))
	pkl1 = {}
	train = {}
	test = {}
	valid ={}

	train['id'] = id_train
	train['modality1'] = modality1_train
	train['modality2'] = modality2_train
	train['modality3'] = modality3_train
	train['modality4'] = modality4_train
	train['label'] = label_train

	valid['id'] = id_val
	valid['modality1'] = modality1_val
	valid['modality2'] = modality2_val
	valid['modality3'] = modality3_val
	valid['modality4'] = modality4_val
	valid['label'] = label_val

	test['id'] = id_test
	test['modality1'] = modality1_test
	test['modality2'] = modality2_test
	test['modality3'] = modality3_test
	test['modality4'] = modality4_test
	test['label'] = label_test

	pkl1['train'] = train
	pkl1['valid'] = valid
	pkl1['test'] = test

	pickle.dump(pkl1,pkl)
	print('done'+ str(epoch))
	return

def DEAP (array,lenth,modality11,modality21,modality31,modality41,label1):
	for i in range(10):
		train1 = []
		val_start = int(i*lenth/10)
		val_end = test_start = int((i+1)*lenth/10)
		test_end = int((i+2)*lenth/10)
		final_test = int(0.1*lenth)
		if i < 9:
			val = array[val_start:val_end]
			test = array[test_start:test_end]
		else:
			val = array[val_start:val_end]
			test = array[:final_test]
		for k in array:
			if k not in np.append(val,test):
				train1.append(k)
		train = np.array(train1)
		pkl1 = open(str(i)+'.pkl','wb')
		pkl_make(modality11,modality21,modality31,modality41,label1,train,val,test,pkl1,i)
	return 

	
if __name__ == '__main__':

	txt = open('Raw_DEAP_list.txt','r')
	txt1 = txt.readlines()

	modality11 = []
	modality21 = []
	modality31 = []
	modality41 = []
	label1 = []
	for i in txt1:
		k = i.rstrip('\n')
		print(k)
		csv1 = open(k,'r')
		csv2 = csv.reader(csv1)
		modality1_data,modality2_data,modality3_data,modality4_data,label_data,csv_len = data(csv2)
		modality11.extend(modality1_data)
		modality21.extend(modality2_data)
		modality31.extend(modality3_data)
		modality41.extend(modality4_data)
		label1.extend(label_data)

	print(len(modality31),len(modality11),len(label1))
	indices = np.arange(len(modality11))
	np.random.shuffle(indices)
	print('nasa')
	DEAP(indices,indices.shape[0],modality11,modality21,modality31,modality41,label1)




