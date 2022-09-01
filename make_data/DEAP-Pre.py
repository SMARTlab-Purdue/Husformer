import numpy as np
import pickle
import csv
import scipy.io
# label : valence, arousal, dominance, liking
def data(mat_data1,mat_label1):
	modality1_data = []
	modality2_data = []
	modality3_data = []
	modality4_data = []

	modality1_data1 = []
	modality2_data1 = []
	modality3_data1 = []
	modality4_data1 = []
	label_data = []

	for k in range(mat_data1.shape[0]):#8064
		modality1_data1.append(mat_data1[k][:32]) 
		modality2_data1.append(mat_data1[k][32:34])
		modality3_data1.append(mat_data1[k][34:36])
		modality4_data1.append(mat_data1[k][36])
	for x in range(0,len(modality3_data1),128):#63
		modality1_data.append(modality3_data1[i:i+128])
		modality2_data.append(modality2_data1[i:i+128])
		modality3_data.append(modality4_data1[i:i+128])
		modality4_data.append(modality1_data1[i:i+128])

		if 1<=round(mat_label1[0])<=3:
			label_data.append(-1)
		elif 4<=round(mat_label1[0])<=6:
			label_data.append(1)
		elif 7<=round(mat_label1[0])<=9:
			label_data.append(2)

	data_len = len(modality3_data)
	return modality1_data,modality2_data,modality3_data,modality4_data,label_data,data_len




def pkl_make(modality1,modality2,modality3,modality4,label,train_id,val_id,test_id,pkl,epoch):
	print('data over'+ str(epoch))

	modality1_train = np.array(modality1)[train_id].reshape(train_id.shape[0],32,128)
	modality1_val = np.array(modality1)[val_id].reshape(val_id.shape[0],32,128)
	modality1_test = np.array(modality1)[test_id].reshape(test_id.shape[0],32,128)

	modality2_train = np.array(modality2)[train_id].reshape(train_id.shape[0],2,128)
	modality2_val = np.array(modality2)[val_id].reshape(val_id.shape[0],2,128)
	modality2_test = np.array(modality2)[test_id].reshape(test_id.shape[0],2,128)

	modality3_train = np.array(modality3)[train_id].reshape(train_id.shape[0],2,128)
	modality3_val = np.array(modality3)[val_id].reshape(val_id.shape[0],2,128)
	modality3_test = np.array(modality3)[test_id].reshape(test_id.shape[0],2,128)

	modality4_train =  np.array(modality4)[train_id].reshape(train_id.shape[0],1,128)
	modality4_val = np.array(modality4)[val_id].reshape(val_id.shape[0],1,128)
	modality4_test = np.array(modality4)[test_id].reshape(test_id.shape[0],1,128)

	id_train = np.arange(train_id.shape[0]).reshape(train_id.shape[0],1,1)
	id_val = np.arange(val_id.shape[0]).reshape(val_id.shape[0],1,1)
	id_test = np.arange(test_id.shape[0]).reshape(test_id.shape[0],1,1)

	label_train = np.array(label)[train_id].reshape(train_id.shape[0],1,1)
	label_val = np.array(label)[val_id].reshape(val_id.shape[0],1,1)
	label_test = np.array(label)[test_id].reshape(test_id.shape[0],1,1)

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

	txt1 = open('Preprocessed_deap_lisr.txt','r').readlines()

	modality11 = []
	modality21 = []
	modality31 = []
	modality41 = []
	label1 = []
	#modality2->modality2,modality4->modality4
	for i,line in enumerate(txt1):#32个mat
		mat_path = line.rstrip('\n')
		print(mat_path)
		mat_cont = scipy.io.loadmat(mat_path)
		mat_data = np.transpose(mat_cont['data'], (0, 2, 1))#40*8064*40
		mat_label = mat_cont['labels']#40*4
		for j in range(mat_data.shape[0]):
			mat_data1 = mat_data[j]#8064*40
			mat_label1 = mat_label[j]#4
			modality1_data,modality2_data,modality3_data,modality4_data,label_data,data_len = data(mat_data1,mat_label1)
			modality11.extend(modality1_data)
			modality21.extend(modality2_data)
			modality31.extend(modality3_data)
			modality41.extend(modality4_data)
			label1.extend(label_data)

	indices = np.arange(len(modality11))
	np.random.shuffle(indices)
	DEAP(indices,indices.shape[0],modality11,modality21,modality31,modality41,label1)
