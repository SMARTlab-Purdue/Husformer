import numpy as np
import pickle
import csv

def data(csv):
	modality1_data = []
	modality2_data = []
	modality3_data = []
	modality4_data = []
	modality5_data = []
	label_data = []
	
	for i,line in enumerate(csv):
		if i >= 1:
			modality1_data1 = []
			modality2_data1 = []
			modality3_data1 = []
			modality4_data1 = []
			modality5_data1 = []

			modality1_data1.append(line[54].strip('[').rstrip(']').split(', '))
			modality2 = line[52].strip('[').rstrip(']').replace("\n", '').split(', ')
			modality2_data2 = list(filter(None, modality2))
			modality2_data1.append(modality2_data2)
			for b in range(12,17,1):
				modality3_data1.append(line[b].strip('[').rstrip(']').split(', '))
			for a in range(19,44,1):
				modality4_data1.append(line[a].strip('[').rstrip(']').split(', '))
			modality5_data1.append(float(line[-5].strip('[').rstrip(']')))

			modality1_data.append(modality1_data1)
			modality2_data.append(modality2_data1)
			modality3_data.append(modality3_data1)
			modality4_data.append(modality4_data1)
			modality5_data.append(modality5_data1)
			if line[9] == "high":
				label_data.append(2)
			elif line[9] == "medium":
				label_data.append(1)
			elif line[9] == "low":
				label_data.append(-1)
	csv_len = len(modality1_data)
	return modality1_data,modality2_data,modality3_data,modality4_data,modality5_data,label_data,csv_len



def pkl_make(modality11,modality21,modality31,modality41,modality51,label1,train_id,val_id,test_id,pkl,epoch):
	print('data over'+ str(epoch))
	
	modality1_train = np.array(modality11)[train_id].reshape(train_id.shape[0],1,6)
	modality1_val = np.array(modality11)[val_id].reshape(val_id.shape[0],1,6)
	modality1_test = np.array(modality11)[test_id].reshape(test_id.shape[0],1,6)

	modality2_train = np.array(modality21)[train_id].reshape(train_id.shape[0],1,128)
	modality2_val = np.array(modality21)[val_id].reshape(val_id.shape[0],1,128)
	modality2_test = np.array(modality21)[test_id].reshape(test_id.shape[0],1,128)

	modality3_train = np.array(modality31)[train_id].reshape(train_id.shape[0],5,128)
	modality3_val = np.array(modality31)[val_id].reshape(val_id.shape[0],5,128)
	modality3_test = np.array(modality31)[test_id].reshape(test_id.shape[0],5,128)

	modality4_train = np.array(modality41)[train_id].reshape(train_id.shape[0],25,8)
	modality4_val = np.array(modality41)[val_id].reshape(val_id.shape[0],25,8)
	modality4_test = np.array(modality41)[test_id].reshape(test_id.shape[0],25,8)

	modality5_train = np.array(modality51)[train_id].reshape(train_id.shape[0],1,1)
	modality5_val = np.array(modality51)[val_id].reshape(val_id.shape[0],1,1)
	modality5_test = np.array(modality51)[test_id].reshape(test_id.shape[0],1,1)

	id_train = np.arange(train_id.shape[0]).reshape(train_id.shape[0],1,1)
	id_val = np.arange(val_id.shape[0]).reshape(val_id.shape[0],1,1)
	id_test = np.arange(test_id.shape[0]).reshape(test_id.shape[0],1,1)

	label_train = np.array(label1)[train_id].reshape(train_id.shape[0],1,1)
	label_val = np.array(label1)[val_id].reshape(val_id.shape[0],1,1)
	label_test = np.array(label1)[test_id].reshape(test_id.shape[0],1,1)
	print('array over'+ str(epoch))
	pkl1 = {}
	train = {}
	test = {}
	valid ={}

	train['id'] = id_train
	train['modality_1'] = modality1_train
	train['modality_2'] = modality2_train
	train['modality_3'] = modality3_train
	train['modality_4'] = modality4_train
	train['modality_5'] = modality5_train
	train['label'] = label_train
	
	valid['id'] = id_val
	valid['modality_1'] = modality1_val
	valid['modality_2'] = modality2_val
	valid['modality_3'] = modality3_val
	valid['modality_4'] = modality4_val
	valid['modality_5'] = modality5_val
	valid['label'] = label_val
	
	test['id'] = id_test
	test['modality_1'] = modality1_test
	test['modality_2'] = modality2_test
	test['modality_3'] = modality3_test
	test['modality_4'] = modality4_test
	test['modality_5'] = modality5_test
	test['label'] = label_test
	
	pkl1['train'] = train
	pkl1['valid'] = valid
	pkl1['test'] = test
	print('1')
	pickle.dump(pkl1,pkl)
	print('done'+ str(epoch))
	return

def MOCAS (array,lenth,modality11,modality21,modality31,modality41,modality51,label1):
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
		pkl_make(modality11,modality21,modality31,modality41,modality51,label1,train,val,test,pkl1,i)
	return 

	
if __name__ == '__main__':

	txt = open('Raw_MOCAS_list.txt','r')
	txt1 = txt.readlines()

	modality11 = []
	modality21 = []
	modality31 = []
	modality41 = []
	modality51 = []
	label1 = []

	for i in txt1:
		k = i.rstrip('\n')
		print(k)
		csv1 = open(k,'r')
		csv2 = csv.reader(csv1)
		modality1_data,modality2_data,modality3_data,modality4_data,modality5_data,label_data,csv_len = data(csv2)
		modality11.extend(modality1_data)
		modality21.extend(modality2_data)
		modality31.extend(modality3_data)
		modality41.extend(modality4_data)
		modality51.extend(modality5_data)
		label1.extend(label_data)
		csv1.close()
	indices = np.arange(len(modality11))
	np.random.shuffle(indices)

	MOCAS(indices,indices.shape[0],modality11,modality21,modality31,modality41,modality51,label1)




