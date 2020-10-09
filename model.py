import numpy as np
import re
import pathlib
import random
import os
import pickle
import sys

from random import shuffle
from collections import defaultdict, Counter

from keras.preprocessing.text import Tokenizer
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, MaxPooling1D, AveragePooling1D, Bidirectional
from keras.layers import Flatten, concatenate
from keras.layers import Embedding
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.metrics import f1_score, classification_report,precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from utils import make_data,make_graph,entities_2_graph,prepare_eval,eval,entities_2_directed_graph,data_for_error_analysis,eval_2class
from read_corpus import read_data

np.set_printoptions(threshold=np.inf,linewidth=50000)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS'] = "device=cuda*"

setting = sys.argv[1] # eg. 8class,5class,2class
indicators = sys.argv[2] # eg. mrole
epochs = int(sys.argv[3]) # eg. 20
case = sys.argv[4] # directed / undirected
window_size = int(sys.argv[5])

test = False
outputs = 0
dimensions = 300
batch_size = 20
embeddings_preloaded = False

lb = MultiLabelBinarizer()
if case == 'directed':
	if setting == 'basic':
		lb.fit(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
	elif setting == 'mixed':
		lb.fit(['0','1','2','3','4','5','6','7','8','9'])
	else:
		lb.fit(['0','1','2','3'])
else:
	if setting == 'basic':
		lb.fit(['0','1','2','3','4','5','6','7'])
	elif setting == 'mixed':
		lb.fit(['0','1','2','3','4'])
	else:
		lb.fit(['0','1'])


total_precision = []
total_recall = []
total_f1 = []

graph_total_precision = []
graph_total_recall = []
graph_total_f1 = []

grand_pred = []
grand_gold = []

files = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

for number in files:
	training_file = 0
	val_data = 0
	data = read_data(file='fanfic-corpus.txt',fold=number,window_size=window_size,experiment=setting,direction=case,test=test)
	train = data[0]
	train_y = data[1]
	val = data[2]
	val_y = data[3]

	X = []
	y = []

	data_train = train
	make_data(data_train,train_y,X,y,indicators)
	split = len(X)

	data_val = val
	make_data(data_val,val_y,X,y,indicators)

	y_nontransform = y
	outputs = []
	for i in y_nontransform:
		for e in i:
			outputs.append(e)
	y = lb.fit_transform(y)
	t = Tokenizer(split=" ",lower=True, filters='@')
	t.fit_on_texts(X)
	encoded_docs = t.texts_to_sequences(X)
	vocab_size = len(t.word_index) + 1

	max_len = []
	for each in encoded_docs:
		max_len.append(len(each))
	pad2 = max(max_len)
	padded_docs = pad_sequences(encoded_docs, maxlen=pad2, padding='post')
	embeddings_index = dict()

	if embeddings_preloaded:
		embedding_matrix = pickle.load(open('embeddings/embeddings_%s_%s_%s_%.pkl' %(str(number),setting,case,str(window_size)),'rb'))
	else:
		f = open('embeddings/glove.6B.300d.txt', encoding='utf-8')
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		print('Loaded %s word vectors.' % len(embeddings_index))
		embedding_matrix = np.zeros((vocab_size, dimensions))
		for word, i in t.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
			else:
				vec = [0]*dimensions
				embedding_matrix[i] = np.array(vec)
		pickle.dump(embedding_matrix,open('embeddings/embeddings_%s_%s_%s_%s.pkl' %(str(number),setting,case,str(window_size)),'wb'))

	embedding_matrix = pickle.load(open('embeddings/embeddings_%s_%s_%s_%s.pkl' %(str(number),setting,case,str(window_size)),'rb'))

	if not test:
		model = Sequential()
		model.add(Embedding(vocab_size, dimensions,weights=[embedding_matrix], input_length=pad2, trainable=True))
		model.add(GRU(128, return_sequences=True))
		model.add(MaxPooling1D())
		model.add(AveragePooling1D())
		model.add(GRU(128))
		model.add(Dense(len(set(outputs)), activation='sigmoid'))

		checkpoint = ModelCheckpoint('model-%s-%s-%s-%s' %(setting,indicators,case,str(window_size)), verbose=1, monitor='val_acc',save_best_only=True, mode='auto')
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		print(model.summary())

		train = padded_docs[:split]
		dev = padded_docs[split:]
		y_train = y[:split]
		y_dev = y[split:]

		model.fit(train, y_train, epochs=epochs, verbose=1, callbacks=[checkpoint], batch_size=batch_size,validation_data=(dev,y_dev))
		pred = model.predict(dev, batch_size=batch_size)
	else:
		dev = padded_docs[split:]
		y_dev = y[split:]
		model = load_model('model-%s-%s-%s-%s' %(setting,indicators,case,str(window_size)))
		print('Model loaded')
		pred = model.predict(dev, batch_size=batch_size)

	pred_x = []
	for p in pred:
		p[p < max(p)] = 0
		p[p==max(p)] = 1
		pred_x.append(list(p).index(max(list(p))))

	dev_x = []
	for d in y_dev:
		dev_x.append(list(d).index(max(list(d))))

	for i,j,t in zip(dev_x,pred_x,X[split:]):
		grand_gold.append(i)
		grand_pred.append(j)

	p = precision_recall_fscore_support(dev_x,pred_x,average='micro')[0]
	r = precision_recall_fscore_support(dev_x,pred_x,average='micro')[1]
	f = precision_recall_fscore_support(dev_x,pred_x,average='micro')[2]
	total_precision.append(p)
	total_recall.append(r)
	total_f1.append(f)
	print("intermediate metrics: ", p,r,f)

	X = []
	y = []
	X_pred = []
	y_pred = []
	graph = defaultdict(dict)
	graph2 = defaultdict(dict)

	data_val = val
	make_graph(data_val,val_y,X, y, pred_x, X_pred, y_pred)
	characters = entities_2_graph(X, y)
	characters2 = entities_2_graph(X, y_pred)

	graph = prepare_eval(characters,y,graph,param='gold')
	graph2 = prepare_eval(characters2,y_pred,graph2,param='pred')

	if setting!="2class":
		eval(graph,graph2,graph_total_precision,graph_total_recall,graph_total_f1)
	else:
		eval_2class(graph,graph2,graph_total_precision,graph_total_recall,graph_total_f1)

print(setting,'indicators:',indicators,'window_size:',window_size)
print(np.mean(total_precision),np.mean(total_recall),np.mean(total_f1))
print(np.mean(graph_total_precision),np.mean(graph_total_recall),np.mean(graph_total_f1))

totp = str(round(np.mean(total_precision),2))
totr = str(round(np.mean(total_recall),2))
totf = str(round(np.mean(total_f1),2))

gp = str(round(np.mean(graph_total_precision),2))
gr = str(round(np.mean(graph_total_recall),2))
gf = str(round(np.mean(graph_total_f1),2))

grand_precision_micro = precision_recall_fscore_support(grand_gold,grand_pred,average='micro')[0]
grand_recall_micro = precision_recall_fscore_support(grand_gold,grand_pred,average='micro')[1]
grand_f1_micro = precision_recall_fscore_support(grand_gold,grand_pred,average='micro')[2]

grand_precision_macro = precision_recall_fscore_support(grand_gold,grand_pred,average='macro')[0]
grand_recall_macro = precision_recall_fscore_support(grand_gold,grand_pred,average='macro')[1]
grand_f1_macro = precision_recall_fscore_support(grand_gold,grand_pred,average='macro')[2]

report = classification_report(grand_gold,grand_pred)
print(totp, totr, totf, gp, gr, gf, grand_precision_micro, grand_recall_micro, grand_f1_micro, grand_precision_macro, grand_recall_macro, grand_f1_macro, report)
