from collections import defaultdict
from collections import Counter

def role_indicator(word_list):
	for i, w in enumerate(word_list):
		if w.startswith("TARG"):
			w = w.split("-")
			w1 = '<target>'
			w2 = w[1]
			w3 = '</target>'
			try:
				word_list[i] = w1
				word_list[i+1] = w2
				word_list[i+2] = w3
			except IndexError:
				word_list.insert(i+2,w3)
		elif w.startswith("EXP"):
			w = w.split("-")
			w1 = '<exp>'
			w2 = w[1]
			w3 = '</exp>'
			try:
				word_list[i] = w1
				word_list[i+1] = w2
				word_list[i+2] = w3
			except IndexError:
				word_list.insert(i+2,w3)
	return word_list

def insert_entity_indicator(word_list):
	for i, w in enumerate(word_list):
		if w.startswith("TARG"):
			w = w.split("-")
			w1 = '<entity-obj>'
			w2 = w[1]
			w3 = '</entity-obj>'
			try:
				word_list[i] = w1
				word_list[i+1] = w2
				word_list[i+2] = w3
			except IndexError:
				word_list.insert(i+2,w3)
		elif w.startswith("EXP"):
			w = w.split("-")
			w1 = '<entity-obj>'
			w2 = w[1]
			w3 = '</entity-obj>'
			try:
				word_list[i] = w1
				word_list[i+1] = w2
				word_list[i+2] = w3
			except IndexError:
				word_list.insert(i+2,w3)
	return word_list

def remove_indicators(word_list):
	for i, w in enumerate(word_list):
		if "TARG" in w:
			word_list[i] = w.replace('TARG','').replace('-','')
		elif "EXP" in w:
			word_list[i] = w.replace('EXP','').replace('-','')
	return word_list

def remove_indicators_with_position(word_list):
	subj = int()
	obj = int()
	for i, w in enumerate(word_list):
		if "TARG" in w:
			word_list[i] = w.replace('TARG','').replace('-','')
			obj = i
		elif "EXP" in w:
			word_list[i] = w.replace('EXP','').replace('-','')
			subj = i
	return word_list,subj,obj

def insert_mrole(word_list):
	for i, w in enumerate(word_list):
		if w.startswith("TARG"):
			word_list[i] = 'target_object'
		elif w.startswith("EXP"):
			word_list[i] = 'experiencer_object'
	return word_list

def insert_mEntity(word_list):
	for i, w in enumerate(word_list):
		if w.startswith("TARG"):
			word_list[i] = 'entity-object'
		elif w.startswith("EXP"):
			word_list[i] = 'entity-object'
	return word_list

def data_for_error_analysis(x,X):
	data_points = defaultdict(set)
	for line in x:
		line = line.split("\t")
		label = line[0]
		string = line[1]
		if not string in data_points:
			data_points[string] = set(label)
		else:
			data_points[string].add(label)
	for k,v in data_points.items():
		X.append(k)

def make_data(x,labels,X,y,indicators):
	data_points = defaultdict(set)
	for line,label in zip(x,labels):
		# line = line.split("\t")
		if indicators == "role":
			word_list = [w.lower() for w in role_indicator(line.split())]
			string = " ".join(word_list)
			label = label
			if not string in data_points:
				data_points[string] = set(label)
			else:
				data_points[string].add(label)
		elif indicators == "no-ind":
			word_list = [w.lower() for w in remove_indicators(line.split())]
			string = " ".join(word_list)
			label = label
			if not string in data_points:
				data_points[string] = set(label)
			else:
				data_points[string].add(label)
		elif indicators == "mrole":
			word_list = [w.lower() for w in insert_mrole(line.split())]
			string = " ".join(word_list)
			label = label
			if not string in data_points:
				data_points[string] = set(label)
			else:
				data_points[string].add(label)
		elif indicators == "mentity":
			word_list = [w.lower() for w in insert_mEntity(line.split())]
			string = " ".join(word_list)
			label = label
			if not string in data_points:
				data_points[string] = set(label)
			else:
				data_points[string].add(label)
		elif indicators == "entity":
			word_list = [w.lower() for w in insert_entity_indicator(line.split())]
			string = " ".join(word_list)
			label = label
			if not string in data_points:
				data_points[string] = set(label)
			else:
				data_points[string].add(label)

	for k,v in data_points.items():
		X.append(k)
		y.append(list(v))

def make_graph(dev_val,dev_labels,X,y,prediction,X_pred, y_pred):
	data_points = defaultdict(set)
	data_points2 = defaultdict(set)
	for line,label,predicted_class in zip(dev_val,dev_labels,prediction):
		predicted_class = str(predicted_class)
		word_list = [w.lower() for w in line.split()]
		string = " ".join(word_list)
		if not string in data_points:
			data_points[string] = set(label)
		else:
			data_points[string].add(label)

		if not string in data_points2:
			data_points2[string] = set(predicted_class)
		else:
			data_points2[string].add(predicted_class)

	for k,v in data_points.items():
		X.append(k)
		y.append(list(v))
	for k,v in data_points2.items():
		X_pred.append(k)
		y_pred.append(v)

def entities_2_graph(texts,labels):
	entities = []
	for text,l in zip(texts,labels):
		char1 = str()
		char2 = str()
		for w in text.split():
			if w.startswith("targ-"):
				char1 = w.replace('targ','').replace('-','')
			elif w.startswith("exp-"):
				char2 = w.replace('exp','').replace('-','')
		if not char1=='':
			if not char2=='':
				entities.append((char1,char2))
	return entities

def entities_2_directed_graph(texts):
	entities = []
	for text in texts:
		temp_ent = []
		for w in text.split():
			if w.startswith("targ-"):
				tar = w.replace('targ','').replace('-','')
				if not tar=='':
					temp_ent.append(tar)
			elif w.startswith("exp-"):
				exp = w.replace('exp','').replace('-','')
				if not exp=='':
					temp_ent.append(exp)
		entities.append(tuple(temp_ent))
	return entities

def prepare_eval(characters,y,graph,param=str()):
	for i,j in zip(characters,y):
		for k in j:
			try:
				pair = i[0]+i[1]
				if not pair in graph:
					graph[pair][param] = [k]
				else:
					graph[pair][param].append(k)
			except IndexError:
				pair = "NoneNone"
	return graph

def eval(graph,graph2,graph_total_precision,graph_total_recall,graph_total_f1):
	tp = 0
	fp = 0
	fn = 0
	for (k,v),(key,val) in zip(graph.items(),graph2.items()):
		if k==key:
			c = list((Counter(v['gold']) & Counter(val['pred'])).elements())
			tp+=len(c)
			for i in val['pred']:
				if not i in v['gold']:
					fp+=1
			for i in v['gold']:
				if not i in val['pred']:
					fn+=1
	precision = 0
	recall = 0
	f1 = 0
	try:
		precision = tp / (tp + fp)
		graph_total_precision.append(precision)
	except ZeroDivisionError:
		graph_total_precision.append(0)
	try:
		recall = tp / (tp + fn)
		graph_total_recall.append(recall)
	except ZeroDivisionError:
		graph_total_recall.append(0)
	try:
		f1 = 2 * ((precision * recall) / (precision + recall))
		graph_total_f1.append(f1)
	except ZeroDivisionError:
		graph_total_f1.append(0)

def eval_2class(graph,graph2,graph_total_precision,graph_total_recall,graph_total_f1):
	tp = 0
	fp = 0
	fn = 0
	for (k,v),(key,val) in zip(graph.items(),graph2.items()):
		if k==key:
			# print(v,val)
			if len(v['gold'])==1 and len(val['pred'])==1 and v['gold']==val['pred']:
				tp+=1
			correct = Counter(v['gold'])
			predicted = Counter(val['pred'])
			# print(correct,predicted)
			for c,n in correct.items():
				if c in predicted:
					if n>predicted[c]:
						diff = n-predicted[c]
						fn+=n-predicted[c]
						tp+=n-diff
					if n<predicted[c]:
						fp+=predicted[c]-n
						diff = predicted[c]-n
						tp+=predicted[c]-diff

				else:
					fn+=1
	precision = 0
	recall = 0
	f1 = 0
	try:
		precision = tp / (tp + fp)
		graph_total_precision.append(precision)
	except ZeroDivisionError:
		graph_total_precision.append(0)
	try:
		recall = tp / (tp + fn)
		graph_total_recall.append(recall)
	except ZeroDivisionError:
		graph_total_recall.append(0)
	try:
		f1 = 2 * ((precision * recall) / (precision + recall))
		graph_total_f1.append(f1)
	except ZeroDivisionError:
		graph_total_f1.append(0)
