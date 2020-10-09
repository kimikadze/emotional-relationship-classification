def read_labels(line,labels,direction,experiment):
    if direction == "directed" and experiment == "8class":
        labels.append(line[3])
    elif  direction == "directed" and experiment == "5class":
        labels.append(line[5])
    elif  direction == "directed" and  experiment == "2class":
        labels.append(line[7])
    elif direction == "undirected" and experiment == "8class":
        labels.append(line[2])
    elif direction == "undirected" and experiment == "5class":
        labels.append(line[4])
    elif direction == "undirected" and experiment == "2class":
        labels.append(line[6])

def read_data(file,fold,window_size,experiment,direction,test):
    validation = []
    validation_labels = []
    train = []
    train_labels = []

    for line in open(file,encoding='utf-8'):
        label = str()
        line = line.strip().split("\t")
        if line[8]=='relation':
            if line[0]==str(fold):
                validation.append(slice_window(line=line[9],window_size=window_size))
                read_labels(line=line,labels=validation_labels,direction=direction,experiment=experiment)
            else:
                train.append(slice_window(line=line[9],window_size=window_size))
                read_labels(line=line,labels=train_labels,direction=direction,experiment=experiment)

    dev = validation[:round(len(validation)/2)]
    dev_labels = validation_labels[:round(len(validation_labels)/2)]

    test = validation[round(len(validation)/2):]
    test_labels = validation_labels[round(len(validation_labels)/2):]

    if not test:
        return train,train_labels,dev,dev_labels
    else:
        return train,train_labels,test,test_labels

def slice_window(line,window_size):
    line = line.split()
    new_line = str()
    targ_ind = 0
    exp_ind = 0
    for token in line:
        if token.startswith('TARG'):
            targ_ind = line.index(token)
        elif token.startswith('EXP'):
            exp_ind = line.index(token)
    if targ_ind>exp_ind:
        new_line = " ".join(line[exp_ind-window_size:targ_ind+window_size])
    elif targ_ind<exp_ind:
        new_line = " ".join(line[targ_ind-window_size:exp_ind+window_size])
    return new_line
