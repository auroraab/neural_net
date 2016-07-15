from keras.utils import np_utils

import numpy as np
import random

aas = list('ARNDCQEGHILKMFPSTWYV-') 

#convert amino acid string into binary vector 
aa2num = dict.fromkeys(list('ARNDCQEGHILKMFPSTWYV-') , 0)
for aa in list('ARNDCQEGHILKMFPSTWYV-'):
    aa2num[aa] = 'ARNDCQEGHILKMFPSTWYV-'.index(aa)

###   FEATURE GENERATION   ######################################################

#truncate sequence to desired residues
def trunc_seq(string, rr_indices, hk_indices, zeroindex = 0):
    ''' Truncates sequence to include only desired residues '''
    rr = ""
    for i in rr_indices:
        try:
            rr+= string[i-zeroindex]
        except IndexError:
            print "failed at index ",i, " of " , string
    hk = ""
    for i in hk_indices:
        try:
            hk+= string[i-zeroindex]
        except IndexError:
            print "failed at index ",i, " of " , string
    return rr, hk

### WAZZUP GITHUB

def aa2bin(seq):
    ''' Takes in a sequence and returns binary matrix form (size = len x 21) '''
    bin_aa = np.zeros((len(seq),21))
    aa = list(seq)
    for i in range(len(seq)):
        try: 
            bin_aa[i,aa2num[aa[i]]] = 1
        except KeyError:
            bin_aa[i,aa2num["-"]] = 1
    return bin_aa

def make_all_features(seqs,rr_ind, hk_ind, edges = False):
    l = len(makeFeature(seqs[0],rr_ind, hk_ind, edges = edges))
    n = len(seqs)
    features = np.zeros((n,l))
    for i in range(n):
        features[i] = makeFeature(seqs[i],rr_ind, hk_ind, edges=edges)
    return features

def makeFeature(s,rr_ind, hk_ind, edges = False):
    feat = []
    rr, hk = trunc_seq(s, rr_ind, hk_ind)
    if not edges:
        feat = aa2bin(rr+hk).reshape(-1)
        return feat
    else:
        rr_feat = aa2bin(rr)
        hk_feat = aa2bin(hk)
        for edge in edges:
            aa1 = s[edge[0]]
            aa2 = s[edge[1]]
            edge_feat = np.outer(aa2bin(aa1),aa2bin(aa2)).reshape(-1)
            feat = np.concatenate((feat, edge_feat))
        return feat
    
                
###   BUILDING DATASETS   ######################################################

def extract_sequences(filename):
    ''' Import data for training/validation/testing ''' 
    datafile = open(filename,'r')
    seqs = []
    labels = []
    i = 0
    for line in datafile:
        line = line.rstrip().split()
        labels += [int(line[0])]
        seqs += [line[1]]
        i += 1
    labels = np.array(labels)
    return labels, seqs

def balance_data(features,labels):
    ''' Balances +/- points (approximately) in dataset by removing negative data points ''' 
    new_features = np.zeros(features.shape)
    new_labels = np.zeros(labels.shape)
    
    if len(labels.shape) == 2:
        balance = (labels.sum(axis = 0)/len(labels))
        p = balance[1]/balance[0]
        j = 0
        for i in range(labels.shape[0]):
            if labels[i,1] == 1 or random.random() < p:
                new_features[j] = features[i]
                new_labels[j] = labels[i]
                j += 1
    else:
        p = labels.sum()/float(len(labels))
        j = 0
        for i in range(labels.shape[0]):
            if labels[i] == 1 or random.random() < p:
                new_features[j] = features[i]
                new_labels[j] = labels[i]
                j += 1

    return new_features[:j], new_labels[:j]

def rand_aas(l):
    ''' Random amino acid sequence generator ''' 
    
    seq = ""
    for i in range(l):
        seq += random.choice(aas)
    return seq

def build_datasets(rr_ind, hk_ind, balanced = 1, random_seqs = False):
    ''' Builds model input features from text file data '''
  
    train_labels, train_seqs = extract_sequences("training_data.txt")
    val_labels, val_seqs = extract_sequences("validation_data.txt")
    test_labels, test_seqs = extract_sequences("test_data.txt")

    train_features = make_all_features(train_seqs,rr_ind, hk_ind)
    val_features = make_all_features(val_seqs,rr_ind, hk_ind)
    test_features = make_all_features(test_seqs,rr_ind, hk_ind)

    if random_seqs:
        # add random sequences (labeled nonfunctional) as 10% training data
        n = int(0.1*len(train_y))
        print n
        l = len(train_seqs[0])
        rand_labels = np.zeros((n))
        rand_seqs = []
        for i in range(n):
            rand_seqs += [rand_aas(l)]
        rand_features = make_all_features(rand_seqs,rr_ind, hk_ind)
        train_features = np.vstack((randX, trainX))
        train_labels = np.hstack((rand_y, train_y))

    # categorizing the labels
    num_classes = 2
    train_labels = np_utils.to_categorical(train_labels, num_classes)
    val_labels = np_utils.to_categorical(val_labels, num_classes)
    test_labels = np_utils.to_categorical(test_labels, num_classes)

    if balanced:
        test_features, test_labels = balance_data(test_features, test_labels)
        val_features, val_labels = balance_data(val_features, val_labels)
        
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

