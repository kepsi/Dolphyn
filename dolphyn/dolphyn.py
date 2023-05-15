#!/usr/bin/env python
# coding: utf-8

#import csv
import pandas as pd
import pkg_resources
import random
from sklearn.ensemble import RandomForestClassifier

def balance_DS(X, y, random_state=42):
    wildtypeIDs = set([item[0] for item in X.index.str.split("_")])
    random.seed(random_state)
    size_smaller_group = y.value_counts().min()
    pos_IDs = set(y[y==1].index)
    neg_IDs = set(y[y==0].index)
    neg_IDs= random.sample(neg_IDs, size_smaller_group)
    pos_IDs= random.sample(pos_IDs, size_smaller_group)
    balancedIDs = set(pos_IDs + neg_IDs)
    y_bal = y.loc[balancedIDs,]
    X_bal = X.loc[balancedIDs,]    
    return(X_bal, y_bal)

def train_test_split_wildtypes(X, y, test_size, random_state=42):
    wildtypeIDs = set([item[0] for item in X.index.str.split("_")])
    random.seed(random_state)
    test_IDs = random.sample(wildtypeIDs, int(test_size*len(wildtypeIDs)))
    train_IDs = wildtypeIDs.difference(test_IDs)
    X_train = X.iloc[[item[0] in train_IDs for item in X.index.str.split("_")],:]
    X_test = X.iloc[[item[0] in test_IDs for item in X.index.str.split("_")],:]
    y_train = y.iloc[[item[0] in train_IDs for item in y.index.str.split("_")]]
    y_test = y.iloc[[item[0] in test_IDs for item in y.index.str.split("_")]]    
    return(X_train, X_test, y_train, y_test)

def initEpiPredictor(epi_size = 15, n_jobs=6, random_state=42, testset_size = 0):
    feat,labs = getPEDSTrainingSet(epi_size)
    X=feat
    y=labs
    X_test, y_test = None, None
    if testset_size > 0:
        X, X_test, y, y_test = train_test_split_wildtypes(feat, labs, test_size=testset_size, random_state = random_state) 
    X_train, y_train = balance_DS(X, y, random_state = random_state) 
    clf=RandomForestClassifier(n_estimators=100, n_jobs=6, random_state=random_state, oob_score=True)
    clf.fit(X_train,y_train)
    return(clf, ((X_test, y_test), (X_train, y_train)))

def getPEDSTrainingSet(epi_size):
    stream_feat = pkg_resources.resource_stream(__name__, 'trainingdata/peds_features_'+str(epi_size) + 'mer.csv')
    stream_label = pkg_resources.resource_stream(__name__, 'trainingdata/peds_labels_'+str(epi_size) + 'mer.csv')
    feat = pd.read_table(stream_feat, sep=",", index_col = 0)
    labs = pd.read_table(stream_label, sep=",", index_col = 0)
    labs=labs["reactivity_binary"]
    labs=labs.astype('int')
    return(feat,labs)
    

AA_DIAM = {'A' : "S", 'R': "K", 'N': "K", 'D': "K", 'C': "C", 'Q': "K", 'E': "K", 'G': "G", 'H': "H", 'I': "I", 'L': "I", 'K': "K", 'M': "M", 'F': "F", 'P': "P", 'S': "S", 'T': "S", 'W': "W", 'Y': "Y", 'V': "I"}
AMINOACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
DIAMONDS = ['K', 'C', 'G', 'H', 'I', 'M', 'F', 'Y', 'W', 'P', 'S']

def all_feature_names():
    feat_list = ["sc_hydrophobic","sc_polaruncharged","sc_poseleccharged","sc_negeleccharged"]
    for aa in AMINOACIDS:
        feat_list.append(aa)
    
    for d_aa in DIAMONDS:
        feat_list.append("d_" + d_aa)
        
    for aa1 in AMINOACIDS:
        for aa2 in AMINOACIDS:
            feat_list.append(aa1+aa2)
    
    for d_aa1 in DIAMONDS:
        for d_aa2 in DIAMONDS:
            feat_list.append("d_" + d_aa1 + d_aa2)
            
    return feat_list

FEATURE_LIST = all_feature_names()

def get_diam(aa_seq):
    diam = ""
    for aa in aa_seq:
        diam += AA_DIAM[aa]        
    return(diam)

def kmer_features_of_protein(seq, k):   
    feat = {}
    l = len(seq) 
    diam_seq = get_diam(seq)
    lk1 = l-k+1
    for f in FEATURE_LIST:
        feat[f] = [0] * (l-k+1)    
    feat = pd.DataFrame(feat)    
    for i in range(0,l):
        aa = seq[i]
        feat[aa][max(0,i-k+1):min(lk1,i+1)] += 1
        for f in AA_FEAT[aa]:
            feat[f][max(0,i-k+1):min(lk1,i+1)] += 1    
    rangelen = l-1
    for i in range(0, rangelen):
        double = seq[i:i+2]
        feat[double][max(0,i-k+2):min(lk1,i+1)] += 1
        diam_double = diam_seq[i:i+2]
        feat["d_" + diam_double][max(0,i-k+2):min(lk1,i+1)] += 1
    return(feat)  

AA_FEAT = {'A':["sc_hydrophobic", "d_S"], 
           'R':["sc_poseleccharged", "d_K"], 
           'N':["sc_polaruncharged", "d_K"],
           'D':["sc_negeleccharged", "d_K"], 
           'C':["d_C"], 
           'Q':["sc_polaruncharged", "d_K"], 
           'E':["sc_negeleccharged", "d_K"], 
           'G':["d_G"],
           'H':["sc_poseleccharged", "d_H"], 
           'I':["sc_hydrophobic", "d_I"], 
           'L':["sc_hydrophobic", "d_I"], 
           'K':["sc_poseleccharged", "d_K"], 
           'M':["sc_hydrophobic", "d_M"],
           'F':["sc_hydrophobic", "d_F"], 
           'P':["d_P"], 
           'S':["sc_polaruncharged", "d_S"], 
           'T':["sc_polaruncharged", "d_S"], 
           'W':["sc_hydrophobic", "d_W"],
           'Y':["sc_hydrophobic", "d_Y"], 
           'V':["sc_hydrophobic", "d_I"]
          }