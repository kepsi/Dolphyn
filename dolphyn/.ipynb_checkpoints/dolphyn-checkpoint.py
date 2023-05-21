#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pkg_resources
import random
from sklearn.ensemble import RandomForestClassifier

def balance_DS(X, y, random_state=42):
    wildtypeIDs = set([item[0] for item in X.index.str.split("_")])
    random.seed(random_state)
    random.seed(10)
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

def findEpitopes(protein_seq_file, epitile_size, testrun = 0, epitope_probability_cutoff = 0.5):
    count = 0
    global_epitopes = {}   
    ep, _ = initEpiPredictor(epitile_size)    
    
    with open(protein_seq_file, "r") as org:
        for line in org.readlines():
            line = line.strip()
            if testrun!= 0 and count/2 > testrun:
                break
            count = count + 1

            if (count/2)%500 == 0:
                print(str(count/2), "sequences done")

            if line.startswith(">"):
                protein = line[1:]
            else:
                pseq = line
                protein_length = len(pseq)
                epitope_proba_atpos = {}

                prot_15mer_feat = kmer_features_of_protein(pseq,epitile_size)
                probas = [class_proba[1] for class_proba in ep.predict_proba(prot_15mer_feat)]
                for startpos in range(len(pseq)-epitile_size):
                    epitope_proba_atpos[startpos] = np.min(probas[startpos:startpos+3])

                while len(epitope_proba_atpos)>0:
                    max_epi_proba_pos = max(epitope_proba_atpos, key=epitope_proba_atpos.get) 
                    min_epi_proba = epitope_proba_atpos[max_epi_proba_pos]
                    offset = np.argmax(probas[max_epi_proba_pos:max_epi_proba_pos+3])
                    max_epi_proba_pos+=offset
                    epi_proba = probas[max_epi_proba_pos]
                    epi_seq = pseq[max_epi_proba_pos:max_epi_proba_pos+epitile_size]
                    if min_epi_proba < epitope_probability_cutoff :
                        break
                    else:
                        if epi_seq not in global_epitopes:
                            global_epitopes[epi_seq] = {"probability":epi_proba,"proteins":[],"start_pos":[]}   
                        global_epitopes[epi_seq]["proteins"].append(protein)                    
                        global_epitopes[epi_seq]["start_pos"].append(max_epi_proba_pos)                   

                        for pos in range(max_epi_proba_pos-epitile_size, max_epi_proba_pos+epitile_size):
                            if pos in epitope_proba_atpos:
                                del epitope_proba_atpos[pos]
    return(global_epitopes)

def saveGlobalEpitopes(global_epitopes, filename):
    import json   
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(filename,"w") as f:
        f.write(json.dumps(global_epitopes, cls=NpEncoder))
        
def readGlobalEpitopes(filename):
    import json
    with open(filename) as json_file:
        global_epitopes = json.load(json_file)
    return(global_epitopes)

def peptideStitching(no_epis_per_tile, linker_seq, global_epitopes, return_unused_epis = False):
    dolphyn_tiles = {}
    unused_epitopes = set()
    unused_epitopes_count = 0
    epi_df = pd.DataFrame.from_dict(global_epitopes, orient='index')
    epi_df["no_proteins"] = [len(v) for v in epi_df["proteins"]]
    epi_df = epi_df.sort_values(by=["no_proteins", "probability"], ascending=False)
    epis_to_be_treated = set(epi_df.index)

    for epi in epi_df.index:
        if epi in epis_to_be_treated:
            #find epitopes which have the same protein set
            pset = epi_df.loc[epi,"proteins"]
            proteinset_epis=epi_df[epi_df["proteins"].apply(lambda x: x==pset)]
            epis_to_be_treated = epis_to_be_treated-set(proteinset_epis.index)

            #the following will drop epitopes which have lowest probability and have no partners to make tiles of 3
            no_tiles = int(len(proteinset_epis)/no_epis_per_tile) 
            tiles = ["" for _ in range(no_tiles)]
            tile_probas = [list() for _ in range(no_tiles)]
            tile_seq = ""
            epi_index = 0
            for stich in range(1,no_epis_per_tile+1):
                for tn in range(no_tiles):
                    tile_probas[tn].append(proteinset_epis.iloc[epi_index,:]["probability"])
                    sequence = proteinset_epis.iloc[epi_index,:].name
                    epi_index += 1
                    if stich == 1:
                        tiles[tn] = sequence
                    else:
                        tiles[tn] = tiles[tn] + linker_seq + sequence                

            while len(proteinset_epis) > epi_index :
                unused_epitopes_count += 1
                if(return_unused_epis):
                    unused_epitopes.add(proteinset_epis.iloc[epi_index,:].name)
                epi_index += 1

            for idx, tile in enumerate(tiles):
                dolphyn_tiles[tile] = {"protein set":pset, 
                                      "tile number":(idx+1), 
                                      "tiles in protein(set)": no_tiles,
                                     "probabilities":tile_probas[idx]}

    print("total number of dolphyn tiles = ", len(dolphyn_tiles))
    print("number of unused epitopes:", unused_epitopes_count)
    return(dolphyn_tiles,unused_epitopes)

def writeDolphynPeptidesToFAA(dolphynTiles, filename, hash_protein_names=True):
    protein_hash = dict()
    with open(filename, "w") as pf:
        for seq, tile in dolphynTiles.items():
            n = str(tile["protein set"])+"_"+str(tile["tile number"])+"of"+str(tile["tiles in protein(set)"])
            if hash_protein_names:
                h = abs(hash(n))
            else:
                h = str(n)
            protein_hash["dolphyn_"+str(h)] = "dolphyn_"+n
            pf.write(">"+"dolphyn_"+str(h)+"\n")
            pf.write(seq+"*\n") # adds a stop codon in the end
    return(protein_hash)