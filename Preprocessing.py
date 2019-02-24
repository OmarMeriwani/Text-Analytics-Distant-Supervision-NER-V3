import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv

def StringCaseStatus(sent):
    sent = str(sent)
    count = 0
    if (sent[0].isupper() == True):
        count += 1
    for c in range(1, len(sent)):
        if sent[c].isupper() == True:
            count += 1
    if count == len(sent):
        return 3
    if count > 1 and count < len(sent):
        return 2
    if count == 1:
        return 1
    else:
        return 0


count = 0
'''with open("DataSet.csv", "rt") as fin:
    with open("DataSet2.csv", "wt") as fout:
        for line in fin:
            count += 1
            listline = list(line)
            listline[0] = ''
            listline[len(line) - 1] = ''
            line = ''.join(listline)
            fout.write(line.replace(')(', ';'))
print(count)
'''
slist= []
#skiprows=512*n,
file = pd.read_csv('DataSet.csv', header=None, nrows=2000000, chunksize=1000)
corpus = ''
print('Data collection..')
for chunk in file:
    list = chunk.values.tolist()
    tempString = ''
    for st in list:
        #print(st[1])
        if str(st[1]) != '.':
            tempString = tempString + ' ' + str(st[1]).lower()
        else:
            slist.append(tempString)
            tempString = ''
vec = CountVectorizer()
print('Count Vector Fitting..')
x = vec.fit_transform(slist)
print(type(x))
print(vec.get_feature_names())
print(vec.vocabulary_.get('daniel')/len(vec.get_feature_names()))
CorpusSize = len(vec.get_feature_names())
print('Corpus size: ', CorpusSize)
dfWithVectors = pd.DataFrame(columns=['word','pos','prevPOS','PrevWord','iob','VectorCount','PrevIOB','CaseStatus'])
seq = 0
PrevIOB = ''
print('Setting extra features..')

file = pd.read_csv('DataSet.csv', header=None, nrows=2000000, chunksize=500)
with open('Features.csv', mode='w', encoding='utf-8') as csvfile:
    for chunk in file:
        list = chunk.values.tolist()
        tempString = ''
        for st in list:
            if st[1] == None:
                PrevIOB = st[6]
                seq += 1
                continue
            #print(st[1])
            TFWeight = 0
            word = st[1]
            try:
                word = st[1].lower()
            except:
                word = st[1]
            if vec.vocabulary_.get(word) != None:
                TFWeight = vec.vocabulary_.get(word)
                TFWeight = int(TFWeight)/int(CorpusSize)
            #3,Companion,NNP,True,NNP,Oxford,I-MISC
            caseStatus = StringCaseStatus(st[1])
            dfWithVectors.loc[seq] = ([st[1], st[2],st[4], st[5],st[6],str(TFWeight.__round__(3)),PrevIOB, str(caseStatus)])
            output = [st[1], st[2],st[4], st[5],st[6],TFWeight,PrevIOB, caseStatus]
            '''
            try:
                dfWithVectors.to_csv(csvfile, header=False)
            except:
                PrevIOB = st[6]
                seq += 1
                continue
                '''
            if seq % 500 == 0 :
                print(seq)
                dfWithVectors.to_csv(csvfile, header=False)
                dfWithVectors = pd.DataFrame(columns=['word','pos','prevPOS','PrevWord','iob','VectorCount','PrevIOB','CaseStatus'])
            PrevIOB = st[6]
            seq += 1