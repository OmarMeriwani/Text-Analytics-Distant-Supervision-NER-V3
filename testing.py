import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer

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

file = pd.read_csv('wikigoldPOS.csv', header=None, chunksize=1000)
corpus = ''
slist = []
print('Data collection..')
for chunk in file:
    list = chunk.values.tolist()
    tempString = ''
    for st in list:
        #print(st[1])
        if str(st[1]) != '.' or str(st[1]) == ' ':
            tempString = tempString + ' ' + str(st[1]).lower()
        else:
            slist.append(tempString)
            tempString = ''
vec = CountVectorizer()
print('Count Vector Fitting..')
print(slist)
x = vec.fit_transform(slist)
print(type(x))
print(vec.get_feature_names())
#print(vec.vocabulary_.get('daniel')/len(vec.get_feature_names()))
CorpusSize = len(vec.get_feature_names())

print('Iterating again..')
df = pd.read_csv('wikigoldPOS.csv',header=None, chunksize=1000)
previousWordPOS = ''
IsFirstUpperCase = False
previousPOS = ''
PrevIOB = ''
WordEnding = ['.','\n','\n\r',' ' ]
IsPreviousUpperCase = False
previousWord = ''
df2 = pd.DataFrame(columns=['word','pos','prevPOS','PrevWord','iob','Vector','PrevIOB','CaseStatus'])
seq = 1
for chunk in df:
    list = chunk.values.tolist()
    tempString = ''
    for i in list:
        TFWeight = 0
        word = i[1]
        if vec.vocabulary_.get(word.lower()) != None:
            TFWeight = vec.vocabulary_.get(word.lower())
            TFWeight = int(TFWeight) / int(CorpusSize)
        CaseStatus = 0
        pos = i[2]
        iob = i[3]
        EndOfStringBefore = previousWord in WordEnding
        if EndOfStringBefore == False:
            CaseStatus = StringCaseStatus(word)
        df2.loc[seq]=[word,pos,previousPOS,previousWord,iob,TFWeight,PrevIOB,CaseStatus]
        previousWord = word
        previousPOS = pos
        PrevIOB = iob
        IsPreviousUpperCase = CaseStatus
        seq += 1
    #tagged_tokens = nltk.pos_tag(word_list)
'''
with open('TestDS.csv', mode='w') as csvfile:
    if seq % 10000 == 0:
        print(seq)
        df.to_csv(csvfile, header=False)
        df = pd.DataFrame(columns=['word', 'pos', 'uppercase', 'prevPOS', 'PrevWord', 'iob', 'VectorCount'])
        # break
        seq = seq + 1
'''
df2.to_csv('wikiGoldFeatures.csv')
