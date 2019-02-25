import pandas as pd
import numpy as np
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

df = pd.read_csv('wikigoldPOS.csv')
previousWordPOS = ''
IsFirstUpperCase = False
WordEnding = ['.','\n','\n\r' ]
IsPreviousUpperCase = False
previousWord = ''
df = pd.DataFrame(columns=['word','pos','prevPOS','PrevWord','iob','VectorCount','PrevIOB','CaseStatus'])
seq = 1
for i in df:
    CaseStatus = 0
    word = i[1]
    iob = splittedword[1]
    EndOfStringBefore = previousWord in WordEnding
    if EndOfStringBefore == False:
        CaseStatus = StringCaseStatus(word)
    df.loc[seq]=[word,CaseStatus,previousWordPOS,previousWord,iob,0]
    previousWord = word
    IsPreviousUpperCase = CaseStatus
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
print(df)

