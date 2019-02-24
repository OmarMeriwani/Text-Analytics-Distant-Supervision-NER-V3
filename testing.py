import pandas as pd
import numpy as np
import csv
import nltk
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
textdata = open('wikigold.conll.txt', 'r')
tokens = textdata.read().split(' ')
previousWordPOS = ''
IsFirstUpperCase = False
WordEnding = ['.','\n','\n\r' ]
IsPreviousUpperCase = False
previousWord = ''
df = pd.DataFrame(columns=['word','CaseStatus','PrevWord','iob'])
seq = 1
for i in tokens:
    CaseStatus = 0

    if i != None and i != ' ' and i != '\n':
        i = i.rstrip('\r\n')
        splittedword = i.split('|')
        word = splittedword[0]
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
