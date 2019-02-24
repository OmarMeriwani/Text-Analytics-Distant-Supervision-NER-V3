import pandas as pd
import numpy as np
import csv
textdata = open('aij-wikiner-en-wp2', 'r')
tokens = textdata.read().split(' ')
previousWordPOS = ''
IsFirstUpperCase = False
WordEnding = ['.','\n','\n\r' ]
IsPreviousUpperCase = False
previousWord = ''
df = pd.DataFrame(columns=['word','pos','uppercase','prevPOS','PrevWord','iob','VectorCount'])
seq = 1
with open('DataSet3333.csv', mode='w') as csvfile:
    for i in tokens:
        IsFirstUpperCase = False

        if i != None and i != ' ' and i != '\n':
            i = i.rstrip('\r\n')
            splittedword = i.split('|')
            word = splittedword[0]
            pos = splittedword[1]
            iob = splittedword[2]
            FirstLetter = word[0].isupper()
            EndOfStringBefore = previousWord in WordEnding
            if EndOfStringBefore == False and FirstLetter == True:
                IsFirstUpperCase = True
            #output = {'seq':seq, 'word': word.rstrip("\n\r"),'pos': pos,'FirstUC': IsFirstUpperCase,'prevPOS':previousWordPOS,'PREVW':previousWord.rstrip("\n\r"),'iob':iob.rstrip("\n\r")}
            df.loc[seq]=[word,pos,IsFirstUpperCase,previousWordPOS,previousWord,iob,0]
            #output2 = seq, word.rstrip("\n\r"),pos,IsFirstUpperCase,previousWordPOS,previousWord.rstrip("\n\r"),iob.rstrip("\n\r")
            #output2 = str(output2).replace('\n','')
            #csvfile.writelines(str(output2))

            #print(output)
            if seq % 10000 == 0 :
                print(seq)
                df.to_csv(csvfile, header=False)
                df = pd.DataFrame(columns=['word','pos','uppercase','prevPOS','PrevWord','iob','VectorCount'])
                #break
            previousWordPOS = pos
            previousWord = word
            IsPreviousUpperCase = IsFirstUpperCase
            seq = seq + 1
print(df)

'''
trigramsDF = pd.DataFrame()
trigramsDF.columns = ['trigram','count']
for row in df:
    word = df['word']
    for i in range(0, len(word) - 1):
        try:
            trigramChars = [word[i], word[i+1], word[i+2]]
            trigram = ''.join(trigramChars)
            if trigram in trigramsDF['trigram']:
                CurrentCount = trigramsDF['trigram']['count']
                trigramsDF['trigram']['count'] = CurrentCount + 1
            else:
                trigramsDF.append(trigram,1)
        except:
            continue
'''





