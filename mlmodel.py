import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import KFold, cross_val_score
import seaborn.apionly as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pickle
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = "(%.2f)"%(cm[i, j])
        #print t
#         plt.text(j, i, t,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('IOB-Confusion-Matrix-SVM.png')

df = pd.read_csv('wikiGoldFeatures.csv',header=None)
#df = pd.read_csv('Features.csv',header=None,nrows=50000)
le = LabelEncoder()

categories = ['LOC','MISC', 'ORG']
categories2 = ['I-LOC','I-MISC','I-PER', 'I-ORG','O']
'''
for i in range(0,len(df)):
    for j in range(0,2):
        if categories[j] in df.at[i,5]:
            df.iat(i, 5, categories2[j])
'''
#print(len(df))
df = df[df[5].isin(categories2)]
df = df[df[7].isin(categories2)]
#print(len(df))
#df.drop(df.index[df[5] != 'I-PER'], inplace = True)
#df.dropna()
df[2] = le.fit_transform(df[2])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)
df[3] = le.fit_transform(df[3].astype(str))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)
df[5] = le.fit_transform(df[5].astype(str))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)
df[7] = le.fit_transform(df[7].astype(str))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)
df.drop(df.columns[4],axis=1,inplace=True)
df.drop(df.columns[0],axis=1,inplace=True)
#df.drop(df.columns[4], axis=1,inplace=False)
#print(df[2:20])

#data = sns.load(df)
#data.hist(by=5,column = 6)
#plt.savefig('Vector-Histogram.png')

X = df[[2,3,7,6,8]]
Y = df[5]

#print(X[:10])
#print(Y[:10])
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state=100)

gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)
#print("NAIVE BAYES Accuracy:",metrics.accuracy_score(y_test, y_pred))


#svmm = SVC(gamma='auto',random_state=100)
#svmm.fit(x_train,y_train)
pkl_file = open('SVMModel.sav', 'rb')
svmm = pickle.load(pkl_file)

#svmm = pickle.load('SVMModel.sav')
#pickle.dump(svmm,open('SVMModel.sav','wb'))
y_pred2 = svmm.predict(x_test)
#pickle.dump(svmm, 'svmmodel.sav')
#'''
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred2))
#Confusion matrix
np.set_printoptions(precision=2)
plt.figure()
cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=range(len(set(y_test))), normalize = False,title='Confusion matrix')
scores = cross_val_score(svmm, x_train, y_train, cv=10)
print(scores)

'''ann = MLPClassifier(activation='tanh', hidden_layer_sizes=(10, 10, 10), solver='lbfgs', random_state=100)
ann.fit(x_train,y_train)
y_pred3 = ann.predict(x_test)
print("ANN Accuracy:",metrics.accuracy_score(y_test, y_pred3))
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
#Confusion matrix
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(len(set(y_test))), normalize = False,title='Confusion matrix')
'''