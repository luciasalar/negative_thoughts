# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import resample
import emot
from sklearn.svm import SVC

# Importing the dataset
dataset = pd.read_csv('status_dep_demog_big5_schwartz_swl_like.csv')


###prediction 

data = pd.read_csv('text_id_1000.csv')   ##this file contain text and userid AND SENTIMENT in the sample

####add emoticon
emo_fea = []
for i in data['text']:
	emo = emot.emoticons(i)
	if len(emo) > 0:
		emo_fea.append(emo[0]['value'])
	else:
		emo_fea.append(False)

emo_df = pd.DataFrame(emo_fea)
frames = [data, emo_df]
data1 = pd.concat(frames, axis = 1)
data1.columns.values[7] = 'emoticon'


####for some reason concat doesn't work well in here, so I join the dataframe mannually 
#data1 = pd.read_csv('data1.csv')

#remove na in id
data2 = data1[pd.notnull(data1['userid'])]
data2['senti_score'] = data2['Positive'] + data2['Negative']
data2 = data2[pd.notnull(data2['userid'])]

###process data

def preprocess(sent):

    words = str(sent).split()
    new_words = []
    # ps = PorterStemmer()
    
    for w in words:
        w = w.lower().replace("**bobsnewline**","")
        # remove non English word characters
        w = re.sub(r'[^\x00-\x7F]+',' ', w)
        # remove puncutation 
        w = re.sub(r'[^\w\s]','',w)
        # w = ps.stem(w)
        new_words.append(w)
        
    return ' '.join(new_words)

data2['text'] = data2['text'].apply(preprocess)

####length of post as feature
word_len =[]
for i in data2['text']:
	length = len(i)
	word_len.append(length)

length_df = pd.DataFrame(word_len)

#data3 = pd.concat([data2,length_df], axis = 1)
#data3.columns.values[9] = 'post_leng'

data2['post_len'] = word_len

###merge text, id with all FB features
all_data = pd.merge(dataset, data2, on = 'userid', how = 'inner')



#####convert categorical data 
#select useful variables in dataset
selected = ['userid', 'marital_status', 'ethnicity', 'gender','age','relationship_status', 'network_size','negative_yn','thoughtcat','text_y','emoticon','Positive','Negative','senti_score','post_len']
data_n = all_data[selected]
data_dep = data_n

data_dep['ethnicity'] = data_dep['ethnicity'].fillna('Other')
data_dep['marital_status'] = data_dep['marital_status'].fillna('Other')
data_dep['age'] = data_dep['age'].fillna(data_dep['age'].mean())
data_dep['relationship_status'] = data_dep['relationship_status'].fillna(0)
data_dep['network_size'] = data_dep['network_size'].fillna(data_dep['network_size'].mean())


####one hot encoding
features_oneHot = ['marital_status', 'ethnicity', 'gender','relationship_status','emoticon']

x = data_dep[features_oneHot].values
y = data_dep['negative_yn'].values
# Encoding categorical data

marital_status = pd.get_dummies(x[:, 0])
ethnicity = pd.get_dummies(x[:, 1])
gender = pd.get_dummies(x[:, 2])
relationship_status = pd.get_dummies(x[:, 3])
emoticon = pd.get_dummies(x[:, 4])

fea = pd.concat([marital_status,ethnicity,gender,relationship_status,emoticon], axis =1).values


features = ['age','network_size','Positive','Negative','senti_score','post_len'] 
x2 = data_dep[features].values

fb_fea = np.concatenate((fea, x2), axis=1)


###text as feature
cv = CountVectorizer()
text_vec = cv.fit_transform(data_dep['text_y']).toarray()


#convert label to number  ####one hot encoder randomly assign yes/no to 1 or 2
#labelencoder = LabelEncoder()
#y = labelencoder.fit_transform(y.astype(str))
y[y == 'yes'] = 1
y[y == 'no'] = 2
y[y == 'mixed'] = 1 #replace mix with yes
y=y.astype('int')

####convert to tfidf
tfidf_transformer = TfidfTransformer()
text_vec = tfidf_transformer.fit_transform(text_vec).toarray()

####combine with all features
x = np.concatenate((text_vec, fb_fea), axis=1)

####convert time 
#t = data['time'].values.reshape((81,1))
#x = np.concatenate((x, t), axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)
#0.64436619718309862

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean() # 0.61982082594022903


print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

# 0.537867869264
# 0.537341117801
# 0.538841093117


####SVM no  oversample

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(x_train, y_train) 

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)
cm

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

#0.519806763285
#0.560515873016
#0.530870445344


#####SVM   address imbalanced class
#y2 = y.reshape((944,1))
#xy = np.concatenate((x, y2), axis=1)
#xy_df = pd.DataFrame(xy)


####oversample before crossed validation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#Separate majority and minority classes

x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)
x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

xy_df = pd.concat([x_train_df, y_train_df], axis =1)
xy_df2 = pd.concat([x_test_df, y_test_df], axis =1)


xy_df.iloc[:,-1]
df_majority = xy_df[xy_df.iloc[:,-1]==1]   ##461
df_minority = xy_df[xy_df.iloc[:,-1]==2]   #199


 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=461,    # to match majority class
                                 random_state=12) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.iloc[:,-1].value_counts()

#2.0    461
#1.0    461
#Name: 5609, dtype: int64
train_x2 = df_upsampled.iloc[:,:-1]
train_y2 = df_upsampled.iloc[:,-1]

########

xy_df2.iloc[:,-1]
df_majority2 = xy_df2[xy_df2.iloc[:,-1]==1]   ##461
df_minority2 = xy_df2[xy_df2.iloc[:,-1]==2]   #199


 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=461,    # to match majority class
                                 random_state=12) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.iloc[:,-1].value_counts()

#2.0    461
#1.0    461
#Name: 5609, dtype: int64
train_x2 = df_upsampled.iloc[:,:-1]
train_y2 = df_upsampled.iloc[:,-1]



#######SVC
from sklearn.svm import SVC
clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(x_train, y_train) 

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)
cm

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

##
# 0.79091233809
# 0.807734303913
# 0.792995049505


#confusion matrix
#array([[137,  63],
#       [ 20, 182]])



#####grid search 
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


grid_search_item = GridSearchCV(estimator = clf,
                          param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search_item.fit(x_train, y_train)

grid_search.best_score_   
grid_search.best_params_


#######do grid search 10 times
grid_searches = []
best_accuracy = []
best_parameters = []

for i in range(10):
	grid_search_item = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
	grid_search = grid_search_item.fit(x_train, y_train)
	print(type(grid_search))
	best_accuracy_item = grid_search.best_score_   
	best_parameters_item = grid_search.best_params_
	grid_searches.append(grid_search_item)
#	np.vstack(best_accuracy, best_accuracy_item)
	best_parameters.append(best_parameters_item)
	best_accuracy.append(best_accuracy_item)


###10 fold cross validation                        
accuracies = cross_val_score(estimator = clf, X = x, y = y, cv = 10)
accuracies.mean()
#0.8273292627770239




##merge with lda
lda_500 = pd.read_csv('~/phd_work/crowdflower/lda_500/lda_500.csv')

status_sample = pd.read_csv('~/phd_work/crowdflower/status_sample.csv')
status_sample = pd.DataFrame(status_sample['text'])

lda_500 = pd.concat([status_sample,lda_500], axis =1)

all_fea_lda = pd.merge(data_dep, lda_500, left_on = 'text_y', right_on = 'text', how = 'left')

####select lda features and convert it to array
lda2 = all_fea_lda.loc[21:500]