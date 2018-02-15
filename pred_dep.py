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


# Importing the dataset
dataset = pd.read_csv('~/phd_work/mypersonality_data/status_dep_demog_big5_schwartz_swl_like.csv')

dep = dataset.iloc[:, 12:32].values

# append userid 
userid = dataset['userid'].values.reshape((333,1))
dep = np.append(arr = userid, values = dep, axis = 1)

#filter invalid result, -1 means invalid
dep = np.asarray([row for row in dep if -1 not in row])

#item 4, 8, 12 and 16 has to be reversed scored
scale = lambda x: x * -1 
dep[:,3] = scale(dep[:,3])
dep[:,7] = scale(dep[:,7])
dep[:,11] = scale(dep[:,11])
dep[:,15] = scale(dep[:,15])

#caculate dep score
dep_sum = dep[:, 1:].sum(axis=1).reshape((301,1))

#append userid and dep_sum
dep_id = dep[:, 0].reshape((301,1))
dep_d = np.append(arr = dep_id, values = dep_sum, axis = 1)

#convert array to dataframe 
dep_df = pd.DataFrame(dep_d)
dep_df = dep_df.rename(index=str, columns={0: "userid", 1: "dep_score"})




###prediction 

data = pd.read_csv('text_id_1000.csv')   ##this file contain text and userid in the sample

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

data['text'] = data['text'].apply(preprocess)

###merge text, id with all FB features
all_data = pd.merge(dataset, data, on = 'userid', how = 'inner')



#####convert categorical data 
#select useful variables in dataset
selected = ['userid', 'marital_status', 'ethnicity', 'gender','age','relationship_status', 'network_size', 'ope','con','ext',
'agr','neu','swl','negative_yn','thoughtcat','text_y']
data = all_data[selected]
data_dep = data

data_dep['ethnicity'] = data_dep['ethnicity'].fillna('Other')
data_dep['marital_status'] = data_dep['marital_status'].fillna('Other')
data_dep['age'] = data_dep['age'].fillna(data_dep['age'].mean())
data_dep['relationship_status'] = data_dep['relationship_status'].fillna(0)
data_dep['network_size'] = data_dep['network_size'].fillna(data_dep['network_size'].mean())

features = ['userid', 'marital_status', 'ethnicity', 'gender','age','relationship_status', 'network_size', 'ope','con','ext',
'agr','neu','swl']

x = data_dep[features].values
y = data_dep['negative_yn'].values
# Encoding categorical data
#Encode labels with value between 0 and n_classes-1.
labelencoder = LabelEncoder()

x[:, 1] = labelencoder.fit_transform(x[:, 1])
x[:, 2] = labelencoder.fit_transform(x[:, 2])
x[:, 3] = labelencoder.fit_transform(x[:, 3])
x[:, 5] = labelencoder.fit_transform(x[:, 5])

onehotencoder = OneHotEncoder(categorical_features = [8])
x1 = x[:, 1:]
x1 = onehotencoder.fit_transform(x1).toarray()


###text as feature
cv = CountVectorizer()
x = cv.fit_transform(data['text_y']).toarray()


#convert label to number  ####one hot encoder randomly assign yes/no to 1 or 2
#labelencoder = LabelEncoder()
#y = labelencoder.fit_transform(y.astype(str))
y[y == 'yes'] = 1
y[y == 'no'] = 2
y[y == 'mixed'] = 1 #replace mix with yes
y=y.astype('int')

####convert to tfidf
tfidf_transformer = TfidfTransformer()
x = tfidf_transformer.fit_transform(x).toarray()

####combine with all features
x = np.concatenate((x1, x), axis=1)

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
#0.63028169014084512


accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean() # 0.60007793201823056

print(f1_score(y_test,y_pred, average = 'macro'))
print(precision_score(y_test,y_pred, average = 'macro'))
print(recall_score(y_test,y_pred, average = 'macro'))

# 0.537867869264
# 0.537341117801
# 0.538841093117



#####SVM   address imbalanced class
y2 = y.reshape((944,1))
xy = np.concatenate((x, y2), axis=1)
xy_df = pd.DataFrame(xy)


from sklearn.utils import resample

#Separate majority and minority classes

xy_df[5609]
df_majority = xy_df[xy_df[5609]==1]   ##669
df_minority = xy_df[xy_df[5609]==2]   #275
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=669,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled[5609].value_counts()

#2.0    669
#1.0    669
#Name: 5609, dtype: int64
x = df_upsampled.iloc[:,:-1]
y = df_upsampled[5609]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#from sklearn.linear_model import SGDClassifier

###fit svm
# svm_classifier = SGDClassifier(alpha=0.03, average=False, class_weight=None, epsilon=0.07,
#        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#        learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
#        n_jobs=1, penalty='l1', random_state=None,
#        shuffle=True, tol=None, verbose=0, warm_start=False)
# svm_classifier.fit(x_train, y_train)

# # Predicting the Test set results
# y_pred = svm_classifier.predict(x_test)

# cm = confusion_matrix(y_test, y_pred)
# np.mean(y_pred == y_test)

# print(f1_score(y_test,y_pred, average = 'macro'))
# print(precision_score(y_test,y_pred, average = 'macro'))
# print(recall_score(y_test,y_pred, average = 'macro'))

# #####grid search 
# parameters = [{'loss': ['hinge'], 'penalty': ['l1'], 'alpha': [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.1],
# 				'epsilon': [0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'learning_rate': ['optimal']}]

# grid_searches = []
# best_accuracy = []
# best_parameters = []

# for i in range(10):
# 	grid_search_item = GridSearchCV(estimator = svm_classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            n_jobs = -1)
# 	grid_search = grid_search_item.fit(x_train, y_train)
# 	print(type(grid_search))
# 	best_accuracy_item = grid_search.best_score_   
# 	best_parameters_item = grid_search.best_params_
# 	grid_searches.append(grid_search_item)
# #	np.vstack(best_accuracy, best_accuracy_item)
# 	best_parameters.append(best_parameters_item)
# 	best_accuracy.append(best_accuracy_item)

# best_accuracy
# best_parameters



###10 fold cross validation                        
#accuracies = cross_val_score(estimator = svm_classifier, X = x, y = y, cv = 10)
#accuracies.mean()  #0.63224855205962738
#0.60205396813929135


#####train and test set prediction 
# f1_score_list =[]
# precision_score_list =[]
# recall_score_list =[]

# for i in range(10):
# 	classifier = SGDClassifier(alpha=0.3, average=False, class_weight=None, epsilon=0.1,
#        	eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#        	learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
#        	n_jobs=1, penalty='l1', random_state=None,
#        	shuffle=True, tol=None, verbose=0, warm_start=False)
# 	classifier.fit(x_train, y_train)

# # Predicting the Test set results
# 	y_pred = classifier.predict(x_test)

# 	#cm = confusion_matrix(y_test, y_pred)
# #np.mean(y_pred == y_test)
# 	f1_score_item = f1_score(y_test,y_pred, average = 'macro')
# 	precision_score_item = precision_score(y_test,y_pred, average = 'macro')
# 	recall_score_item = recall_score(y_test,y_pred, average = 'macro')
# 	f1_score_list.append(f1_score_item)
# 	precision_score_list.append(precision_score_item)
# 	recall_score_list.append(recall_score_item)

# f1_score_list
# precision_score_list
# recall_score_list

# ####random search 
# parameters = {'loss': ['hinge'], 'penalty': ['l1'], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
# 				'epsilon': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}


# random_search = RandomizedSearchCV(estimator = svm_classifier, param_distributions=parameters,n_iter=20)
# random_search = random_search.fit(x_train, y_train)
# best_accuracy = random_search.best_score_   
# best_parameters = random_search.best_params_
# best_accuracy
# best_parameters


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
# 0.80294484466
# 0.806267343316
# 0.803242574257

#confusion matrix
###array([[151,  49],
#       [ 30, 172]])

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


##merge with lda
lda_500 = pd.read_csv('~/phd_work/crowdflower/lda_500/lda_500.csv')

status_sample = pd.read_csv('~/phd_work/crowdflower/status_sample.csv')
status_sample = pd.DataFrame(status_sample['text'])

lda_500 = pd.concat([status_sample,lda_500], axis =1)

all_fea_lda = pd.merge(data_dep, lda_500, left_on = 'text_y', right_on = 'text', how = 'left')

####select lda features and convert it to array
lda2 = all_fea_lda.loc[21:500]