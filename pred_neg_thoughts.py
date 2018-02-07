import pandas as pd
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


data = pd.read_csv('a1228449.csv')
data2 = data[['negative_yn','thoughtcat','text']]

####clean data
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
        ps = PorterStemmer()
        review = [ps.stem(word) for word in w if not word in set(stopwords.words('english'))]
        new_words.append(w)
        
    return ' '.join(new_words)

data2['text'] = data2['text'].apply(preprocess)

# Creating the Bag of Words model

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(data2['text']).toarray()
y = data2.iloc[:, 0].values


#convert label to number 
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y.astype(str))
y[y == 0] = 2 #replace mix with yes

####convert to tfidf
tfidf_transformer = TfidfTransformer()
x = tfidf_transformer.fit_transform(x).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

######
from sklearn.naive_bayes import MultinomialNB
nclf = MultinomialNB().fit(x_train, y_train)
# Predicting the Test set results
y_pred2 = nclf.predict(x_test)

cm2 = confusion_matrix(y_test, y_pred)
np.mean(y_pred2 == y_test)
#0.58823529411764708

##################################################################
###combine with second sample
data_s2 = pd.read_csv('a1232881.csv')
data_s2 = data_s2[['negative_yn','thoughtcat','text']]
data_s2['text'] = data_s2['text'].apply(preprocess)

data3 = data2.append(data_s2)
data3 = data3.drop_duplicates(subset = ['text'],keep = 'last')

# Creating the Bag of Words model

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(data3['text']).toarray()
y = data3.iloc[:, 0].values


#convert label to number 
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y.astype(str))
y[y == 0] = 2 #replace mix with yes

####convert to tfidf
tfidf_transformer = TfidfTransformer()
x = tfidf_transformer.fit_transform(x).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)

#0.5

###################################
#combine only 'yes' label with the first sample

data_s2_y = data_s2[data_s2['negative_yn'] == 'no']


data4 = data2.append(data_s2_y)
data4 = data4.drop_duplicates(subset = ['text'],keep = 'last')

# Creating the Bag of Words model

cv = CountVectorizer()
x = cv.fit_transform(data4['text']).toarray()
y = data4.iloc[:, 0].values


#convert label to number 
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y.astype(str))
y[y == 0] = 2 #replace mix with yes

####convert to tfidf
tfidf_transformer = TfidfTransformer()
x = tfidf_transformer.fit_transform(x).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)

#0.58823529411764708

from sklearn.linear_model import SGDClassifier
###fit svm
classifier = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
np.mean(y_pred == y_test)

#0.68627450980392157

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()  #0.6120
accuracies.std()  #0.085



#grid search 1
parameters = [{'loss': ['hinge'], 'penalty': ['l1'], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
				'epsilon': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'loss': ['hinge'],'penalty': ['l2'], 'alpha': [0.00001, 0.0001, 0.001,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              'epsilon': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'loss': ['hinge'],'penalty': ['elasticnet'], 'alpha': [0.00001, 0.0001, 0.001,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              'epsilon': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_   #0.65024630541871919
best_parameters = grid_search.best_params_  #{'alpha': 0.01, 'epsilon': 0.9, 'loss': 'hinge', 'penalty': 'l1'}

##grid search 2

parameters = [{'loss': ['hinge'], 'penalty': ['l1'], 'alpha': [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.1],
				'epsilon': [0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'learning_rate': ['optimal']}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_   
best_parameters = grid_search.best_params_
best_accuracy
best_parameters









