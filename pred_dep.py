# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('status_dep_demog_big5_schwartz_swl_like.csv')

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

data = pd.read_csv('a1228449.csv')
data2 = data[['negative_yn','thoughtcat','text']]
data_s2 = pd.read_csv('a1232881.csv')
data_s2 = data_s2[['negative_yn','thoughtcat','text']]
data3 = data2.append(data_s2)
data3 = data3.drop_duplicates(subset = ['text'],keep = 'last')

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

data3['text'] = data3['text'].apply(preprocess)


all_data = merge(dataset, data3, on = 'userid', how = 'inner')


#####convert categorical data 
#select useful variables in dataset
selected = ['userid', 'marital_status', 'ethnicity', 'gender','age','relationship_status', 'network_size', 'ope','con','ext',
'agr','neu','swl']
data = dataset[selected]
data_dep = pd.merge(data, dep_df, on='userid', how = 'inner')


data_dep['ethnicity'] = data_dep['ethnicity'].fillna('Other')
data_dep['marital_status'] = data_dep['marital_status'].fillna('Other')
data_dep['age'] = data_dep['age'].fillna(data_dep['age'].mean())
data_dep['relationship_status'] = data_dep['relationship_status'].fillna(0)
data_dep['network_size'] = data_dep['network_size'].fillna(data_dep['network_size'].mean())

x = data_dep.iloc[:, :-1].values
y = data_dep.iloc[:, -1].values
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

