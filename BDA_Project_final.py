import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer 
# splitting the data set into training set and test set 
from sklearn.model_selection import train_test_split
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
#from sklearn.model_selection import GridSearchCV

#load dataset
data = pd.read_csv('F:\study/BDA/project/bdaproject/BDA project/Datasets/Edited-winemag-data-130k-v2.csv',sep=',')

#handle varieties with special characters 
data = data.loc[data['variety'].str.contains(r'[^\x00-\x7F]+') == False]

#creating subset with sufficient samples for countries and variety 
data = data.groupby('country').filter(lambda x: len(x) >3764)
data = data.groupby('variety').filter(lambda x: len(x) >5000).reset_index()

#vectorize country
df_country = data.country
country = ['US','Italy','France','Spain','Portugal']
k=0
for i in country:
    df_country=df_country.replace(i, k)
    k=k+1

#vectorize variety
df_variety=data.variety
variety_names=["Pinot Noir","Chardonnay","Cabernet Sauvignon","Red Blend","Bordeaux-style Red Blend","Riesling","Sauvignon Blanc","Syrah","RosÃ©","Merlot","Zinfandel","Malbec","Sangiovese","Nebbiolo","Portuguese Red"]
k=0
for i in variety_names:
    df_variety=df_variety.replace(i, k)
    k=k+1


#vectorize points
df_review = data.points
for i in range(80,101):
    if(i<=85):
        df_review=df_review.replace(i,'Bad')
    elif(i<=90):
        df_review=df_review.replace(i,'Good')
    elif(i<=95):
        df_review=df_review.replace(i,'Better')
    else:
        df_review=df_review.replace(i,'Best')

#kernel
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['description'].values
print("data variety:",data['variety'].count())
print("description count:",data['description'].count())


stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

# creating bag of words model 
cv = CountVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 3000) 
X = cv.fit_transform(desc).toarray() 
#cv._validate_vocabulary()
#print('loaded_vectorizer.get_feature_names(): {0}'.format(cv.get_feature_names()))
X = pd.DataFrame(X)
print(X)

df_gth = data.filter(['price'], axis=1)
df_gth['Variety']=df_variety
df_gth['Country']=df_country

df_gth = pd.concat([df_gth, X], axis=1, sort=False)
y = df_review.values 
print(df_gth.head(5))
print(df_gth.shape)

def classify(clf,X_df, dem, color):
    a = np.zeros(shape=(6))
    px=[10,30,50,70,90]
    
    k=0
    #spilt train-test data
    for i in px:    
        X_train, X_test, y_train, y_test = train_test_split( 
               X_df, y, test_size = i/100)
        
        if(clf=='knn'):
            #KNN
            '''
            model = KNeighborsClassifier(n_neighbors=4)
            #Hyper Parameters Set
            params = {'n_neighbors':[4],
                      'leaf_size':[1,2,3,5],
                      'weights':['uniform', 'distance'],
                      'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                      'n_jobs':[-1]}
            #Making models with hyper parameters sets
            knn = GridSearchCV(model, param_grid=params, n_jobs=1)
            '''
            knn = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors=4, weights = 'uniform')            
            #Train the model using the training sets
            knn.fit(X_train, y_train)
            '''
            #The best hyper parameters set
            print("Best Hyper Parameters:\n",knn.best_params_)
            '''
            #Predict the response for test dataset
            y_pred = knn.predict(X_test)
        elif(clf=='nb'):
            #naive Bayes
            classifier = GaussianNB(); 
            classifier.fit(X_train, y_train) 
              
            # predicting test set results 
            y_pred = classifier.predict(X_test) 
        elif(clf=='svm'):
            #SVM
            clf = SVC(C=2.5, gamma='auto', kernel='rbf')
            y_pred = clf.fit(X_train, y_train).predict(X_test)
        else:
            print('not valid classifier')
              
        # making the confusion matrix 
        cm = confusion_matrix(y_test, y_pred) 
        print(cm) 
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy_score: ", accuracy)
        
        a[k]=accuracy
        k=k+1
    #set plot features    
    if(dem == 'original'):
        linestyle = 'solid'
    elif(dem == 'PCA'):
        linestyle = 'dotted'
    else:
        linestyle = 'dashed'
    plt.scatter(px,a[0:5])
    plt.plot(px,a[0:5],marker='',linestyle = linestyle, markersize=12, color=color, linewidth=4)     


classify('knn',df_gth, 'original', 'blue')
classify('nb',df_gth, 'original', 'green')
classify('svm',df_gth, 'original', 'aqua')


#PCA
pca = PCA(n_components=2)
X_pca = pca.fit(df_gth).transform(df_gth)       
# Percentage of variance explained for each components
print('Explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
classify('knn',X_pca, 'PCA','blue')
classify('nb',X_pca, 'PCA','green')
classify('svm',X_pca, 'PCA','aqua')

#LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(df_gth, y).transform(df_gth)
classify('knn',X_lda, 'LDA', 'blue')
classify('nb',X_lda, 'LDA','green')
classify('svm',X_lda, 'LDA','aqua')

plt.show()
