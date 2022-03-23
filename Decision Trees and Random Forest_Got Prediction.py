## Decision Tree & Random Forest Project 
# Research Question: Can we predict whether a character will die judging by his/her personal data and background?

### Imports Library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

### Read the data and examine basic infomation of it
got = pd.read_csv('GOT.csv')
got.head()
got.info()
got.describe()

### Asses and clean the data
# 1. The statistical information of "age" is unreasonable. More spefically, the mean of age is -1293.56.
# (1) Assess the age column
print(got["age"].mean())
# (2) Check who has a negative age and its value.
print(got["name"][got["age"] < 0])
print(got['age'][got['age'] < 0])
# (3) Replace negative ages(I google how old Doreah and Rhaego should be)
got.loc[1684, "age"] = 24.0
got.loc[1868, "age"] = 0.0
# (4) Check the mean again, and it goes back to reasonable range now.(36.7)
print(got["age"].mean())

# 2. Check how many null values in each variable
nans = got.isna().sum()
nans
# We need to replace these NAN values, but—
# Removing them directly will cause some problems, since nan values take a big portion in some columns. 
# 13 of them have over 1000 null values. Therefore we need to replace them with proper values meaningful to our model.
# (1) For "age" column, it's reasonable to fill the average value in.
# (2) For other categorical values, I'll try to replace them with "-1"
got["age"].fillna(got["age"].mean(), inplace=True)
got.fillna(value=-1, inplace=True)


### Prepare for Decision Tree Model
# 1. Data Visualization :
sns.countplot(x='isAlive', data=got, palette='icefire', alpha = 0.9)
plt.title('Counts of Charachers Alive')
rate = (got['isAlive'].value_counts()/len(got['isAlive']))*100
print (rate)
# Take a brief overview of our targeted value. More characters are alive in the end(about 75%).

# 2. Here I would like to obersve: Is it possible that a character's popularity could affect their fate? 
# Maybe the screenwriter would refer to the popularity and decide their end?
plt.figure(figsize=(10,6))
got[got['isAlive']==1]['popularity'].hist(alpha=0.5,color='blue',bins=30,label='isAlive=1')
got[got['isAlive']==0]['popularity'].hist(alpha=0.5,color='red',bins=30,label='isAlive=0')
plt.legend()
plt.xlabel('popularity')
plt.ylabel('count')
plt.title('Popularity vs Alive')
# Result: Let's zoom in to see the condition of population below 0.4, since the volume of >0.4 is too small to compare.
got[got['isAlive']==1]['popularity'].hist(alpha=0.5,color='blue',bins=30,label='isAlive=1').set_xlim([0,0.4])
got[got['isAlive']==0]['popularity'].hist(alpha=0.5,color='red',bins=30,label='isAlive=0').set_xlim([0,0.4])
plt.legend()
plt.xlabel('popularity')
plt.ylabel('count')
plt.title('Popularity vs Alive')
# Result: Among those who survive, unpopular characters takes a large portion.
# The trend is different from those who die in the end(isAlive=0), where unpopular characters don't takes such large portion.
# From my perspective, the difference of the trend may mean that this variable could help the model training.

# 3. Here I want to check the correlation between gender and their rate of death
sns.countplot(x='gender', data=got, hue='isAlive', palette='crest', alpha = 0.8)
plt.title('Gender vs Alive')
# Result: The rate of survival is slightly different for each gender, so I would suspect gender also helps for predicting.

# 4. isPopular v.s. isAlive
sns.countplot(x='isPopular', data=got, hue='isAlive', palette='crest', alpha = 0.8)
plt.title('isPopular vs Alive')
# Result: Although the variable "isPopular" is based on the other variable "popularity" (threshold: 0.34), I'm surprised 
# that the rate of survival is quite different for popular and (relatively) unpopular groups.
# Let me make a bold guess: Maybe making popular characters die would be more dramatic and trigger heated debates?

# 5. Age v.s. isAlive
got_forplot = pd.read_csv('GOT.csv')
got_forplot = got_forplot.loc[got_forplot['age'] > 0]
plt.figure(figsize=(10,6))
got_forplot[got_forplot['isAlive']==1]['age'].hist(alpha=0.5,color='blue',bins=30,label='isAlive=1')
got_forplot[got_forplot['isAlive']==0]['age'].hist(alpha=0.5,color='red',bins=30,label='isAlive=0')
plt.legend()
plt.xlabel('age')
plt.ylabel('count')
plt.title('Age vs Alive')
# In normal condition(reality), there should be a positive correlation between age and the death rate.
# But based on the result, among all the dead characters, aging is not necessarily why they die. Battles and murders could be the reasons especially in Game of Thrones.

### Data Cleaning II 
# After browsing through all the variables, I think some variables should be removed first due to below reasons:
# 1. 'S.No', 'name': It's basically like IDs, each variable has its unique value, which won't help the classification.
# 2. 'spouse','heir': For similar reason, these two would be "nearly" distinct value for everyone(if they have one). For marriage status, we have "isMarried" as the parameter.
# 3. 'plod','DateoFdeath': Putting the probability of death and date of death into the model is like cheating when we want to predict death. So I remove them as well.
# Remove variables mentioned aboove
drop = (['S.No', 'plod', 'name', 'DateoFdeath','spouse','heir'])
got.drop(drop, inplace=True, axis=1)
# Next I need to transform categorical data into dummy variables, so that model can take those in training properly.
got = pd.get_dummies(got,drop_first=True)
# After data cleaning, this is what new data looks like:
got.head()

### Build, run and evaluate the model
# 1-1. Build the Decision Tree model:
# Split the data into a training set and a testing set with test size 30%
from sklearn.model_selection import train_test_split
X = got.drop(columns = ['isAlive'],axis = 1)
y = got['isAlive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
# Build up a tree model and fit it with training sets
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
# Make prediction with the model
tree_prediction = tree.predict(X_test)

# 1-2. Evaluate the Tree model with classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,tree_prediction))
# It looks like the model can predict better in "true negative", which refers to who can survive in the end.
print(classification_report(y_test,tree_prediction))
# The overall accuracy is around 78%
# For accuracy, there's room for improvement. Next I'll try the random forest model, which uses bagging method and decorrelates trees.


# 2-1. Build the Random Forest model:
# Build up a random forest model and fit it with training sets
# And I'll set n_estimators=200 initially.
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(n_estimators=200)
RF.fit(X_train, y_train)
RF_prediction1 = RF.predict(X_test)

# 2-2. Evaluate the Random Forest model with classification report and confusion matrix
print(confusion_matrix(y_test,RF_prediction1))
print(classification_report(y_test,RF_prediction1))
# With random forest model, the accuracy is slightly improved (78% -> 80%)
# I would like to see if changing the number of trees(n_estimators) in the forest would help, so I'll adjust from 200 to 500.
RF= RandomForestClassifier(n_estimators=500)
RF.fit(X_train, y_train)
RF_prediction2 = RF.predict(X_test)
print(confusion_matrix(y_test,RF_prediction2))
print(classification_report(y_test,RF_prediction2))
# With n_estimator=500, though not much, the model improves in several espects, including 
# precision of predicting the death, recall of predicting survival, f1 score of predicting death, and finally, overall accuracy's improved slightly as well.
# Seeing this uptrend, I want to try increasing the n_estimator again.
RF= RandomForestClassifier(n_estimators=1000)
RF.fit(X_train, y_train)
RF_prediction3 = RF.predict(X_test)
print(confusion_matrix(y_test,RF_prediction3))
print(classification_report(y_test,RF_prediction3))
# The accuracy remains unchanged, but some metrics slightly get worse, including precision and f1-score of predicting death. Therefore, I think I should stop adding trees.
# I'll create some visualizations showing the findings.
# 1
n_estimator = [200, 500, 1000]
accuracy = [0.80, 0.81, 0.81]
accu_perf = {'n_estimator': [200, 500, 1000], 'accuracy': [0.80, 0.81, 0.81]}
df1 = pd.DataFrame(data = accu_perf)
sns.barplot(x='n_estimator',y='accuracy',data=df1, palette = 'crest')
plt.title('n_estimator vs Accuracy')
# 2
est200_perf = {'is_Alive' : [0, 1], 'precision': [0.72, 0.82], 'recall': [0.39, 0.95], 'f1_score' : [0.50, 0.88]}
df2 = pd.DataFrame(data = est200_perf)
est500_perf = {'is_Alive' : [0, 1], 'precision': [0.74, 0.82], 'recall': [0.40, 0.95], 'f1_score' : [0.52, 0.88]}
df3 = pd.DataFrame(data = est500_perf)
est1000_perf = {'is_Alive' : [0, 1], 'precision': [0.72, 0.82], 'recall': [0.39, 0.95], 'f1_score' : [0.51, 0.88]}
df4 = pd.DataFrame(data = est1000_perf)
# n_estimator=200
df2[['precision','recall','f1_score']].plot.bar(cmap='crest')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))
plt.xlabel('Alive or not')
plt.ylabel('rate')
plt.title('n_estimator=200')
# n_estimator=500
df3[['precision','recall','f1_score']].plot.bar(cmap='crest')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))
plt.xlabel('Alive or not')
plt.ylabel('rate')
plt.title('n_estimator=500')
# n_estimator=1000
df4[['precision','recall','f1_score']].plot.bar(cmap='crest')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))
plt.xlabel('Alive or not')
plt.ylabel('rate')
plt.title('n_estimator=1000')

## Brief Conclusion: Putting the visualization together and examining performane dataframe again, 
# I found that the model performance of different n_estimator actually didn't vary a lot. Next, I want to see if I can improve model performance by adjusting variables
# Review again about the features
got.head()

# 1. I want to try removing "dateOfBirth" because everyone's birthday is almost unique and may not be a strong factor for classification. 
# Also I want to remove "isPopular" because we have "popularity" variable, which is more precise. 
nX = got.drop(columns = ['isAlive','dateOfBirth','isPopular'],axis = 1)
ny = got['isAlive']
nX_train, nX_test, ny_train, ny_test = train_test_split(nX, ny, test_size=0.30, random_state=101)
# with best n_estimator -> 500
RF= RandomForestClassifier(n_estimators=500)
RF.fit(nX_train, ny_train)
RF_prediction4 = RF.predict(nX_test)
print(confusion_matrix(ny_test,RF_prediction4))
print(classification_report(ny_test,RF_prediction4))
# For accuracy, it didn't get significantly better or worse(80%). 
# I would like to check whether the choice of variables really matters, so I'll drop different variables this time.
# I want to try removing "book1"~"book5", maybe it doesn't matter which book each character appear when predicting their death. 
nX = got.drop(columns = ['isAlive','dateOfBirth','book1','book2','book3','book4','book5'],axis = 1)
ny = got['isAlive']
nX_train, nX_test, ny_train, ny_test = train_test_split(nX, ny, test_size=0.30, random_state=101)
RF= RandomForestClassifier(n_estimators=500)
RF.fit(nX_train, ny_train)
RF_prediction5 = RF.predict(nX_test)
print(confusion_matrix(ny_test,RF_prediction5))
print(classification_report(ny_test,RF_prediction5))
## Brief Conclusion: I think the choice of variables do affect characters' death. Because in my second adjustment, performance goes down relatively significant. 
# The problem is-- which variables should be dropped? It's hard to choose only by human intuition.








