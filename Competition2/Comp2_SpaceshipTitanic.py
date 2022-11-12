# ----------------------------
# Andres Graterol 
# CAP5610 - Fall 22
# 4031393
# ----------------------------
# Spaceship Titanic Competition
# ----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

training_path = 'input/train.csv'
testing_path = 'input/test.csv'

training_set = pd.read_csv(training_path)
testing_set = pd.read_csv(testing_path)
combine = [training_set, testing_set]

# DATA PREPROCESSING
# =======================================
print(training_set.describe())
print('\n')
print(training_set.describe(include=['O']))

# Interpolate missing numerical values 
# Age
training_set['Age'] = training_set['Age'].interpolate()
testing_set['Age'] = testing_set['Age'].interpolate()
# RoomService 
training_set['RoomService'] = training_set['RoomService'].interpolate()
testing_set['RoomService'] = testing_set['RoomService'].interpolate()
# FoodCourt 
training_set['FoodCourt'] = training_set['FoodCourt'].interpolate()
testing_set['FoodCourt'] = testing_set['FoodCourt'].interpolate()
# ShoppingMall
training_set['ShoppingMall'] = training_set['ShoppingMall'].interpolate()
testing_set['ShoppingMall'] = testing_set['ShoppingMall'].interpolate()
# Spa
training_set['Spa'] = training_set['Spa'].interpolate()
testing_set['Spa'] = testing_set['Spa'].interpolate()
# VRDeck
training_set['VRDeck'] = training_set['VRDeck'].interpolate()
testing_set['VRDeck'] = testing_set['VRDeck'].interpolate()

combine = [training_set, testing_set]

# Fill missing categorical attributes with the mode of each feature
# CryoSleep
freq_CryoSleep = training_set.CryoSleep.dropna().mode()[0]

for dataset in combine:
    dataset['CryoSleep'] = dataset['CryoSleep'].fillna(freq_CryoSleep)

# VIP 
freq_VIP = training_set.VIP.dropna().mode()[0]

for dataset in combine:
    dataset['VIP'] = dataset['VIP'].fillna(freq_VIP)

# HomePlanet 
freq_HomePlanet = training_set.HomePlanet.dropna().mode()[0]

for dataset in combine:
    dataset['HomePlanet'] = dataset['HomePlanet'].fillna(freq_HomePlanet)

# Destination
freq_Destination = training_set.Destination.dropna().mode()[0]

for dataset in combine:
    dataset['Destination'] = dataset['Destination'].fillna(freq_Destination)

# Create new feature that is a combination of all monetary values
for dataset in combine:
    dataset['MoneySpent'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']

# Dropping all monetary values in favor of moneyspent
training_set = training_set.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
testing_set = testing_set.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)

print(training_set.describe())

# Data Analysis through pivoting features (Categorical Features)
# ------------------------------------------------------------
print('\n')
print(training_set[['VIP', 'Transported']].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False))
print('\n')
print(training_set[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=False).mean().sort_values(by='Transported', ascending=False))
print('\n')
print(training_set[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=False).mean().sort_values(by='Transported', ascending=False))
print('\n')
print(training_set[['Destination', 'Transported']].groupby(['Destination'], as_index=False).mean().sort_values(by='Transported', ascending=False))
print('\n')

combine = [training_set, testing_set]

# Convert the categorical titles to ordinal and map them into the datasets
# Home Planet and Destination 
homePlanet_mapping = {"Earth": 0, "Europa": 1, "Mars": 2}
destination_mapping = {"55 Cancri e": 0, "PSO J318.5-22": 1, "TRAPPIST-1e": 2}
for dataset in combine:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(homePlanet_mapping)
    dataset['Destination'] = dataset['Destination'].map(destination_mapping)

    dataset['HomePlanet'] = dataset['HomePlanet'].fillna(0)
    dataset['Destination'] = dataset['Destination'].fillna(0)

print('\n')
print(training_set.head())
print('\n')

# Data Analyis through visualization (numerical features)
# -----------------------------------------------------------
# Transported based on Age
transported_plot = sns.FacetGrid(training_set, col='Transported')
transported_plot.map(plt.hist, 'Age', bins=20)
#plt.show()

# Transported based on MoneySpent
transported_plot = sns.FacetGrid(training_set, col='Transported')
transported_plot.map(plt.hist, 'MoneySpent', bins=10)
#plt.show()

# Money spent vs VIP out of curiosity 
#money_plt = sns.FacetGrid(training_set, col='MoneySpent', size=2.2, aspect=1.6)
#money_plt.map(sns.barplot, 'VIP', alpha=.5, ci=None)
#plt.show()

# Transported based on VIP, HomePlanet
vip_homePlanet_grid = sns.FacetGrid(training_set, row='HomePlanet', size=2.2, aspect=1.6)
vip_homePlanet_grid.map(sns.barplot, 'VIP', 'Transported',  alpha=.5, ci=None)
vip_homePlanet_grid.add_legend()
#plt.show()

# Transported based on VIP, Destination 
vip_destination_grid = sns.FacetGrid(training_set, row='Destination', size=2.2, aspect=1.6)
vip_destination_grid.map(sns.barplot, 'VIP', 'Transported', alpha=.5, ci=None)
vip_destination_grid.add_legend()
#plt.show()

# Transported based on CryoSleep, HomePlanet 
cryoSleep_homePlanet_grid = sns.FacetGrid(training_set, row='HomePlanet', size=2.2, aspect=1.6)
cryoSleep_homePlanet_grid.map(sns.barplot, 'CryoSleep', 'Transported',  alpha=.5, ci=None)
cryoSleep_homePlanet_grid.add_legend()
#plt.show()

# Transported based on CryoSleep, Destination 
cryoSleep_destination_grid = sns.FacetGrid(training_set, row='Destination', size=2.2, aspect=1.6)
cryoSleep_destination_grid.map(sns.barplot, 'CryoSleep', 'Transported', alpha=.5, ci=None)
cryoSleep_destination_grid.add_legend()
#plt.show()

# Wrangle the rest of the data ----------------------
# Create age bands and determine correlation with survival 
training_set['AgeBand'] = pd.cut(training_set['Age'], 5)
print(training_set[['AgeBand', 'Transported']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
print('\n')

# Replacing age with ordinal age bands
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 47), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 47) & (dataset['Age'] <= 63), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 63, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)

# Now drop the age band feature
training_set = training_set.drop(['AgeBand'], axis=1)
combine = [training_set, testing_set]

training_set['SpentBand'] = pd.qcut(training_set['MoneySpent'], 4, duplicates='drop')
print(training_set[['SpentBand', 'Transported']].groupby(['SpentBand'], as_index=False).mean().sort_values(by='SpentBand', ascending=True))
print('\n')

# Convert the spentband feature to ordinal
for dataset in combine:
    dataset.loc[ dataset['MoneySpent'] <= 724.0, 'MoneySpent'] = 0
    dataset.loc[(dataset['MoneySpent'] > 724.0) & (dataset['MoneySpent'] <= 1503.0), 'MoneySpent'] = 1
    dataset.loc[ dataset['MoneySpent'] > 1503.0, 'MoneySpent'] = 2
    dataset['MoneySpent'] = dataset['MoneySpent'].astype(int)

# Now drop the spent band feature
training_set = training_set.drop(['SpentBand'], axis=1)
combine = [training_set, testing_set]

for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].str.strip().str[-1]
    dataset['Cabin'] = dataset['Cabin'].replace('P', 0)
    dataset['Cabin'] = dataset['Cabin'].replace('S', 1)

freq_cabin = training_set.Cabin.dropna().mode()[0]

# finding the majority for cabin side
print("Most frequent cabin:") 
print(freq_cabin)
print('\n')

# fill the null cabin values with the mode of cabin 
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].fillna(freq_cabin)
    dataset['Cabin'] = dataset['Cabin'].astype(int)

print(training_set[['Cabin', 'Transported']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Transported', ascending=False))
print('\n')

print(training_set.describe(include=['O']))
print('\n')

# Dropping Name 
training_set = training_set.drop(['Name', 'PassengerId'], axis=1)
testing_set = testing_set.drop(['Name'], axis=1)
combine = [training_set, testing_set]

print(training_set.head(10))
print('\n')
print(testing_set.head(10))
print('\n')

# Print out a list of all the columns and the amount of null values 
# if they still exist 
#print("Nulls in the training set:")
#print(training_set.isnull().sum())
#print('\n')
#print("Nulls in the testing set:")
#print(testing_set.isnull().sum())
#print('\n')

#print(training_set.head())
#print('\n')

# Model 
# ===============================================
X_train = training_set.drop("Transported", axis=1)
Y_train = training_set["Transported"]
X_test = testing_set.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)
print('\n')

# Logistic Regression 
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Modeling correlation (TODO: Note that this is different values)
coeff_df = pd.DataFrame(training_set.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))
print('\n')

# Support Vector Machines (TODO: Note that this is a different value)
svc = SVC()
svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
#Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Function to tune the hyperparameters of a model
def fine_tune(model, params):
    grid_search = GridSearchCV(estimator=model,
                                param_grid=params,
                                cv=4, 
                                scoring='accuracy')
    return grid_search

# Decision Tree
decision_tree = DecisionTreeClassifier()

# Parameters we want to cycle through for the decision tree
dt_params = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
}

dt_grid_search = fine_tune(decision_tree, dt_params)
dt_grid_search.fit(X_train, Y_train)

print("\nBest decision tree parameters:")
print(dt_grid_search.best_estimator_)

# Set the decision tree to the best parameters 
tuned_decision_tree = dt_grid_search.best_estimator_
tuned_decision_tree.fit(X_train, Y_train)
Y_pred = tuned_decision_tree.predict(X_test)
acc_decision_tree = round(tuned_decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier()

# Parameters we want to cycle through for the random forest
rf_params = {
    'min_samples_split': [3, 5, 10, 20],
    'max_depth': [3, 5, 10, 20],
    'min_samples_leaf': [3, 5, 10, 20]
}

rf_grid_search = fine_tune(random_forest, rf_params)
rf_grid_search.fit(X_train, Y_train)

print("\nBest random forest parameters:")
print(rf_grid_search.best_estimator_)
print('\n')

# Set the random forest model to the best parameters 
tuned_random_forest = rf_grid_search.best_estimator_
tuned_random_forest.fit(X_train, Y_train)
#Y_pred = tuned_random_forest.predict(X_test)
acc_random_forest = round(tuned_random_forest.score(X_train, Y_train) * 100, 2)

# These are the list of models that we considered 
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": testing_set["PassengerId"],
        "Transported": Y_pred
    })
submission.to_csv('comp2_submission.csv', index=False)