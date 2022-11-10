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

training_path = 'input/train.csv'
testing_path = 'input/test.csv'

training_set = pd.read_csv(training_path)
testing_set = pd.read_csv(testing_path)
combine = [training_set, testing_set]

# DATA PREPROCESSING
# =======================================

# Display the data type for every feature 
#training_set.info()
#print('_'*40)
#testing_set.info()

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

# Drop rows that we cannot fill in through statistical methods
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].dropna()
    dataset['Name'] = dataset['Name'].dropna()