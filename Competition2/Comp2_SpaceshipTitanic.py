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

print(training_set.describe(include=['O']))

