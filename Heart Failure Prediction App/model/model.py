import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the heart failure dataset
heart_data = pd.read_csv('heart.csv')

# No need for encoding as the data is already numeric

# Separating X and y
X = heart_data.drop('output', axis=1)
Y = heart_data['output']

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('heart_failure_clf.pkl', 'wb'))
