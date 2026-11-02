import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef

# load the dataset using a direct download link
url = "https://drive.google.com/uc?export=download&id=1CWIQZPkZAoZeONqqb5trjf3poYqoHypH"

# use pandas to read the csv file into a table format
train = pd.read_csv(url)

# separate the clues (features) from the answer we want to guess (name)
x = train.drop(columns=['name'])
y = train['name']

# set up the decision tree with the best rules from both approaches
dt = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

# set up a fair splitting method for our data (5 parts total)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# create empty lists to write down our test scores
acc_scores = []
rec_scores = []
f1_scores = []
mcc_scores = []

# loop through each of the 5 fair splits to train and test
for train_index, test_index in skf.split(x, y):

    # grab the training data and testing data for this specific round
    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    
    # teach the decision tree using the training data
    dt.fit(x_train, y_train)
    
    # ask the tree to guess the answers for our test data
    y_pred = dt.predict(x_test)
    
    # grade the tree's guesses and save the scores
    acc_scores.append(accuracy_score(y_test, y_pred))
    rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    mcc_scores.append(matthews_corrcoef(y_test, y_pred))

# print out the final average report card
print("--- average performance across 5 folds ---")
print(f"accuracy: {np.mean(acc_scores):.4f}")
print(f"recall:   {np.mean(rec_scores):.4f}")
print(f"f1 score: {np.mean(f1_scores):.4f}")
print(f"mcc:      {np.mean(mcc_scores):.4f}\n")

# teach the tree one last time using all the data so we can see its rules
dt.fit(x, y)

# print the flowchart rules in plain text
print("--- decision tree rules ---")
print(export_text(dt, feature_names=list(x.columns)))