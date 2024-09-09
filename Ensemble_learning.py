# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:46:09 2024

@author: EliteBook 840
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import classification_report

df_leor = pd.read_csv("E:/Ensemble_learning/pima-indians-diabetes.csv")

# Add column names
column_names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
df_leor.columns = column_names

# Display the first few rows of the DataFrame
print(df_leor.head())

print(df_leor.dtypes)
print(df_leor.isnull().sum())
print(df_leor.describe())
categorical_columns = [col for col in df_leor.columns if df_leor[col].dtype == 'object']
print(categorical_columns)
print(df_leor['class'].value_counts())


# Prepare a standard scaler transformer
transformer_leor = StandardScaler()

# Split features (X) and class (y)
X = df_leor.drop('class', axis=1)
y = df_leor['class']

# Split data into train and test sets
X_train_leor, X_test_leor, y_train_leor, y_test_leor = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit and transform the transformer to the training features
X_train_leor_scaled = transformer_leor.fit_transform(X_train_leor)

# Transform the test features using the fitted transformer
X_test_leor_scaled = transformer_leor.transform(X_test_leor)


# Define classifiers with names based on your instructions
logreg_G = LogisticRegression(max_iter=1400)
rfc_G = RandomForestClassifier()
svc_G = SVC()
dtc_G = DecisionTreeClassifier(criterion="entropy", max_depth=42)
etc_G = ExtraTreesClassifier()

# Define a voting classifier with all classifiers as estimators
voting_classifier = VotingClassifier(estimators=[
    ('logreg_G', logreg_G),
    ('rfc_G', rfc_G),
    ('svc_G', svc_G),
    ('dtc_G', dtc_G),
    ('etc_G', etc_G)
], voting='hard')

# Fit the training data to the voting classifier
voting_classifier.fit(X_train_leor_scaled, y_train_leor)

# Predict the first three instances of test data
predictions = voting_classifier.predict(X_test_leor_scaled[:3])

# Print out predicted and actual values for each classifier and instance
for classifier in [logreg_G, rfc_G, svc_G, dtc_G, etc_G, voting_classifier]:
    print(f"Classifier: {classifier.__class__.__name__} hard voting")
    print("Predictions for the first three instances of test data:")
    for i, prediction in enumerate(predictions):
        print(f"Instance {i+1}: Predicted - {prediction}, Actual - {y_test_leor.iloc[i]}")
    print()
    
    
    
# Define classifiers with names based on your instructions
logreg_X1 = LogisticRegression(max_iter=1400)
rfc_X1 = RandomForestClassifier(n_estimators=100)  # Changed to adjust for soft voting
svc_X1 = SVC(probability=True)  # Added probability=True for soft voting
dtc_X1 = DecisionTreeClassifier(criterion="entropy", max_depth=42)
etc_X1 = ExtraTreesClassifier(n_estimators=100)  # Changed to adjust for soft voting

# Define a voting classifier with all classifiers as estimators
voting_classifier_soft = VotingClassifier(estimators=[
    ('logreg_X', logreg_X1),
    ('rfc_X', rfc_X1),
    ('svc_X', svc_X1),
    ('dtc_X', dtc_X1),
    ('etc_X', etc_X1)
], voting='soft')

# Fit the training data to the voting classifier
voting_classifier_soft.fit(X_train_leor_scaled, y_train_leor)

# Predict the first three instances of test data
predictions_soft = voting_classifier_soft.predict(X_test_leor_scaled[:3])

# Print out predicted and actual values for each classifier and instance
for classifier in [logreg_X1, rfc_X1, svc_X1, dtc_X1, etc_X1, voting_classifier_soft]:
    print(f"Classifier: {classifier.__class__.__name__} soft voting")
    print("Predictions for the first three instances of test data:")
    for i, prediction in enumerate(predictions_soft):
        print(f"Instance {i+1}: Predicted - {prediction}, Actual - {y_test_leor.iloc[i]}")
    print()
    
# Define a standard scaler transformer
transformer_leor = StandardScaler()

# Define Pipeline #1
pipeline1_leor = Pipeline([
    ('transformer_leor', transformer_leor),  # Using the transformer prepared in step 4
    ('etc_G', etc_G)  # Using the Extra Trees Classifier prepared in step 8.e
])

# Define Pipeline #2
pipeline2_leor = Pipeline([
    ('transformer_leor', transformer_leor),  # Using the transformer prepared in step 4
    ('dtc_G', dtc_G)  # Using the Decision Tree Classifier prepared in step 8.d
])

# Fit the original data to both pipelines
pipeline1_leor.fit(X, y)
pipeline2_leor.fit(X, y)

# Define StratifiedKFold with shuffle=True and random_state=42
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Carry out a 10 fold cross validation for both pipelines using StratifiedKFold
cv_scores_pipeline1 = cross_val_score(pipeline1_leor, X, y, cv=skf, scoring='accuracy')
cv_scores_pipeline2 = cross_val_score(pipeline2_leor, X, y, cv=skf, scoring='accuracy')

# Printout the mean score evaluation for both pipelines
print("Mean score evaluation for Pipeline #1:", cv_scores_pipeline1.mean())
print("Mean score evaluation for Pipeline #2:", cv_scores_pipeline2.mean())

# Predict the test using both pipelines and printout the confusion matrix, precision, recall, and accuracy scores
pipelines = [pipeline1_leor, pipeline2_leor]
pipeline_names = ["Pipeline #1", "Pipeline #2"]

for pipeline, name in zip(pipelines, pipeline_names):
    print(f"\n{50*'-'}\n{name}\n{50*'-'}")
    
    # Predict test data
    y_pred = pipeline.predict(X_test_leor_scaled)
    
    # Calculate and print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_leor, y_pred))
    
    # Calculate and print precision, recall, and accuracy scores
    precision = precision_score(y_test_leor, y_pred)
    recall = recall_score(y_test_leor, y_pred)
    accuracy = accuracy_score(y_test_leor, y_pred)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
    
    
param_1 = {
    'etc_G__n_estimators': np.arange(10, 3001, 20),
    'etc_G__max_depth': np.arange(1, 1001, 2)
}

pipe_1_gridsearch_36 = RandomizedSearchCV(estimator=pipeline1_leor, param_distributions=param_1, 
                                          n_iter=10, cv=5, scoring='accuracy', refit=True, verbose=3, 
                                          random_state=42, n_jobs=-1)
pipe_1_gridsearch_36.fit(X_train_leor, y_train_leor)

# Print out the best parameters and accuracy score
print("Best parameters found:")
print(pipe_1_gridsearch_36.best_params_)
print("Best accuracy score found:", pipe_1_gridsearch_36.best_score_)

# Predict on the test set
y_pred = pipe_1_gridsearch_36.predict(X_test_leor)

# Print precision, recall, and accuracy
print("\nClassification Report:")
print(classification_report(y_test_leor, y_pred))