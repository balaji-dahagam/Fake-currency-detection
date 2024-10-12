import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('C:\\Users\\Aishu\\OneDrive\\mllab\\Fake_currency_project\\data_banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
print(data.head())

sns.pairplot(data, hue='auth')
plt.show()

plt.figure(figsize=(8, 6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=data['auth'])
target_count = data.auth.value_counts()
x = 29
plt.annotate(text=str(target_count[0]), xy=(-0.04, 10 + target_count[0]), size=14)
plt.annotate(text=str(target_count[1]), xy=(0.96, 10 + target_count[1]), size=14)
plt.ylim(0, 900)
plt.show()

nb_to_delete = target_count[0] - target_count[1]
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
data = data[nb_to_delete:]
print(data['auth'].value_counts())

plt.figure(figsize=(8, 6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=data['auth'])
target_count = data.auth.value_counts()
plt.annotate(text=str(target_count[0]), xy=(-0.04, 10 + target_count[0]), size=14)
plt.annotate(text=str(target_count[1]), xy=(0.96, 10 + target_count[1]), size=14)
plt.ylim(0, 900)
plt.show()

from sklearn.metrics import accuracy_score
from sklearnex import patch_sklearn
patch_sklearn()

x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)
scalar = StandardScaler()
x = scalar.fit_transform(x)

pipe = Pipeline([('classifier', LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42))])

param_grid = {
    'classifier__C': [0.1, 1, 10],
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x, y.values.ravel())

print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
print(grid_search.best_params_)

# Evaluate the model using cross-validation
scores = cross_val_score(grid_search.best_estimator_, x, y.values.ravel(), cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Print confusion matrix and accuracy on the test set
clf = grid_search.best_estimator_
clf.fit(x_train, y_train.values.ravel())
y_pred = clf.predict(x_test)
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                        columns=["Pred.Negative", "Pred.Positive"],
                        index=['Act.Negative', "Act.Positive"])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = round((tn + tp) / (tn + fp + fn + tp), 4)
print(conf_mat)
Accuracy = {round(100 * accuracy, 2)}
print("Accuracy of Logistic regression Model is: " + str(Accuracy) + " %")

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearnex import patch_sklearn
patch_sklearn()
X = data.drop('auth', axis=1)
y = data['auth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = DecisionTreeClassifier(random_state=42)

param_grid = {'max_depth': [2, 4, 6, 8, 10]}

grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
print(grid_search.best_params_)

# Evaluate the model using cross-validation
scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Print confusion matrix and accuracy on the test set
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                        columns=["Pred.Negative", "Pred.Positive"],
                        index=['Act.Negative', "Act.Positive"])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
dec_accuracy = round((tn + tp) / (tn + fp + fn + tp), 4)
print(conf_mat)
dec_Accuracy = {round(100 * dec_accuracy, 2)}
print("Accuracy of Decision Tree Model is: " + str(dec_Accuracy) + " %")

scores = [accuracy, dec_accuracy]
algorithms = ["Logistic Regression", "Decision Tree"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str((scores[i] * 100)) + " %")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(algorithms, scores, align='center', color=['blue', 'green'])
ax.set_xlabel('Accuracy')
ax.set_ylabel('Algorithm')
ax.set_title('Comparison of Algorithm Accuracies')

# Add the score values to the plot
for i, v in enumerate(scores):
    ax.text(v + 0.01, i, str(round(v, 5)), color='black', fontweight='bold')

plt.show()

new_banknote = np.array([4.5, -8.1, 2.4, 1.4], ndmin=2)
new_banknote = scalar.transform(new_banknote)
print(f'Prediction:  Class{clf.predict(new_banknote)[0]}')
print(f'Probability [0/1]:  {clf.predict_proba(new_banknote)[0]}')