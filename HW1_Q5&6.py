#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source


# In[25]:


def load_data(csv_path, random_state=0):
    df = pd.read_csv(csv_path)
    df = df.fillna(0) # Fill missing values with 0
    y = df["Dataset"].astype(int) # 1 or 2
    X = df.drop(columns=["Dataset"]) # Features
    X = pd.get_dummies(X, drop_first=True) # Convert categorical variables to numeric columns

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )  # 20% test split
       # 10% of the total (separate from the 20% test split) or 0.1/(1-0.2) = 0.125 for validation; the rest is for training
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=random_state, stratify=y_trainval
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


# In[26]:


from pathlib import Path

csv_path = Path("C:/Users/kaihu/Downloads/HW1_data.csv")
print(csv_path.exists())


# In[27]:


X_train, y_train, X_val, y_val, X_test, y_test = load_data(csv_path, random_state=0)

print("Train size:", X_train.shape[0])
print("Val size:  ", X_val.shape[0])
print("Test size: ", X_test.shape[0])

print("Train %:", X_train.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))
print("Val %:  ", X_val.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))
print("Test %: ", X_test.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))


# In[9]:


# function train_decision_tree to classify between liver disease or no disease
def train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, min_samples_leaf):
    model = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    return model, train_acc, val_acc, test_acc


# In[28]:


# min_samples_leaf=1
model1, train1, val1, test1 = train_decision_tree(
    X_train, y_train, X_val, y_val, X_test, y_test, min_samples_leaf=1
)

print("Decision Tree (min_samples_leaf=1)")
print("Train accuracy:", train1)
print("Validation accuracy:  ", val1)
print("Test accuracy: ", test1)
print()


# In[29]:


# min_samples_leaf=2
model2, train2, val2, test2 = train_decision_tree(
    X_train, y_train, X_val, y_val, X_test, y_test, min_samples_leaf=2
)

print("Decision Tree (min_samples_leaf=2)")
print("Train accuracy:", train2)
print("Validation accuracy:  ", val2)
print("Test accuracy: ", test2)
print()


# In[12]:


# min_samples_leaf=3
model3, train3, val3, test3 = train_decision_tree(
    X_train, y_train, X_val, y_val, X_test, y_test, min_samples_leaf=3
)

print("Decision Tree (min_samples_leaf=3)")
print("Train accuracy:", train3)
print("Validation accuracy:  ", val3)
print("Test accuracy: ", test3)
print()


# In[30]:


# choose model 2 to visualize
model_to_plot = model2   
dot = export_graphviz(
    model_to_plot,
    out_file=None,
    feature_names=list(X_train.columns),
    filled=True,
    rounded=True,
    special_characters=True
)

graph = Source(dot)
graph.format = "png"
graph.render("dt", view=True)


# In[21]:


model_vis = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    min_samples_leaf=2,
    max_depth=2,          # for the first and second layers from the tree’s root of the selected model2
    random_state=0
)

model_vis.fit(X_train, y_train)


# In[31]:


dot = export_graphviz(
    model_vis,
    out_file=None,
    feature_names=list(X_train.columns),
    filled=True,
    rounded=True
)

graph = Source(dot)


# In[32]:


graph


# In[33]:


from sklearn.linear_model import LogisticRegression


# In[45]:


for iter in [50, 100, 300, 500, 1000]:
    lr = LogisticRegression(max_iter=iter, random_state=0)
    lr.fit(X_train, y_train)
    print(f"max_iter={iter}, n_iter_={lr.n_iter_}")


# In[43]:


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(max_iter=500, random_state=0)
# define function for question6
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc   = model.score(X_val, y_val)
    test_acc  = model.score(X_test, y_test)

    return model, train_acc, val_acc, test_acc


# In[44]:


log_model, train_lr, val_lr, test_lr = train_logistic_regression(
    X_train, y_train, X_val, y_val, X_test, y_test
)

print("Logistic Regression")
print("Train accuracy:", train_lr)
print("Validation accuracy:", val_lr)
print("Test accuracy:", test_lr)


# In[ ]:




