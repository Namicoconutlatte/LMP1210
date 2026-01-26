#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


from pathlib import Path

csv_path = Path("C:/Users/kaihu/Downloads/HW1_data.csv")
print(csv_path.exists())


# In[3]:


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


# In[4]:


X_train, y_train, X_val, y_val, X_test, y_test = load_data(csv_path, random_state=0)

print("Train size:", X_train.shape[0])
print("Val size:  ", X_val.shape[0])
print("Test size: ", X_test.shape[0])

print("Train %:", X_train.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))
print("Val %:  ", X_val.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))
print("Test %: ", X_test.shape[0] / (X_train.shape[0] + X_val.shape[0] + X_test.shape[0]))


# In[8]:


def select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test, metric=None):
    train_acc = []
    val_acc = []
    for k in range(1, 21):  # Try k = 1 to 20
        if metric is None:
            model = KNeighborsClassifier(n_neighbors=k)
        else:
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)

        model.fit(X_train, y_train)

        train_acc.append(model.score(X_train, y_train))
        val_acc.append(model.score(X_val, y_val))

    # Plots showing the training and validation accuracy for each k
    plt.figure()
    plt.plot(range(1, 21), train_acc, label="Training accuracy")
    plt.plot(range(1, 21), val_acc, label="Validation accuracy")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")
    title = "KNN (default metric)" if metric is None else f"KNN (metric='{metric}')"
    plt.title(title)
    plt.legend()
    plt.show()

    best_k = val_acc.index(max(val_acc)) + 1 # Find best k (validation accuracy is highest)
    # Train the best model and test it
    if metric is None:
        best_model = KNeighborsClassifier(n_neighbors=best_k)
    else:
        best_model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)

    best_model.fit(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    return best_k, test_accuracy


# In[9]:


best_k_default, test_acc_default = select_knn_model(
    X_train, y_train, X_val, y_val, X_test, y_test
)

print("Best k (default metric):", best_k_default)
print("Test accuracy (default metric):", test_acc_default)


# In[11]:


def select_knn_model(X_train, y_train, X_val, y_val, X_test, y_test, metric=None):
    train_acc = []
    val_acc = []
    for k in range(1, 21):  # Try k = 1 to 20
        if metric is None:
            model = KNeighborsClassifier(n_neighbors=k)
        else:
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)

        model.fit(X_train, y_train)

        train_acc.append(model.score(X_train, y_train))
        val_acc.append(model.score(X_val, y_val))

    # Plots showing the training and validation accuracy for each k
    plt.figure()
    plt.plot(range(1, 21), train_acc, label="Training accuracy")
    plt.plot(range(1, 21), val_acc, label="Validation accuracy")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")
    title = "KNN (metric='cosine')" if metric is None else f"KNN (metric='{metric}')"
    plt.title(title)
    plt.legend()
    plt.show()

    best_k = val_acc.index(max(val_acc)) + 1 # Find best k (validation accuracy is highest)
    # Train the best model and test it
    if metric is None:
        best_model = KNeighborsClassifier(n_neighbors=best_k)
    else:
        best_model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)

    best_model.fit(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    return best_k, test_accuracy


# In[12]:


best_k_cosine, test_acc_cosine = select_knn_model(
    X_train, y_train, X_val, y_val, X_test, y_test, metric="cosine"
)

print("Best k (cosine):", best_k_cosine)
print("Test accuracy (cosine):", test_acc_cosine)


# In[ ]:




