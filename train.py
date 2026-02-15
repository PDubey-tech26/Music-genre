import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df=pd.read_csv("music_genre_dataset.csv")
print("Dataset loaded sucesssfully\n")
print(df.head())


numeric_cols=["tempo","energy","danceability","loudness","acousticness","instrumentalness"]
categorical_cols=["key","genre"]

scaler=MinMaxScaler()
df["tempo_scaled"]=scaler.fit_transform(df[["tempo"]])

label=LabelEncoder()
df["key_encoded"]=label.fit_transform(df["key"])
df["genre_encoded"]=label.fit_transform(df["genre"])


X = df.drop(columns=["tempo","key","genre"],axis=1)

print("\n After Processing:")
print(df.head())

X_train,X_test,y_train,y_test=train_test_split(X,df["genre_encoded"],test_size=0.2,random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))