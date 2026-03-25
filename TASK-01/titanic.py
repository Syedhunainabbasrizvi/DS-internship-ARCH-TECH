import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#Load dataset
df = pd.read_csv("train.csv")
print("First 5 rows of dataset:")
print(df.head())

#Check missing values
print("\nMissing values:")
print(df.isnull().sum())

#Data pre-processing
df = df.copy()
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

#Features & target
features = ['Pclass', 'Sex', 'Age', 'Fare']
if 'Embarked_C' in df.columns and 'Embarked_Q' in df.columns:
    features += ['Embarked_C', 'Embarked_Q']
X = df[features].fillna(df[features].mean())
y = df['Survived']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Predict & evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)