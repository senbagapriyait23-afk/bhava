import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv') # Load Data
print("Original Data Loaded:", df.shape)

df = df.drop_duplicates()  # Data Cleaning
df = df.fillna(df.mean(numeric_only=True))
print("After Cleaning:", df.shape)

corr = df.corr()['target'].abs() # Feature Selection (keep only useful features)
features = corr[corr>0.1].index.drop('target')
print("co-relations:\n",corr,"\nSelected Features:",list(features))

x = df[features] # Split Data for training
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear') #  Train & Evaluate
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

data_df=pd.read_csv('test_data.csv') #prediction
x=data_df.drop(columns=['target'])
y_Pred=model.predict(x)
print(y_Pred)