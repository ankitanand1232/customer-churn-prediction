import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['Churn'] = df['Churn'].str.lower().map({'yes': 1, 'no': 0})
df = df.drop('customerID', axis=1)
df = df.drop_duplicates()
x = df.drop('Churn', axis = 1)
y = df['Churn']
x = pd.get_dummies(x,drop_first=True,dtype= int)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print(x_train.shape, x_test.shape)
print(y_train.mean(), y_test.mean())
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
model2 = LogisticRegression(max_iter=1000, class_weight='balanced')
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)

print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))