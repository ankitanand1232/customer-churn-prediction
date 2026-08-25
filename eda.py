import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['Churn'] = df['Churn'].str.lower()
df['Churn'] = df['Churn'].map({'yes':1,'no':0})
df = df.drop('customerID',axis = 1)
df = df.drop_duplicates()


plt.figure()
sns.countplot(x = 'Contract',hue = 'Churn',data = df)
plt.title('Churn by Contract Type')

plt.figure()
sns.histplot(data = df,x = 'tenure',hue = 'Churn',multiple='stack',bins = 30)
plt.title('Churn by Tenure')

plt.figure()
sns.histplot(data = df,x = 'MonthlyCharges',hue = 'Churn',multiple='stack',bins = 30)
plt.title('Churn by Monthly Charges')

plt.figure(figsize=(8,6))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.show()