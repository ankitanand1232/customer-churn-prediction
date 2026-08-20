import pandas as pd
df = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
g = df[['gender']]
print(g)
