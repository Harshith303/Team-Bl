import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample marketing data
data = {
    'Website_Visits': [5, 10, 3, 20, 15, 7, 12, 25],
    'Time_Spent': [2, 5, 1, 10, 8, 3, 6, 12],
    'Email_Clicks': [1, 3, 0, 5, 4, 1, 3, 6],
    'Purchased': [0, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Website_Visits', 'Time_Spent', 'Email_Clicks']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# Predict new customer
new_customer = [[18, 9, 4]]
result = model.predict(new_customer)

if result[0] == 1:
    print("High chance of purchase")
else:
    print("Low chance of purchase")
