import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Monthly sales data
months = [[1],[2],[3],[4],[5],[6]]
sales = [2000, 2500, 3000, 3500, 4000, 4500]

model = LinearRegression()
model.fit(months, sales)

# Predict next month
prediction = model.predict([[7]])
print("Predicted Sales for Month 7:", prediction[0])

# Plot
plt.scatter(months, sales)
plt.plot(months, model.predict(months))
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
