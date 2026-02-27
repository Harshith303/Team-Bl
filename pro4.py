import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# -----------------------------
# STEP 1: Train Sales Prediction Model
# -----------------------------

sales_data = {
    "Month": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Festival": [0,0,0,0,0,0,1,0,1,0,1,1],
    "Sales": [2000,2200,2100,2300,2400,2500,4000,2600,4500,2700,5000,5200]
}

sales_df = pd.DataFrame(sales_data)

X_sales = sales_df[["Month","Festival"]]
y_sales = sales_df["Sales"]

sales_model = LinearRegression()
sales_model.fit(X_sales, y_sales)

# -----------------------------
# STEP 2: Train Lead Scoring Model
# -----------------------------

lead_data = {
    "Website_Visits":[5,10,15,20,8,12,25,18],
    "Cart_Value":[200,500,700,1000,300,600,1500,900],
    "Festival":[0,1,1,1,0,1,1,1],
    "Purchased":[0,1,1,1,0,1,1,1]
}

lead_df = pd.DataFrame(lead_data)

X_lead = lead_df[["Website_Visits","Cart_Value","Festival"]]
y_lead = lead_df["Purchased"]

lead_model = LogisticRegression()
lead_model.fit(X_lead, y_lead)

# -----------------------------
# STEP 3: Smart Festival Offer Logic
# -----------------------------

def smart_discount(predicted_sales):
    if predicted_sales > 4800:
        return 10   # High demand → small discount
    elif predicted_sales > 3500:
        return 20   # Medium demand
    else:
        return 30   # Low demand → bigger discount


# -----------------------------
# STEP 4: Run Festival Manager
# -----------------------------

if __name__ == "__main__":

    print("🎉 AI Festival Sales & Marketing Manager 🎉\n")

    # Inputs
    month = int(input("Enter month (1-12): "))
    festival = int(input("Festival month? (1 = Yes, 0 = No): "))
    product = input("Enter product name: ")
    visits = int(input("Customer website visits: "))
    cart = int(input("Customer cart value: "))

    # Predict Sales
    predicted_sales = sales_model.predict([[month, festival]])[0]

    # Decide Discount
    discount = smart_discount(predicted_sales)

    # Predict Buyer Probability
    customer_df = pd.DataFrame(
        [[visits, cart, festival]],
        columns=["Website_Visits","Cart_Value","Festival"]
    )

    buyer_prediction = lead_model.predict(customer_df)[0]

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------

    print("\n📊 Predicted Sales:", int(predicted_sales))
    print("🎁 Recommended Discount:", discount, "%")

    if buyer_prediction == 1:
        print("🔥 High Probability Festival Buyer!")
    else:
        print("⚠ Low Probability Buyer")

    print("\n📧 Festival Campaign Message:\n")

    print(f"""
Subject: 🎉 Festival Special – {discount}% OFF on {product}!

Celebrate this festival season with an exclusive {discount}% discount on {product}.

Limited time offer based on seasonal demand!
Shop now and grab the deal before it ends!

Happy Shopping!
""")
