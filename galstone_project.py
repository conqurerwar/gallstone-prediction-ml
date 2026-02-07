import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
st.set_page_config(page_title="Gallstone Prediction System")



df = pd.read_csv(r"C:\Users\hiii\Downloads\gallstone (1).csv")


X = df.drop("Gallstone Status", axis=1)
y = df["Gallstone Status"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

accuracies = {}

for name, model in models.items():
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    accuracies[name] = accuracy_score(y_test, preds)


best_model = models["Random Forest"]
best_accuracy = accuracies["Random Forest"]


st.title("🩺 Gallstone Prediction System")

st.subheader("Enter Patient Details")

user_data = {}
for col in X.columns:
    user_data[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_data])

if st.button("Predict Gallstone Status"):
    prediction = best_model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Gallstone Detected")
    else:
        st.success("✅ No Gallstone Detected")

    st.info(f"📊 Model Accuracy (Random Forest): {best_accuracy*100:.2f}%")


st.subheader("Model Accuracy Comparison")
acc_df = pd.DataFrame.from_dict(accuracies, orient="index", columns=["Accuracy"])
st.table(acc_df)
