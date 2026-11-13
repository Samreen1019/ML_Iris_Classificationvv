
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Iris Flower Classification")
st.write("Machine Learning Application")

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

st.subheader("Enter Flower Measurements")

sl = st.number_input("Sepal Length", value=5.1)
sw = st.number_input("Sepal Width", value=3.5)
pl = st.number_input("Petal Length", value=1.4)
pw = st.number_input("Petal Width", value=0.2)

if st.button("Predict Flower"):
    result = model.predict([[sl, sw, pl, pw]])
    flower = iris.target_names[result[0]]
    st.success("Predicted Flower: " + flower)
