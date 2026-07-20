import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load dataset
data = pd.read_csv("stu.csv")


# Features and target
X = data[['Hours_Studied',
          'Previous_Score',
          'Sleep_Hours']]

y = data['Exam_Score']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# Test accuracy
prediction = model.predict(X_test)

accuracy = r2_score(
    y_test,
    prediction
)

print("Model Accuracy:", accuracy)


# Save model
pickle.dump(
    model,
    open("model.pkl","wb")
)

print("Model saved successfully")