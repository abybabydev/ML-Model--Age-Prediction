import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset (Height in cm, Weight in kg, Age in years)
X = np.array([[170, 65], [160, 58], [180, 75], [175, 70], [165, 62]])
y = np.array([25, 22, 30, 28, 24])  # Corresponding ages

# Creating the model and training
model = LinearRegression()
model.fit(X, y)

# Predicting age for a new person (Height = 172cm, Weight = 68kg)
new_person = np.array([[172, 68]])
predicted_age = model.predict(new_person)

print("Predicted Age (Scikit-Learn):", predicted_age[0])
