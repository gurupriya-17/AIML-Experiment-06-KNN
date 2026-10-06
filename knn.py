from sklearn.neighbors import KNeighborsClassifier

# Training data
X = [
    [1, 1],
    [2, 2],
    [3, 3],
    [6, 6],
    [7, 7],
    [8, 8]
]

# Labels
y = [
    "Class A",
    "Class A",
    "Class A",
    "Class B",
    "Class B",
    "Class B"
]

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

# Test a new sample
sample = [[4, 4]]

# Predict class
prediction = model.predict(sample)

print("New Sample:", sample[0])
print("Predicted Class:", prediction[0])
