import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset from the pickle file.
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels from the loaded dictionary.
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets (80% train, 20% test).
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize the RandomForestClassifier model.
model = RandomForestClassifier()

# Train the model using the training data.
model.fit(x_train, y_train)

# Make predictions on the test data.
y_predict = model.predict(x_test)

# Calculate the accuracy of the model.
score = accuracy_score(y_predict, y_test)

# Print the classification accuracy.
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file for future use.
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
