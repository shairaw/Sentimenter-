# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['REVIEW'],  # Features (reviews)
    df['RATING'] >= 5,  # Binary labels (positive if rating > 5, negative otherwise)
    test_size=0.2,
    random_state=42
)

# Create a TfidfVectorizer to convert text data to numerical features
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Train a Logistic Regression model for sentiment analysis
model = LogisticRegression(random_state=42)
model.fit(train_features, train_labels)

predictions = model.predict(test_features)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(test_labels, predictions))
