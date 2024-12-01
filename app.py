# Save this as `app.py`
import streamlit as st
import joblib

# Load the model and vectorizer
model_path = 'yeh_naya_hai.joblib'  # Ensure this matches your file path
models = joblib.load(model_path)

# Extract individual components
vectorizer = models['vectorizer']
rf_classifier_escalation = models['rf_classifier_escalation']
rf_classifier_category = models['rf_classifier_category']
rf_classifier_sentiment = models['rf_classifier_sentiment']

# Streamlit app
st.title("Customer Support Query Classifier")
st.write("This app predicts the escalation need, category, and sentiment of customer queries.")

# Input field for the query
query = st.text_area("Enter your query below:", placeholder="Type your query here...")

if st.button("Predict"):
    if query.strip():
        # Transform the query using the vectorizer
        query_tfidf = vectorizer.transform([query])

        # Make predictions
        escalation_prediction = rf_classifier_escalation.predict(query_tfidf)[0]
        category_prediction = rf_classifier_category.predict(query_tfidf)[0]
        sentiment_prediction = rf_classifier_sentiment.predict(query_tfidf)[0]

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Escalation Required:** {'Yes' if escalation_prediction == 1 else 'No'}")
        st.write(f"**Category:** {category_prediction}")
        st.write(f"**Sentiment:** {sentiment_prediction}")
    else:
        st.error("Please enter a query to get predictions.")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("This app classifies customer support queries into categories, determines the sentiment, and assesses if escalation is required.")
