import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("model.pkl", "rb") as file:
    vectorizer, model = pickle.load(file)

def preprocess_text(text):
    """Basic preprocessing: Convert text to lowercase"""
    return text.lower()

# Streamlit App
def main():
    st.title("Sentiment Analysis App")
    st.write("Enter a review to analyze its sentiment.")

    user_input = st.text_area("Enter your text here:")

    if st.button("Analyze Sentiment"):
        if user_input:
            processed_input = preprocess_text(user_input)
            transformed_input = vectorizer.transform([processed_input])
            prediction = model.predict(transformed_input)
            sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"

            st.write(f"**Sentiment:** {sentiment}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
