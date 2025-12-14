import streamlit as st
from inference import predict

st.set_page_config(
    page_title="RoBERTa Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below and click **Predict Sentiment** "
    "to see the sentiment predicted by the RoBERTa model."
)

review = st.text_area(
    "Movie Review",
    height=150,
    placeholder="Type your movie review here..."
)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predict(review)

        st.success("Prediction complete!")
        st.subheader("Result")
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}")
