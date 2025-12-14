import streamlit as st
from inference import predict

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# Sentiment → Emoji mapping
# --------------------------------------------------
SENTIMENT_EMOJI = {
    "Very Negative": "😡",
    "Negative": "☹️",
    "Neutral": "😐",
    "Positive": "🙂",
    "Very Positive": "😄"
}

# --------------------------------------------------
# UI Header
# --------------------------------------------------
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and click **Predict Sentiment** "
    "to see the sentiment predicted by the RoBERTa model."
)

# --------------------------------------------------
# Text input
# --------------------------------------------------
review = st.text_area(
    "Movie Review",
    height=150,
    placeholder="Type your movie review here..."
)

# --------------------------------------------------
# Prediction button
# --------------------------------------------------
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predict(review)

        st.success("Prediction complete!")

        # --------------------------------------------------
        # Result display
        # --------------------------------------------------
        st.subheader("Result")

        label = result["label"]
        confidence = result["confidence"]
        emoji = SENTIMENT_EMOJI[label]

        st.markdown(f"### {emoji} **{label}**")
        st.progress(confidence)
        st.write(f"**Confidence:** {confidence:.2f}")

