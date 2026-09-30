
import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text
from embedding import create_embeddings
from attention import calculate_attention


# Page settings
st.set_page_config(
    page_title="AI Attention Visualizer",
    page_icon="🧠"
)

# Title
st.title("🧠 AI Attention Visualizer")

st.write(
    "Upload a study-notes image to extract text, "
    "generate embeddings and visualize word attention."
)


# Upload image
uploaded_file = st.file_uploader(
    "Upload your study-notes image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.subheader("📷 Uploaded Image")
    st.image(image, width=600)


    # -------------------------
    # OCR
    # -------------------------

    st.subheader("📝 Extracted Text")

    text = extract_text(image)

    if not text.strip():

        st.error("No text found in the image.")

        st.stop()

    st.write(text)


    # -------------------------
    # Word Processing
    # -------------------------

    words = text.split()

    # Clean words
    words = [
        word.strip(".,!?;:()[]{}")
        for word in words
    ]

    # Remove short words
    words = [
        word
        for word in words
        if len(word) > 2
    ]

    # Maximum 20 words
    words = words[:20]


    if len(words) == 0:

        st.warning("No suitable words found.")

        st.stop()


    st.subheader("🔤 Processed Words")

    st.write(", ".join(words))


    # -------------------------
    # Embeddings
    # -------------------------

    with st.spinner("Creating embeddings..."):

        embeddings = create_embeddings(words)


    st.success(
        f"Embeddings created for {len(words)} words."
    )


    # -------------------------
    # Attention
    # -------------------------

    with st.spinner("Calculating attention..."):

        scores = calculate_attention(embeddings)


    # Normalize scores
    display_scores = scores / scores.max()


    # -------------------------
    # Word Attention
    # -------------------------

    st.subheader("🧠 Word Attention")

    for word, score in zip(
        words,
        display_scores
    ):

        st.write(f"**{word}**")

        st.progress(
            float(score)
        )


    # -------------------------
    # Highest Attention Word
    # -------------------------

    top_index = np.argmax(scores)

    top_word = words[top_index]


    st.success(
        f"⭐ Highest Attention Word: **{top_word}**"