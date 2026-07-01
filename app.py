

import streamlit as st

from predict import predict_sentiment


st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="movie",
    layout="centered",
)

st.markdown(
    """
    <style>
    @keyframes moveBackdrop {
        0% { background-position: 0 0, 0 0; }
        100% { background-position: 220px 120px, -180px 90px; }
    }

    .stApp {
        color: #f7f2ff;
        background:
            linear-gradient(115deg, rgba(166, 95, 255, 0.12) 0 1px, transparent 1px 34px),
            linear-gradient(65deg, rgba(255, 255, 255, 0.06) 0 1px, transparent 1px 42px),
            #09070f;
        background-size: 220px 120px, 180px 90px, auto;
        animation: moveBackdrop 22s linear infinite;
    }

    .block-container {
        max-width: 860px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .review-shell {
        background: rgba(12, 9, 20, 0.9);
        border: 1px solid rgba(166, 95, 255, 0.34);
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
    }

    .main-title {
        color: #ffffff;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 0.75rem;
    }

    .intro-copy {
        color: #cfc3e8;
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 1.6rem;
    }

    .section-label {
        color: #f7f2ff;
        font-size: 1rem;
        font-weight: 700;
        margin: 1.1rem 0 0.6rem;
    }

    .helper-text {
        color: #a99bc4;
        font-size: 0.9rem;
        margin-top: -0.25rem;
    }

    .result-box {
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1rem;
        padding: 1rem 1.1rem;
    }

    .positive-result {
        background: rgba(38, 141, 87, 0.18);
        border: 1px solid rgba(91, 201, 133, 0.55);
        color: #8ee0a6;
    }

    .negative-result {
        background: rgba(187, 45, 45, 0.18);
        border: 1px solid rgba(255, 126, 126, 0.55);
        color: #ff9a8f;
    }

    .stTextArea textarea {
        background: #14101f;
        border: 1px solid rgba(166, 95, 255, 0.45);
        border-radius: 8px;
        color: #f7f2ff;
        font-size: 1rem;
    }

    .stTextArea textarea:focus {
        border-color: #a66bff;
        box-shadow: 0 0 0 1px #a66bff;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(166, 95, 255, 0.5);
        background: rgba(255, 255, 255, 0.05);
        color: #f7f2ff;
        font-weight: 650;
    }

    .stButton > button:hover {
        border-color: #a66bff;
        color: #a66bff;
    }

    .stButton > button[kind="primary"] {
        background: #a66bff;
        border-color: #a66bff;
        color: #111111;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

st.markdown('<div class="review-shell">', unsafe_allow_html=True)
st.markdown(
    """
    <h1 class="main-title">Movie Review Sentiment Analysis</h1>
    <p class="intro-copy">
        Let us know how you felt about a movie, scene, season, or performance.
        The app will read your review and predict whether the mood feels positive
        or negative.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Pick a feeling tag</div>', unsafe_allow_html=True)

first_row = st.columns(3)
second_row = st.columns(3)

with first_row[0]:
    if st.button("Loved it", use_container_width=True):
        st.session_state.review_text = "I loved this movie. The acting and story were excellent."

with first_row[1]:
    if st.button("Not for me", use_container_width=True):
        st.session_state.review_text = "Not a good movie. The previous season was much better."

with first_row[2]:
    if st.button("Not bad", use_container_width=True):
        st.session_state.review_text = "Not bad. The story was simple, but I enjoyed the stunts."

with second_row[0]:
    if st.button("Great acting", use_container_width=True):
        st.session_state.review_text = "The acting was great and the characters felt real."

with second_row[1]:
    if st.button("Too slow", use_container_width=True):
        st.session_state.review_text = "The movie was too slow and I lost interest halfway through."

with second_row[2]:
    if st.button("Bad stunts", use_container_width=True):
        st.session_state.review_text = "The stunt scenes were bad and the action felt fake."

st.markdown('<div class="section-label">Write your review</div>', unsafe_allow_html=True)

review = st.text_area(
    "Movie review",
    key="review_text",
    height=190,
    placeholder="Example: The movie looked amazing, but the story was not good.",
    label_visibility="collapsed",
)

st.markdown(
    f"<p class='helper-text'>{len(review.strip())} characters written</p>",
    unsafe_allow_html=True,
)

predict_clicked = st.button("Analyze Review", type="primary", use_container_width=True)

if predict_clicked:
    if not review.strip():
        st.warning("Please write a review before analyzing it.")
    else:
        sentiment = predict_sentiment(review)

        if sentiment == "positive":
            st.markdown(
                "<div class='result-box positive-result'>Predicted mood: Positive review</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-box negative-result'>Predicted mood: Negative review</div>",
                unsafe_allow_html=True,
            )

st.markdown('</div>', unsafe_allow_html=True)
