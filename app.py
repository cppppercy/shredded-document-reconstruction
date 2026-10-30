"""Small Streamlit interface for the V0.1 reconstruction pipeline."""

import streamlit as st

from shredded_reconstruction.document import DOCUMENT_VARIANTS
from shredded_reconstruction.evaluation import run_case
from shredded_reconstruction.learned import train_compatibility_model
from shredded_reconstruction.visualize import comparison_figure, shuffled_montage

st.set_page_config(page_title="Shredded Document Reconstruction", layout="wide")
st.title("Automatic Reconstruction of Shredded Documents")
st.caption("V0.1 · learned local compatibility + exact global ordering")


@st.cache_resource(show_spinner="Training the lightweight compatibility model...")
def get_model():
    return train_compatibility_model()


with st.sidebar:
    variant = st.selectbox("Document type", DOCUMENT_VARIANTS, index=2)
    strip_count = st.slider("Number of strips", 8, 12, 10)
    seed = st.number_input("Document seed", value=2026, step=1)
    shuffle_seed = st.number_input("Shuffle seed", value=73, step=1)
    run = st.button("Generate and reconstruct", type="primary", width="stretch")

if run or "case" not in st.session_state:
    st.session_state.case = run_case(
        seed=int(seed),
        variant=variant,
        strip_count=strip_count,
        shuffle_seed=int(shuffle_seed),
        model=get_model(),
    )

case = st.session_state.case
shuffled = shuffled_montage(case.shredded.strips)
comparison = comparison_figure(
    case.original,
    shuffled,
    case.baseline_reconstruction,
    case.learned_reconstruction,
    case.baseline_metrics,
    case.learned_metrics,
)
st.image(comparison, caption="End-to-end result", width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Baseline")
    st.json(case.baseline_metrics)
with right:
    st.subheader("Learned compatibility")
    st.json(case.learned_metrics)

st.info("The reconstruction methods receive only shuffled strip pixels. Correct strip indices are used after prediction for evaluation.")
