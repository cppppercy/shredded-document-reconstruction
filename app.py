"""Small Streamlit interface for the V0.2 reconstruction pipeline."""

import streamlit as st

from shredded_reconstruction.conditions import CONDITION_NAMES, condition_description
from shredded_reconstruction.document import DOCUMENT_VARIANTS
from shredded_reconstruction.evaluation import run_case
from shredded_reconstruction.learned import train_compatibility_model
from shredded_reconstruction.splits import TRAIN_DOCUMENTS
from shredded_reconstruction.visualize import comparison_figure, shuffled_montage

st.set_page_config(page_title="Shredded Document Reconstruction", layout="wide")
st.title("Automatic Reconstruction of Shredded Documents")
st.caption("V0.2 | learned local compatibility + exact global ordering under controlled disturbances")


@st.cache_resource(show_spinner="Training the lightweight compatibility model...")
def get_model(method: str):
    if method == "V0.1 learned":
        return train_compatibility_model()
    return train_compatibility_model(
        TRAIN_DOCUMENTS,
    )


with st.sidebar:
    variant = st.selectbox("Document type", DOCUMENT_VARIANTS, index=2)
    strip_count = st.slider("Number of strips", 8, 12, 10)
    condition = st.selectbox("Observation condition", CONDITION_NAMES, format_func=condition_description)
    method = st.selectbox("Reconstruction method", ("V0.2 learned", "V0.1 learned", "Traditional baseline"))
    seed = st.number_input("Document seed", value=2026, step=1)
    shuffle_seed = st.number_input("Shuffle seed", value=73, step=1)
    run = st.button("Generate and reconstruct", type="primary", width="stretch")

if run or "case" not in st.session_state:
    st.session_state.case = run_case(
        seed=int(seed), variant=variant, strip_count=strip_count, shuffle_seed=int(shuffle_seed),
        model=None if method == "Traditional baseline" else get_model(method), condition_name=condition,
    )

case = st.session_state.case
comparison = comparison_figure(
    case.original, shuffled_montage(case.shredded.strips), case.baseline_reconstruction,
    case.learned_reconstruction, case.baseline_metrics, case.learned_metrics,
)
st.image(comparison, caption="原图、打乱纸条与自动拼接结果", width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Traditional baseline")
    st.json(case.baseline_metrics)
with right:
    st.subheader(method)
    st.json(case.learned_metrics if case.learned_metrics is not None else {"note": "Traditional baseline selected"})

st.info("重建器只能看到打乱后的纸条像素；真实纸条编号只在预测完成后用于计算指标。V0.2 在局部比较时可尝试有限的非循环上下对齐。")
