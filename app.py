"""Step-by-step Streamlit interface for document reconstruction."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import streamlit as st

from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.learned import train_v03_compatibility_model
from shredded_reconstruction.ui_reconstruction import predict_both_methods
from shredded_reconstruction.shred import compose_strips
from shredded_reconstruction.ui_state import (
    CUT_KEY, RESULTS_KEY, SHUFFLED_KEY, clear_after_cut, clear_after_shuffle,
    clear_after_source_change, prepare_cut, shuffle_cut_document,
)
from shredded_reconstruction.v03_splits import V03_TRAIN_DOCUMENTS
from shredded_reconstruction.visualize import reconstruction_diagnostic, shuffled_montage, wrong_positions


ROOT = Path(__file__).resolve().parent
DEMO_DIRECTORY = ROOT / "demo_documents"
st.set_page_config(page_title="碎纸文档自动重建", layout="wide")


@st.cache_resource(show_spinner="正在准备学习匹配模型…")
def get_learning_model():
    """Cache the fixed V0.3 learned configuration, never its predictions."""
    return train_v03_compatibility_model(
        V03_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )


@st.cache_data
def load_demo_manifest() -> list[dict[str, object]]:
    return list(json.loads((DEMO_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))["documents"])


def _image(record: dict[str, object]) -> Image.Image:
    return Image.open(DEMO_DIRECTORY / str(record["filename"])).convert("RGB")


def _metrics(name: str, result: dict[str, object]) -> None:
    metrics = result["metrics"]
    a, b, c = st.columns(3)
    a.metric("相邻匹配正确率", f"{metrics['adjacency_accuracy']:.1%}")
    b.metric("完全恢复", "是" if metrics["exact_match"] else "否")
    c.metric("运行耗时", f"{result['seconds']:.3f} 秒")
    st.caption(f"{name}的预测顺序：{metrics['predicted_original_order']}")


def _error_note(shuffled, result: dict[str, object]) -> str:
    if result["metrics"]["exact_match"]:
        return "所有纸条都处在正确位置。"
    positions = "、".join(map(str, wrong_positions(result["order"], shuffled.original_indices)))
    return f"红框位置：{positions}。重复文字、表格线、空白区域或图文边界都可能造成相邻关系歧义。"


def _show_result(name: str, shuffled, result: dict[str, object], mark_errors: bool) -> None:
    image = reconstruction_diagnostic(shuffled.strips, result["order"], shuffled.original_indices) if mark_errors else compose_strips(shuffled.strips, result["order"])
    st.image(image, caption=name, width="stretch")
    _metrics(name, result)
    if mark_errors:
        st.caption(_error_note(shuffled, result))


def _analysis() -> None:
    st.subheader("固定实验结果")
    st.caption("这些是已保存的正式实验记录，和当前操作得到的即时结果分开显示。")
    historic = ROOT / "artifacts" / "v0.3" / "evaluation.json"
    if historic.exists():
        summary = json.loads(historic.read_text(encoding="utf-8"))["final_test_summary"]
        st.bar_chart({
            "规则匹配": summary["traditional_v0_2"]["document_mean_adjacency"],
            "学习匹配": summary["learned_v0_3"]["document_mean_adjacency"],
        }, horizontal=True)
        st.caption("历史合成基准：按文档平均的相邻恢复率。")
    chart = ROOT / "artifacts" / "v0.4" / "ablation-confirmation.png"
    if chart.exists():
        st.image(str(chart), caption="独立确认实验：不同固定配置分别在新文档族上运行。", width="stretch")
    success = ROOT / "artifacts" / "v0.4" / "success-v04-usgs-landstat-economic-valuation-10-external_page_simulated_cut-usgs-landstat-economic-valuation-9.png"
    failure = ROOT / "artifacts" / "v0.4" / "failure-v04-gpo-annual-report-2023-10-external_page_simulated_cut-gpo-annual-report-2023-4.png"
    if success.exists() and failure.exists():
        left, right = st.columns(2)
        left.image(str(success), caption="外部页面：成功案例", width="stretch")
        right.image(str(failure), caption="外部页面：失败案例", width="stretch")
    with st.expander("技术与来源说明"):
        st.markdown("本页的学习匹配加载固定的 V0.3 训练划分、五种观测条件轮换和有限上下对齐配置。独立确认实验使用另一组训练文档，二者的数值不可混为同一次评测。")
        st.markdown("外部页面均为完整公开页面的模拟裁切，不是物理碎纸扫描；来源在 `demo_documents/manifest.json`。")


st.title("碎纸文档自动重建")
st.caption("选择一页文档，亲手完成切条、打乱和自动重建。")
main_tab, analysis_tab = st.tabs(["自动重建", "实验分析"])

with main_tab:
    mode = st.radio("文档来源", ("演示文档", "上传图片"), horizontal=True)
    source_image: Image.Image | None = None
    source_key, source_title, selected = "no-source", "", None
    if mode == "演示文档":
        records = load_demo_manifest()
        by_id = {str(record["id"]): record for record in records}
        selected_id = st.selectbox("选择文档", list(by_id), format_func=lambda key: str(by_id[key]["title"]))
        selected = by_id[selected_id]
        source_image, source_key, source_title = _image(selected), f"demo:{selected_id}", str(selected["title"])
        with st.expander("浏览演示文档缩略图"):
            for start in range(0, len(records), 5):
                for column, record in zip(st.columns(5), records[start:start + 5], strict=True):
                    column.image(_image(record), caption=str(record["title"]), width="stretch")
    else:
        uploaded = st.file_uploader("上传 PNG/JPG 文档图片", type=("png", "jpg", "jpeg"))
        if uploaded is not None:
            payload = uploaded.getvalue()
            source_image = Image.open(io.BytesIO(payload)).convert("RGB")
            source_key, source_title = f"upload:{hashlib.sha256(payload).hexdigest()}", uploaded.name

    clear_after_source_change(st.session_state, source_key)
    preview, controls = st.columns((2, 1))
    with preview:
        if source_image is None:
            st.info("请选择演示文档，或上传一张 PNG/JPG 图片。")
        else:
            st.image(source_image, caption=f"文档预览：{source_title}", width="stretch")
            if selected and selected.get("kind") == "external":
                st.caption(f"来源：{selected['license_note']}")
    with controls:
        strip_count = st.select_slider("纸条数量", options=(8, 10, 12), value=10)
        cut = st.session_state.get(CUT_KEY)
        if cut is not None and cut.strip_count != strip_count:
            clear_after_cut(st.session_state)
            cut = None
        cut_clicked = st.button("切成纸条", type="primary", width="stretch", disabled=source_image is None)
        shuffle_clicked = st.button("随机打乱", width="stretch", disabled=cut is None)
        reconstruct_clicked = st.button("开始重建", width="stretch", disabled=st.session_state.get(SHUFFLED_KEY) is None)
        st.caption("选择文档 → 切成纸条 → 随机打乱 → 开始重建")

    if cut_clicked and source_image is not None:
        st.session_state[CUT_KEY] = prepare_cut(source_image, strip_count)
        clear_after_shuffle(st.session_state)
        st.rerun()
    cut = st.session_state.get(CUT_KEY)
    if cut is not None:
        st.subheader("纸条（保持原顺序）")
        st.image(compose_strips(cut.strips), caption="已切成纸条；尚未打乱。", width="stretch")

    if shuffle_clicked and cut is not None:
        previous = st.session_state.get(SHUFFLED_KEY)
        prior_indices = previous.original_indices if previous is not None else None
        st.session_state[SHUFFLED_KEY] = shuffle_cut_document(cut, previous_original_indices=prior_indices)
        clear_after_shuffle(st.session_state)
        st.rerun()
    shuffled = st.session_state.get(SHUFFLED_KEY)
    if shuffled is not None:
        st.subheader("打乱后的纸条")
        st.image(shuffled_montage(shuffled.strips), caption="数字只表示当前输入位置，不是原始正确顺序。", width="stretch")

    if reconstruct_clicked and shuffled is not None:
        with st.status("正在使用当前打乱纸条进行重建…", expanded=True) as status:
            st.write("规则匹配正在计算…")
            st.write("学习匹配正在计算…")
            st.session_state[RESULTS_KEY] = predict_both_methods(shuffled, get_learning_model())
            status.update(label="重建完成", state="complete", expanded=False)
        st.rerun()
    results = st.session_state.get(RESULTS_KEY)
    if results is not None and shuffled is not None:
        st.subheader("重建结果")
        view = st.radio("查看", ("并排比较", "规则匹配", "学习匹配", "原始文档"), horizontal=True)
        mark_errors = st.toggle("标记错误纸条位置", value=True)
        if view == "原始文档":
            st.image(source_image, caption="原始文档（只用于重建后的对比与评测）", width="stretch")
        elif view == "规则匹配":
            _show_result("规则匹配", shuffled, results["rule"], mark_errors)
        elif view == "学习匹配":
            _show_result("学习匹配", shuffled, results["learned"], mark_errors)
        else:
            left, right = st.columns(2)
            with left:
                _show_result("规则匹配", shuffled, results["rule"], mark_errors)
            with right:
                _show_result("学习匹配", shuffled, results["learned"], mark_errors)

with analysis_tab:
    _analysis()
