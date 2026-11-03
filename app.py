"""A compact, step-by-step Streamlit workbench for document reconstruction."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import joblib
from PIL import Image
import streamlit as st

from shredded_reconstruction.conditions import CONDITION_NAMES, observation_condition
from shredded_reconstruction.learned import train_v03_compatibility_model
from shredded_reconstruction.shred import compose_strips
from shredded_reconstruction.ui_reconstruction import predict_both_methods
from shredded_reconstruction.ui_state import (
    CUT_KEY, RESULTS_KEY, SHUFFLED_KEY, VIEW_KEY, clear_after_cut,
    clear_after_shuffle, clear_after_source_change, prepare_cut,
    shuffle_cut_document,
)
from shredded_reconstruction.v03_splits import V03_TRAIN_DOCUMENTS
from shredded_reconstruction.research_analysis import (
    CONDITION_NAMES_ZH,
    DOCUMENT_TYPE_NAMES,
    case_categories,
    comparison_table,
    concise_conclusion,
    load_confirmation,
)
from shredded_reconstruction.visualize import (
    fit_for_display, reconstruction_diagnostic, strip_workbench_montage,
    wrong_positions,
)


ROOT = Path(__file__).resolve().parent
DEMO_DIRECTORY = ROOT / "demo_documents"
INTERACTIVE_MODEL = ROOT / "artifacts" / "final" / "interactive_model" / "compatibility_model.joblib"
VIEW_WIDGET_KEY = "workbench_view_choice"

# Archive metadata keeps full titles; the chooser uses short projector-friendly labels.
DISPLAY_TITLES = {
    "external-usgs-landstat-economic-valuation-page-03": "USGS 报告 · 文字页",
    "external-usgs-landstat-economic-valuation-page-09": "USGS 报告 · 图表页",
    "external-nist-digital-identity-guidelines-page-06": "NIST 指南 · 说明页",
    "external-nist-digital-identity-guidelines-page-16": "NIST 指南 · 表格页",
    "external-gpo-annual-report-2023-page-04": "政府年报 · 图文页",
    "external-gpo-annual-report-2023-page-11": "政府年报 · 数据页",
    "synthetic-report": "合成 · 文字报告",
    "synthetic-table": "合成 · 数据表格",
    "synthetic-mixed": "合成 · 图文混排",
    "synthetic-sparse": "合成 · 稀疏版式",
}

st.set_page_config(page_title="碎纸文档自动重建", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="正在准备学习匹配模型…")
def get_learning_model():
    """Load the fixed UI model once; predictions still use current strips."""
    if INTERACTIVE_MODEL.exists():
        return joblib.load(INTERACTIVE_MODEL)
    # This fallback keeps a fresh clone usable before its model artifact is
    # prepared, but normal project use loads the serialized fixed model above.
    return train_v03_compatibility_model(
        V03_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )


@st.cache_data
def load_demo_manifest() -> list[dict[str, object]]:
    return list(json.loads((DEMO_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))["documents"])


@st.cache_data
def _load_demo_image(filename: str) -> Image.Image:
    """Decode each bundled page once per Streamlit cache lifetime."""
    with Image.open(DEMO_DIRECTORY / filename) as image:
        return image.convert("RGB")


@st.cache_data
def _demo_preview(filename: str, max_size: tuple[int, int]) -> Image.Image:
    return fit_for_display(_load_demo_image(filename), max_size)


@st.cache_data
def _uploaded_image(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


@st.cache_data
def _research_rows() -> list[dict[str, object]]:
    return load_confirmation(ROOT / "artifacts" / "final" / "independent_evaluation.json")


def _image(record: dict[str, object]) -> Image.Image:
    return _load_demo_image(str(record["filename"]))


def _display_title(record: dict[str, object]) -> str:
    return DISPLAY_TITLES.get(str(record["id"]), str(record["title"]))


def _show_preview(
    image: Image.Image,
    *,
    caption: str | None = None,
    max_size: tuple[int, int] = (760, 430),
) -> None:
    """Render a proportional UI copy, never the image supplied to solvers."""
    preview = fit_for_display(image, max_size)
    st.image(preview, caption=caption, width=preview.width)


def _show_demo_preview(filename: str, *, caption: str | None = None, max_size: tuple[int, int]) -> None:
    preview = _demo_preview(filename, max_size)
    st.image(preview, caption=caption, width=preview.width)


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _metrics(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    a, b, c = st.columns(3)
    a.metric("相邻匹配正确率", f"{metrics['adjacency_accuracy']:.1%}")
    b.metric("完全恢复", "是" if metrics["exact_match"] else "否")
    c.metric("运行耗时", f"{result['seconds']:.3f} 秒")


def _error_note(shuffled, result: dict[str, object]) -> str:
    if result["metrics"]["exact_match"]:
        return "所有纸条都处在正确位置。"
    positions = "、".join(map(str, wrong_positions(result["order"], shuffled.original_indices)))
    return f"红框位置：{positions}。重复文字、表格线、空白区域或图文边界都可能造成相邻关系歧义。"


def _show_result(name: str, shuffled, result: dict[str, object], mark_errors: bool) -> None:
    reconstructed = (
        reconstruction_diagnostic(shuffled.strips, result["order"], shuffled.original_indices)
        if mark_errors
        else compose_strips(shuffled.strips, result["order"])
    )
    # A 330px-high result leaves its three metrics visible in a 1080px
    # classroom viewport when the two methods are compared side by side.
    _show_preview(reconstructed, caption=name, max_size=(600, 330))
    _metrics(result)
    with st.expander(f"{name}的预测顺序与说明"):
        st.code(str(result["metrics"]["predicted_original_order"]), language=None)
        if mark_errors:
            st.caption(_error_note(shuffled, result))
        st.download_button(
            "下载此恢复图片", data=_png_bytes(reconstructed),
            file_name=f"{name}-reconstruction.png", mime="image/png", key=f"download-{name}",
        )


def _set_view(value: str) -> None:
    st.session_state[VIEW_KEY] = value
    st.session_state[VIEW_WIDGET_KEY] = value


def _gallery(records: list[dict[str, object]]) -> dict[str, object]:
    """Show a selected-card summary plus an on-demand, directly clickable gallery."""
    by_id = {str(record["id"]): record for record in records}
    if st.session_state.get("demo_selected_id") not in by_id:
        st.session_state["demo_selected_id"] = str(records[0]["id"])
    selected_id = str(st.session_state["demo_selected_id"])
    selected = by_id[selected_id]
    preview, details, action = st.columns((1, 3, 1))
    with preview:
        _show_demo_preview(str(selected["filename"]), max_size=(112, 88))
    with details:
        st.markdown(f"**当前文档：{_display_title(selected)}**")
        st.caption("外部页面已保存在本地，可离线使用。")
    with action:
        if st.button("浏览文档库", key="toggle-gallery", width="stretch"):
            st.session_state["gallery_open"] = not st.session_state.get("gallery_open", False)
            st.rerun()
    if st.session_state.get("gallery_open", False):
        st.caption("点击卡片下方的“选择”即可切换文档。")
        for start in range(0, len(records), 5):
            for column, record in zip(st.columns(5), records[start:start + 5], strict=True):
                record_id = str(record["id"])
                with column:
                    _show_demo_preview(str(record["filename"]), max_size=(132, 104))
                    st.caption(_display_title(record))
                    if record_id == selected_id:
                        st.button("当前文档", key=f"selected-{record_id}", disabled=True, width="stretch")
                    elif st.button("选择", key=f"select-{record_id}", width="stretch"):
                        st.session_state["demo_selected_id"] = record_id
                        st.session_state["gallery_open"] = False
                        st.rerun()
    return selected


def _large_source_controls(source_image: Image.Image, source_title: str) -> None:
    with st.expander("查看或下载原始大图"):
        _show_preview(source_image, caption=source_title, max_size=(1120, 820))
        st.download_button(
            "下载原始文档图片", data=_png_bytes(source_image), file_name="selected-document.png",
            mime="image/png", key="download-original",
        )


def _analysis() -> None:
    """Render only stored, independent confirmation predictions.

    This view intentionally has no model training or image reconstruction work.
    """
    rows = _research_rows()
    conclusion = concise_conclusion(rows)
    overall = conclusion["overall"]
    st.subheader("研究结论概览")
    c1, c2, c3 = st.columns(3)
    c1.metric("规则匹配", f"{overall['rule_adjacency']:.1%}", "相邻恢复率")
    c2.metric("学习匹配", f"{overall['learned_adjacency']:.1%}", "相邻恢复率")
    c3.metric("学习 − 规则", f"{conclusion['delta']:+.1%}", "按文档平均")
    strongest = conclusion["strongest_condition"]
    uncertainty = "文档族自助法区间跨过 0，整体差距不足以证明稳定优劣。" if conclusion["uncertain"] else "文档族自助法区间不跨过 0，但仍只适用于本实验的合成条件。"
    st.info(
        f"在 24 个独立文档族、120 个固定观察案例中，{conclusion['leader']}整体略高。"
        f"差异最大的条件是“{CONDITION_NAMES_ZH[str(strongest['condition'])]}”（学习−规则 {float(strongest['adjacency_delta_learned_minus_rule']):+.1%}）。{uncertainty}"
    )

    st.subheader("按文档类型比较")
    type_rows = comparison_table(rows, "variant")
    st.dataframe([
        {
            "文档类型": DOCUMENT_TYPE_NAMES[str(row["variant"])], "样本数": row["samples"],
            "规则相邻恢复率": f"{float(row['rule_adjacency']):.1%}", "学习相邻恢复率": f"{float(row['learned_adjacency']):.1%}",
            "规则完整恢复率": f"{float(row['rule_exact']):.1%}", "学习完整恢复率": f"{float(row['learned_exact']):.1%}",
        } for row in type_rows
    ], width="stretch", hide_index=True)

    st.subheader("按干扰条件比较")
    condition_rows = comparison_table(rows, "condition")
    st.dataframe([
        {
            "干扰条件": CONDITION_NAMES_ZH[str(row["condition"])], "样本数": row["samples"],
            "规则相邻恢复率": f"{float(row['rule_adjacency']):.1%}", "学习相邻恢复率": f"{float(row['learned_adjacency']):.1%}",
            "规则完整恢复率": f"{float(row['rule_exact']):.1%}", "学习完整恢复率": f"{float(row['learned_exact']):.1%}",
        } for row in condition_rows
    ], width="stretch", hide_index=True)

    st.subheader("文档类型 × 干扰条件")
    combinations = comparison_table(rows, "variant", "condition")
    selected_type = st.selectbox("文档类型", tuple(DOCUMENT_TYPE_NAMES), format_func=lambda value: DOCUMENT_TYPE_NAMES[value], key="analysis_type")
    selected_condition = st.selectbox("干扰条件", tuple(CONDITION_NAMES_ZH), format_func=lambda value: CONDITION_NAMES_ZH[value], key="analysis_condition")
    selected = next(row for row in combinations if row["variant"] == selected_type and row["condition"] == selected_condition)
    st.write({
        "样本数": selected["samples"],
        "规则匹配相邻恢复率": f"{float(selected['rule_adjacency']):.1%}",
        "学习匹配相邻恢复率": f"{float(selected['learned_adjacency']):.1%}",
        "学习−规则": f"{float(selected['adjacency_delta_learned_minus_rule']):+.1%}",
    })
    if int(selected["samples"]) < 10:
        st.caption("该组合只有少量独立页面（每格 6 例），只能作为提示，不能据此断言一种方法更适合。")

    st.subheader("成功与失败案例")
    categories = case_categories(rows)
    labels = {
        "both_success": "两种方法都成功", "rule_only": "规则匹配成功、学习匹配失败",
        "learned_only": "学习匹配成功、规则匹配失败", "both_fail": "两种方法都失败",
    }
    category = st.selectbox("案例类型", tuple(labels), format_func=lambda key: f"{labels[key]}（{len(categories[key])} 例）")
    candidate = categories[category][0]
    st.caption(
        f"保存的确认集案例：{DOCUMENT_TYPE_NAMES[str(candidate['variant'])]}；"
        f"{CONDITION_NAMES_ZH[str(candidate['condition'])]}；{candidate['strips']} 条纸条。"
        "下方图像由该案例已保存的预测顺序重放；不是手工拼接。"
    )
    asset = ROOT / "artifacts" / "final" / "cases" / f"{category}.png"
    if asset.exists():
        with Image.open(asset) as image:
            _show_preview(image.convert("RGB"), max_size=(1100, 640))
    else:
        st.warning("案例图尚未生成；请运行 scripts/prepare_final_analysis.py。")

    with st.expander("研究附录：技术对照与实验边界"):
        st.markdown(
        "本页仅展示两个实际使用的方法，数据来自未参与开发决策的最终确认集。历史消融实验仍保留在 `artifacts/v0.4/`："
            "它检验端点辅助项、干扰训练和上下对齐特征的贡献，不能与这里的两方法表混为同一次结论。"
        )
        st.markdown("所有外部页面测试均为完整公开页面的模拟竖直裁切，不是物理碎纸扫描。")


st.title("碎纸文档自动重建")
st.caption("选择一页文档，依次切条、打乱，再比较两种自动重建结果。")
page = st.radio("页面", ("自动重建", "实验分析"), horizontal=True, label_visibility="collapsed")
if page == "实验分析":
    _analysis()
    st.stop()
main_tab = st.container()

with main_tab:
    mode = st.radio("文档来源", ("演示文档", "上传图片"), horizontal=True)
    source_image: Image.Image | None = None
    source_key, source_title, selected = "no-source", "", None
    if mode == "演示文档":
        selected = _gallery(load_demo_manifest())
        selected_id = str(selected["id"])
        source_image = _image(selected)
        source_key, source_title = f"demo:{selected_id}", _display_title(selected)
    else:
        uploaded = st.file_uploader("上传 PNG/JPG 文档图片", type=("png", "jpg", "jpeg"))
        if uploaded is not None:
            payload = uploaded.getvalue()
            source_image = _uploaded_image(payload)
            source_key, source_title = f"upload:{hashlib.sha256(payload).hexdigest()}", uploaded.name

    source_changed = clear_after_source_change(st.session_state, source_key)
    if source_changed:
        st.session_state.pop(VIEW_WIDGET_KEY, None)
    cut = st.session_state.get(CUT_KEY)
    shuffled = st.session_state.get(SHUFFLED_KEY)
    results = st.session_state.get(RESULTS_KEY)
    controls, viewer = st.columns((1, 3), gap="large")
    with controls:
        st.subheader("操作")
        strip_count = st.select_slider("纸条数量", options=(8, 10, 12), value=10, key="strip_count")
        if cut is not None and cut.strip_count != strip_count:
            clear_after_cut(st.session_state)
            cut, shuffled, results = None, None, None
            st.session_state.pop(VIEW_WIDGET_KEY, None)
        cut_clicked = st.button("切成纸条", type="primary", key="cut_document", width="stretch", disabled=source_image is None)
        shuffle_clicked = st.button("随机打乱", key="shuffle_document", width="stretch", disabled=cut is None)
        reconstruct_clicked = st.button("开始重建", key="reconstruct_document", width="stretch", disabled=shuffled is None)
        st.caption("切条、打乱、重建是三个独立步骤。")
        if source_image is not None:
            _large_source_controls(source_image, source_title)
        if selected and selected.get("kind") == "external":
            with st.expander("资料来源"):
                st.caption(str(selected["license_note"]))
                st.link_button("打开原始资料页", str(selected["source_page_url"]), width="stretch")
        with st.expander("方法说明"):
            st.caption("学习匹配加载固定的交互模型文件；两种方法始终使用当前同一组打乱纸条。页面即时准确率只描述这次操作，不能与独立研究页的最终确认数字混用。")

    if cut_clicked and source_image is not None:
        st.session_state[CUT_KEY] = prepare_cut(source_image, strip_count)
        clear_after_shuffle(st.session_state)
        _set_view("裁切结果")
        st.rerun()
    if shuffle_clicked and cut is not None:
        previous = st.session_state.get(SHUFFLED_KEY)
        prior_indices = previous.original_indices if previous is not None else None
        st.session_state[SHUFFLED_KEY] = shuffle_cut_document(cut, previous_original_indices=prior_indices)
        clear_after_shuffle(st.session_state)
        _set_view("打乱结果")
        st.rerun()
    if reconstruct_clicked and shuffled is not None:
        with st.status("正在使用当前打乱纸条进行重建…", expanded=True) as status:
            st.write("规则匹配正在计算…")
            st.write("学习匹配正在计算…")
            st.session_state[RESULTS_KEY] = predict_both_methods(shuffled, get_learning_model())
            status.update(label="重建完成", state="complete", expanded=False)
        _set_view("恢复结果")
        st.rerun()

    cut = st.session_state.get(CUT_KEY)
    shuffled = st.session_state.get(SHUFFLED_KEY)
    results = st.session_state.get(RESULTS_KEY)
    available_views = ["文档预览"]
    if cut is not None:
        available_views.append("裁切结果")
    if shuffled is not None:
        available_views.append("打乱结果")
    if results is not None and shuffled is not None:
        available_views.append("恢复结果")
    current_view = str(st.session_state.get(VIEW_KEY, available_views[-1]))
    if current_view not in available_views:
        current_view = available_views[-1]
    if st.session_state.get(VIEW_WIDGET_KEY) not in available_views:
        st.session_state[VIEW_WIDGET_KEY] = current_view

    with viewer:
        chosen_view = st.radio("当前查看", available_views, horizontal=True, key=VIEW_WIDGET_KEY)
        st.session_state[VIEW_KEY] = chosen_view
        if chosen_view == "文档预览":
            if source_image is None:
                st.info("请选择演示文档，或上传一张 PNG/JPG 图片。")
            else:
                _show_preview(source_image, caption=f"文档预览：{source_title}", max_size=(700, 430))
        elif chosen_view == "裁切结果" and cut is not None:
            _show_preview(strip_workbench_montage(cut.strips), caption="纸条保持原始左右顺序；间隔和边框仅用于展示。", max_size=(980, 390))
        elif chosen_view == "打乱结果" and shuffled is not None:
            _show_preview(strip_workbench_montage(shuffled.strips, show_input_positions=True), caption="已经打乱。编号仅表示当前输入位置，不是原始正确顺序。", max_size=(980, 390))
        elif chosen_view == "恢复结果" and results is not None and shuffled is not None:
            result_view = st.radio("恢复结果", ("并排比较", "规则匹配", "学习匹配", "原始文档"), horizontal=True, key="result_view")
            mark_errors = st.checkbox("标记错误纸条位置", value=False, key="mark_errors")
            if result_view == "原始文档":
                _show_preview(source_image, caption="原始文档（只用于重建后的对比与评测）", max_size=(700, 430))
            elif result_view == "规则匹配":
                _show_result("规则匹配", shuffled, results["rule"], mark_errors)
            elif result_view == "学习匹配":
                _show_result("学习匹配", shuffled, results["learned"], mark_errors)
            else:
                left, right = st.columns(2, gap="large")
                with left:
                    _show_result("规则匹配", shuffled, results["rule"], mark_errors)
                with right:
                    _show_result("学习匹配", shuffled, results["learned"], mark_errors)
