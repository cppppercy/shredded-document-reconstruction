"""Classroom-oriented Streamlit demonstration for the V0.2 prototype."""

from __future__ import annotations

import random
from time import time_ns

import streamlit as st

from shredded_reconstruction.conditions import CONDITION_NAMES, condition_description
from shredded_reconstruction.document import DOCUMENT_VARIANTS
from shredded_reconstruction.evaluation import CaseResult, run_case
from shredded_reconstruction.learned import train_compatibility_model
from shredded_reconstruction.splits import TRAIN_DOCUMENTS
from shredded_reconstruction.visualize import (
    reconstruction_diagnostic,
    shuffled_montage,
    wrong_positions,
)

st.set_page_config(page_title="碎纸文档自动重建 | V0.2", layout="wide")

SHOWCASES = {
    "成功示例：干净报告（完整恢复）": {
        "seed": 5100, "variant": "report", "strip_count": 8,
        "condition": "clean", "shuffle_seed": 86713,
    },
    "失败示例：边缘缺失报告（保留真实错误）": {
        "seed": 5101, "variant": "report", "strip_count": 10,
        "condition": "edge_loss", "shuffle_seed": 110487,
    },
}


@st.cache_resource(show_spinner="正在训练轻量级兼容度模型...")
def get_model(version: str):
    if version == "V0.1 学习模型":
        return train_compatibility_model()
    return train_compatibility_model(TRAIN_DOCUMENTS)


def _random_spec() -> dict[str, int | str]:
    rng = random.Random(time_ns())
    return {
        "seed": rng.randint(6000, 9999),
        "variant": rng.choice(DOCUMENT_VARIANTS),
        "strip_count": rng.choice((8, 10, 12)),
        "condition": rng.choice(CONDITION_NAMES),
        "shuffle_seed": rng.randint(1, 999_999),
    }


def _reason(case: CaseResult, metrics: dict, order: list[int], method_name: str) -> str:
    if metrics["exact_match"]:
        return f"{method_name} 完整恢复成功：所有纸条都在正确位置。"
    reasons = {
        "edge_loss": "纸条边缘已有永久像素缺失，接缝证据变弱。",
        "vertical_shift": "纸条上下错位会让同一行的文字和线条无法直接对齐。",
        "no_assists": "边框和连续纸张纹理被移除，首尾方向更容易产生歧义。",
        "scan": "亮度差和轻度模糊降低了边缘的局部对比度。",
        "clean": "局部纹理或重复文字片段可能让多个拼接候选得分接近。",
    }
    incorrect = wrong_positions(order, case.shredded.original_indices)
    positions = "、".join(map(str, incorrect))
    return f"红框位置：{positions}。可能原因：{reasons[case.condition]}"


def _run(spec: dict[str, int | str], model_version: str) -> CaseResult:
    with st.status("正在执行真实重建…", expanded=True) as status:
        st.write("1 / 3 生成文档并按设定条件切成纸条")
        st.write("2 / 3 运行传统局部兼容度 + 全局优化")
        st.write("3 / 3 运行学习兼容度 + 全局优化")
        result = run_case(
            seed=int(spec["seed"]),
            variant=str(spec["variant"]),
            strip_count=int(spec["strip_count"]),
            shuffle_seed=int(spec["shuffle_seed"]),
            model=get_model(model_version),
            condition_name=str(spec["condition"]),
            baseline_max_vertical_shift=3,
        )
        status.update(label="真实重建完成", state="complete", expanded=False)
    return result


def _metrics_panel(title: str, metrics: dict, seconds: float | None) -> None:
    st.subheader(title)
    first, second, third = st.columns(3)
    first.metric("相邻关系正确率", f"{metrics['adjacency_accuracy']:.1%}")
    second.metric("完整恢复", "是" if metrics["exact_match"] else "否")
    third.metric("实际运行耗时", f"{seconds or 0.0:.3f} 秒")
    st.caption(f"预测原始顺序：{metrics['predicted_original_order']}")


st.title("碎纸文档自动重建：从打乱纸条到全局拼接")
st.caption("V0.2 课堂演示页｜所有准确率与耗时均来自当前次实际运行。")
st.info(
    "应用背景：该类技术可为司法碎纸证据恢复、受损档案整理与历史文献数字化提供辅助。"
    "本项目目前只是规则竖直纸条的实验原型：不处理旋转、重叠、任意形状碎片或真实扫描件。"
)

with st.sidebar:
    st.header("1. 选择演示")
    requested_showcase = st.query_params.get("showcase", "success")
    default_mode = {"success": 0, "failure": 1, "custom": 2}.get(requested_showcase, 0)
    mode = st.radio("课堂一键示例 / 自定义测试", ("成功示例", "失败示例", "自定义"), index=default_mode)
    model_version = st.radio("机器学习对照模型", ("V0.2 学习模型", "V0.1 学习模型"))

    if mode == "成功示例":
        spec = SHOWCASES["成功示例：干净报告（完整恢复）"].copy()
        st.success("固定成功案例：用于快速讲清完整流程。")
    elif mode == "失败示例":
        spec = SHOWCASES["失败示例：边缘缺失报告（保留真实错误）"].copy()
        st.warning("固定失败案例：用于讨论边缘信息丢失。")
    else:
        if "random_spec" not in st.session_state:
            st.session_state.random_spec = _random_spec()
        if st.button("随机生成一组测试", width="stretch"):
            st.session_state.random_spec = _random_spec()
        random_spec = st.session_state.random_spec
        variant = st.selectbox("文档类型", DOCUMENT_VARIANTS, index=DOCUMENT_VARIANTS.index(str(random_spec["variant"])))
        strip_count = st.select_slider("纸条数量", options=(8, 10, 12), value=int(random_spec["strip_count"]))
        condition = st.selectbox("图像条件", CONDITION_NAMES, format_func=condition_description, index=CONDITION_NAMES.index(str(random_spec["condition"])))
        seed = st.number_input("文档种子", min_value=0, value=int(random_spec["seed"]), step=1)
        shuffle_seed = st.number_input("打乱种子", min_value=0, value=int(random_spec["shuffle_seed"]), step=1)
        spec = {
            "seed": int(seed), "variant": variant, "strip_count": int(strip_count),
            "condition": condition, "shuffle_seed": int(shuffle_seed),
        }
    run = st.button("开始：打乱并自动重建", type="primary", width="stretch")

spec_key = tuple(sorted(spec.items())) + (model_version,)
if run or "case" not in st.session_state:
    st.session_state.case = _run(spec, model_version)
    st.session_state.case_key = spec_key

case: CaseResult = st.session_state.case
if st.session_state.get("case_key") != spec_key:
    st.caption("已修改配置；点击“开始：打乱并自动重建”以运行这组新设置。")

st.header("流程总览")
steps = st.columns(4)
for column, number, text in zip(
    steps,
    ("1", "2", "3", "4"),
    ("选择文档与干扰", "随机打乱纸条", "两种方法真实重建", "比较指标并解释错误"),
    strict=True,
):
    column.markdown(f"### {number}. {text}")

st.header("2. 输入：原始文档与打乱纸条")
input_left, input_right = st.columns(2)
with input_left:
    st.image(case.original, caption="原始文档（正确顺序仅用于最后评测）", width="stretch")
with input_right:
    st.image(
        shuffled_montage(case.shredded.strips),
        caption="打乱后的输入：上方数字只是输入位置，不是正确顺序",
        width="stretch",
    )

st.header("3. 自动重建：传统方法与机器学习方法")
st.caption("绿色框表示该重建位置正确；红色框表示预测位置与真实纸条不一致。红框是预测完成后的评测标记，不会反馈给算法。")
traditional, learned = st.columns(2)
with traditional:
    st.image(
        reconstruction_diagnostic(case.shredded.strips, case.baseline_order, case.shredded.original_indices),
        caption="传统方法：边缘兼容度 + 有限上下对齐 + 全局优化",
        width="stretch",
    )
    _metrics_panel("传统方法", case.baseline_metrics, case.baseline_seconds)
    st.info(_reason(case, case.baseline_metrics, case.baseline_order, "传统方法"))
with learned:
    st.image(
        reconstruction_diagnostic(case.shredded.strips, case.learned_order, case.shredded.original_indices),
        caption=f"{model_version}：学习局部兼容度 + 全局优化",
        width="stretch",
    )
    _metrics_panel(model_version, case.learned_metrics, case.learned_seconds)
    st.info(_reason(case, case.learned_metrics, case.learned_order, model_version))

st.header("4. 结果解读")
if case.baseline_metrics["adjacency_accuracy"] > case.learned_metrics["adjacency_accuracy"]:
    st.warning("本次传统方法更好。学习模型并不保证在每个文档条件下胜出，这正是本项目保留对照的原因。")
elif case.baseline_metrics["adjacency_accuracy"] < case.learned_metrics["adjacency_accuracy"]:
    st.success("本次学习模型的相邻恢复率更高；请结合红框和预测顺序观察它改善了哪些局部接缝。")
else:
    st.info("两种方法本次相邻恢复率相同；可切换扰动条件或随机测试观察差异。")
st.caption("提示：失败不是被隐藏的异常结果。切换“失败示例”或“自定义”中的 edge_loss / vertical_shift 可用于课堂讨论。")
