"""Classroom-oriented Streamlit demonstration for the V0.4 prototype."""

from __future__ import annotations

import io
import json
from pathlib import Path
import random
from time import time_ns

from PIL import Image
import streamlit as st

from shredded_reconstruction.conditions import CONDITION_NAMES, condition_description, observation_condition
from shredded_reconstruction.document import DOCUMENT_VARIANTS
from shredded_reconstruction.evaluation import CaseResult, run_case, run_image_case
from shredded_reconstruction.learned import train_compatibility_model, train_v03_compatibility_model
from shredded_reconstruction.splits import TRAIN_DOCUMENTS
from shredded_reconstruction.v03_splits import V03_TRAIN_DOCUMENTS
from shredded_reconstruction.visualize import reconstruction_diagnostic, shuffled_montage, wrong_positions


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="碎纸文档自动重建 | V0.4", layout="wide")

SHOWCASES = {
    "成功示例": {"seed": 5100, "variant": "report", "strip_count": 8, "condition": "clean", "shuffle_seed": 86713},
    "失败示例": {"seed": 5101, "variant": "report", "strip_count": 10, "condition": "edge_loss", "shuffle_seed": 110487},
}


@st.cache_resource(show_spinner="正在训练轻量级兼容度模型…")
def get_model(version: str):
    if version == "V0.2 原始学习模型":
        return train_compatibility_model(TRAIN_DOCUMENTS)
    return train_v03_compatibility_model(
        V03_TRAIN_DOCUMENTS,
        augmentations=tuple(observation_condition(name) for name in CONDITION_NAMES),
        endpoint_multiplier=0.0,
    )


def _random_spec() -> dict[str, int | str]:
    rng = random.Random(time_ns())
    return {
        "seed": rng.randint(9000, 9999), "variant": rng.choice(DOCUMENT_VARIANTS),
        "strip_count": rng.choice((8, 10, 12)), "condition": rng.choice(CONDITION_NAMES),
        "shuffle_seed": rng.randint(1, 999_999),
    }


def _reason(case: CaseResult, metrics: dict, order: list[int], method_name: str) -> str:
    if metrics["exact_match"]:
        return f"{method_name} 完整恢复成功：所有纸条都在正确位置。"
    reasons = {
        "edge_loss": "纸条边缘有永久像素缺失，接缝证据被削弱。",
        "vertical_shift": "纸条上下错位会让同一行文字和线条无法直接对齐。",
        "no_assists": "关闭边框和连续纹理后，首尾方向更容易产生歧义。",
        "scan": "亮度差和轻度模糊降低了边缘局部对比度。",
        "clean": "重复文字、表格线或空白区域会让多个局部接缝得分接近。",
    }
    positions = "、".join(map(str, wrong_positions(order, case.shredded.original_indices)))
    return f"红框位置：{positions}。可能原因：{reasons.get(case.condition, reasons['clean'])}"


def _run_synthetic(spec: dict[str, int | str], version: str) -> CaseResult:
    with st.status("正在执行真实重建…", expanded=True) as status:
        st.write("1 / 3 生成文档并按设定条件切成纸条")
        st.write("2 / 3 运行传统局部兼容度 + 全局优化")
        st.write("3 / 3 运行学习兼容度 + 全局优化")
        result = run_case(
            seed=int(spec["seed"]), variant=str(spec["variant"]), strip_count=int(spec["strip_count"]),
            shuffle_seed=int(spec["shuffle_seed"]), model=get_model(version), condition_name=str(spec["condition"]),
            baseline_max_vertical_shift=3,
        )
        status.update(label="真实重建完成", state="complete", expanded=False)
    return result


def _run_uploaded(image: Image.Image, spec: dict[str, int | str], version: str) -> CaseResult:
    with st.status("正在将上传页面切条、打乱并重建…", expanded=True) as status:
        st.write("上传图片的真实左右顺序不会输入给任何预测算法。")
        result = run_image_case(
            image, strip_count=int(spec["strip_count"]), shuffle_seed=int(spec["shuffle_seed"]),
            model=get_model(version), condition_name=str(spec["condition"]), baseline_max_vertical_shift=3,
        )
        status.update(label="上传页面的重建完成", state="complete", expanded=False)
    return result


def _metrics_panel(title: str, metrics: dict, seconds: float | None) -> None:
    st.subheader(title)
    first, second, third = st.columns(3)
    first.metric("相邻关系正确率", f"{metrics['adjacency_accuracy']:.1%}")
    second.metric("完整恢复", "是" if metrics["exact_match"] else "否")
    third.metric("实际运行耗时", f"{seconds or 0.0:.3f} 秒")
    st.caption(f"预测原始顺序：{metrics['predicted_original_order']}")


def _result_chart() -> None:
    path = ROOT / "artifacts" / "v0.3" / "evaluation.json"
    if not path.exists():
        st.caption("完成固定 V0.3 评测后，这里会显示其实际汇总图表。")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {name: values.get("document_mean_adjacency", 0.0) for name, values in data.get("final_test_summary", {}).items()}
    if rows:
        st.bar_chart(rows, horizontal=True)
        st.caption("固定 V0.3 最终测试的文档平均相邻恢复率；不是本页临时示例的数字。")
    v04_chart = ROOT / "artifacts" / "v0.4" / "ablation-confirmation.png"
    if v04_chart.exists():
        st.subheader("V0.4 独立确认：关键对照实验")
        st.image(str(v04_chart), caption="新文档族上的固定消融：完整学习配置不保证超过传统方法；图中数字来自实际确认运行。", width="stretch")


st.title("碎纸文档自动重建：从打乱纸条到全局拼接")
st.caption("V0.3 课堂演示页｜准确率与耗时均来自当前实际运行；红框仅在预测完成后用于评测解释。")
st.info("应用背景：该类技术可为司法碎纸证据恢复、受损档案整理和历史文献数字化提供辅助。当前项目仍是规则竖直纸条的实验原型，不处理旋转、重叠、任意形状碎片或物理碎纸扫描。")

with st.sidebar:
    st.header("1. 选择输入")
    mode = st.radio("课堂一键示例 / 自定义 / 上传", ("成功示例", "失败示例", "自定义合成页", "上传文档图片"))
    model_version = st.radio("机器学习对照模型", ("V0.3 改进学习模型", "V0.2 原始学习模型"))
    uploaded = None
    if mode in SHOWCASES:
        spec = SHOWCASES[mode].copy()
        (st.success if mode == "成功示例" else st.warning)("固定案例来自真实运行；失败案例被保留用于课堂讨论。")
    elif mode == "上传文档图片":
        uploaded = st.file_uploader("上传 PNG/JPG 完整文档页面", type=("png", "jpg", "jpeg"))
        strip_count = st.select_slider("纸条数量", options=(8, 10, 12), value=10)
        condition = st.selectbox("额外模拟干扰", CONDITION_NAMES, format_func=condition_description)
        shuffle_seed = st.number_input("打乱种子", min_value=0, value=73, step=1)
        spec = {"strip_count": int(strip_count), "condition": condition, "shuffle_seed": int(shuffle_seed)}
        st.caption("上传页会先被程序切条并打乱；真实顺序只在最终指标计算时使用。")
    else:
        if "random_spec" not in st.session_state:
            st.session_state.random_spec = _random_spec()
        if st.button("随机生成一组测试", width="stretch"):
            st.session_state.random_spec = _random_spec()
        current = st.session_state.random_spec
        variant = st.selectbox("文档类型", DOCUMENT_VARIANTS, index=DOCUMENT_VARIANTS.index(str(current["variant"])))
        strip_count = st.select_slider("纸条数量", options=(8, 10, 12), value=int(current["strip_count"]))
        condition = st.selectbox("图像条件", CONDITION_NAMES, format_func=condition_description, index=CONDITION_NAMES.index(str(current["condition"])))
        seed = st.number_input("文档种子", min_value=0, value=int(current["seed"]), step=1)
        shuffle_seed = st.number_input("打乱种子", min_value=0, value=int(current["shuffle_seed"]), step=1)
        spec = {"seed": int(seed), "variant": variant, "strip_count": int(strip_count), "condition": condition, "shuffle_seed": int(shuffle_seed)}
    run = st.button("开始：打乱并自动重建", type="primary", width="stretch", disabled=(mode == "上传文档图片" and uploaded is None))

spec_key = tuple(sorted(spec.items())) + (model_version, mode)
if run or "case" not in st.session_state:
    if mode == "上传文档图片" and uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
        st.session_state.case = _run_uploaded(image, spec, model_version)
    else:
        st.session_state.case = _run_synthetic(spec if "seed" in spec else SHOWCASES["成功示例"], model_version)
    st.session_state.case_key = spec_key

case: CaseResult = st.session_state.case
if st.session_state.get("case_key") != spec_key:
    st.caption("配置已修改；点击“开始：打乱并自动重建”运行新案例。")

st.header("流程总览：选择文档 → 打乱纸条 → 自动重建 → 查看结果")
for column, number, text in zip(st.columns(4), ("1", "2", "3", "4"), ("选择文档与干扰", "随机打乱纸条", "两种方法真实重建", "比较指标并解释错误"), strict=True):
    column.markdown(f"### {number}. {text}")

st.header("2. 输入：原始文档与打乱纸条")
left, right = st.columns(2)
left.image(case.original, caption="原始文档（正确顺序只用于预测完成后的评测）", width="stretch")
right.image(shuffled_montage(case.shredded.strips), caption="打乱后的输入：上方数字只是输入位置，不是正确顺序", width="stretch")

st.header("3. 自动重建：传统方法与机器学习方法")
st.caption("绿色框表示该位置正确；红色框表示预测完成后发现的位置错误。这些标记绝不会反馈给算法。")
traditional, learned = st.columns(2)
with traditional:
    st.image(reconstruction_diagnostic(case.shredded.strips, case.baseline_order, case.shredded.original_indices), caption="传统 V0.2：边缘兼容度 + 有限上下对齐 + 全局优化", width="stretch")
    _metrics_panel("传统 V0.2", case.baseline_metrics, case.baseline_seconds)
    st.info(_reason(case, case.baseline_metrics, case.baseline_order, "传统 V0.2"))
with learned:
    st.image(reconstruction_diagnostic(case.shredded.strips, case.learned_order, case.shredded.original_indices), caption=f"{model_version}：学习兼容度 + 全局优化", width="stretch")
    _metrics_panel(model_version, case.learned_metrics, case.learned_seconds)
    st.info(_reason(case, case.learned_metrics, case.learned_order, model_version))

st.header("4. 结果解读")
if case.baseline_metrics["adjacency_accuracy"] > case.learned_metrics["adjacency_accuracy"]:
    st.warning("本次传统方法更好。保留这种结果，是为了避免把学习方法误说成必然更强。")
elif case.baseline_metrics["adjacency_accuracy"] < case.learned_metrics["adjacency_accuracy"]:
    st.success("本次学习模型的相邻恢复率更高；可结合红框观察它改善了哪些局部接缝。")
else:
    st.info("两种方法本次相邻恢复率相同；可切换干扰条件或上传不同页面观察差异。")
st.subheader("固定 V0.3 实验结果图表")
_result_chart()
