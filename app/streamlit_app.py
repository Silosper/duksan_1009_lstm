"""Streamlit operations dashboard for the LSTM-AE anomaly detector."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duksan_lstm_ae.config import (
    DEFAULT_CONFIG_PATH,
    artifact_path,
    load_config,
    save_config,
)


st.set_page_config(
    page_title="Duksan LSTM-AE",
    page_icon="📈",
    layout="wide",
)


def _load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_exists(config: dict) -> bool:
    required = [
        artifact_path(config, "history_file"),
        artifact_path(config, "threshold_file"),
        artifact_path(config, "metrics_file"),
        artifact_path(config, "evaluation_file"),
    ]
    return all(path.exists() for path in required)


def _start_background(command: list[str], log_name: str) -> int:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    (log_dir / f"{log_name}.pid").write_text(
        str(process.pid),
        encoding="utf-8",
    )
    return process.pid


def _tail(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return "아직 로그가 없습니다."
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def show_dashboard(config: dict) -> None:
    st.title("LSTM-AE 이상탐지 대시보드")
    if not _artifact_exists(config):
        st.info(
            "학습 결과가 아직 없습니다. 왼쪽 메뉴의 실행 관리에서 "
            "모델 학습을 시작하세요."
        )
        return

    history = _load_table(artifact_path(config, "history_file"))
    metrics = _load_json(artifact_path(config, "metrics_file"))
    evaluation = _load_table(artifact_path(config, "evaluation_file"))
    evaluation["timestamp"] = pd.to_datetime(evaluation["timestamp"])
    threshold = float(metrics["threshold"])

    label_metrics_available = bool(metrics.get("label_metrics_available", False))
    metric_columns = st.columns(6)
    values = [
        ("Threshold", threshold, ".5f"),
        ("탐지율", metrics.get("detection_rate"), ".3f"),
        ("F1", metrics.get("f1"), ".3f"),
        ("Recall", metrics.get("recall"), ".3f"),
        ("Precision", metrics.get("precision"), ".3f"),
        ("ROC-AUC", metrics.get("roc_auc"), ".3f"),
    ]
    for column, (label, value, format_spec) in zip(metric_columns, values):
        text = "N/A" if value is None else format(value, format_spec)
        column.metric(label, text)

    st.subheader("학습 및 검증 손실")
    loss_long = history.melt(
        id_vars=["epoch"],
        value_vars=["train_loss", "validation_loss"],
        var_name="dataset",
        value_name="MSE",
    )
    st.plotly_chart(
        px.line(
            loss_long,
            x="epoch",
            y="MSE",
            color="dataset",
            markers=True,
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("재구성 오차 분포")
        display_distribution = evaluation.copy()
        color_column = None
        if label_metrics_available:
            display_distribution["설정 기반 라벨"] = (
                display_distribution["label"].map(
                    {0: "configured normal", 1: "configured anomaly"}
                )
            )
            color_column = "설정 기반 라벨"
        histogram = px.histogram(
            display_distribution,
            x="anomaly_score",
            color=color_column,
            barmode="overlay",
            opacity=0.60,
            nbins=100,
        )
        histogram.add_vline(
            x=threshold,
            line_color="red",
            line_dash="dash",
            annotation_text="threshold",
        )
        st.plotly_chart(histogram, use_container_width=True)

    with right:
        if label_metrics_available:
            st.subheader("혼동행렬")
            matrix = np.asarray(metrics["confusion_matrix"])
            confusion = px.imshow(
                matrix,
                x=["예측 비탐지", "예측 이상"],
                y=["설정 정상", "설정 이상"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            st.plotly_chart(confusion, use_container_width=True)
        else:
            st.subheader("라벨 기반 평가")
            st.info(
                "STATUS의 normal_values와 anomaly_values가 명시되지 않아 "
                "혼동행렬과 F1을 계산하지 않았습니다."
            )
            st.metric("탐지 윈도우", f"{metrics['detected_count']:,}")

    st.subheader("시간대별 이상 점수")
    max_points = int(config["streamlit"]["max_timeline_points"])
    step = max(1, int(np.ceil(len(evaluation) / max_points)))
    timeline = evaluation.iloc[::step].copy()
    timeline_chart = go.Figure()
    timeline_chart.add_trace(
        go.Scatter(
            x=timeline["timestamp"],
            y=timeline["anomaly_score"],
            mode="lines",
            name="anomaly score",
            line={"color": "#4C78A8", "width": 1},
        )
    )
    detected = timeline.loc[timeline["prediction"].eq(1)]
    timeline_chart.add_trace(
        go.Scatter(
            x=detected["timestamp"],
            y=detected["anomaly_score"],
            mode="markers",
            name="detected anomaly",
            marker={"color": "#E45756", "size": 5},
        )
    )
    timeline_chart.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="threshold",
    )
    timeline_chart.update_layout(
        xaxis_title="시간",
        yaxis_title="재구성 오차",
        height=450,
    )
    st.plotly_chart(timeline_chart, use_container_width=True)

    curve_left, curve_right = st.columns(2)
    known_evaluation = evaluation.loc[evaluation["label"].isin([0, 1])]
    if label_metrics_available and known_evaluation["label"].nunique() == 2:
        fpr, tpr, _ = roc_curve(
            known_evaluation["label"],
            known_evaluation["anomaly_score"],
        )
        precision, recall, _ = precision_recall_curve(
            known_evaluation["label"],
            known_evaluation["anomaly_score"],
        )
        with curve_left:
            st.subheader("ROC 곡선")
            roc_figure = px.line(x=fpr, y=tpr, labels={"x": "FPR", "y": "TPR"})
            roc_figure.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line={"dash": "dot", "color": "gray"},
            )
            st.plotly_chart(roc_figure, use_container_width=True)
        with curve_right:
            st.subheader("Precision-Recall 곡선")
            st.plotly_chart(
                px.line(
                    x=recall,
                    y=precision,
                    labels={"x": "Recall", "y": "Precision"},
                ),
                use_container_width=True,
            )

    st.subheader("센서별 평균 재구성 오차")
    error_columns = [
        column for column in evaluation.columns if column.startswith("error_")
    ]
    if label_metrics_available:
        sensor_summary = (
            known_evaluation.groupby("label")[error_columns]
            .mean()
            .T.rename(columns={0: "configured normal", 1: "configured anomaly"})
            .rename_axis("sensor")
            .reset_index()
        )
    else:
        sensor_summary = (
            evaluation[error_columns]
            .mean()
            .to_frame("all windows")
            .rename_axis("sensor")
            .reset_index()
        )
    sensor_summary["sensor"] = sensor_summary["sensor"].str.replace(
        "error_",
        "",
        regex=False,
    )
    sensor_long = sensor_summary.melt(
        id_vars="sensor",
        var_name="구분",
        value_name="평균 MSE",
    )
    st.plotly_chart(
        px.bar(
            sensor_long,
            x="sensor",
            y="평균 MSE",
            color="구분",
            barmode="group",
        ),
        use_container_width=True,
    )

    st.subheader("이상 점수 상위 구간")
    columns = [
        "timestamp",
        "anomaly_score",
        "label",
        "prediction",
        *error_columns,
    ]
    st.dataframe(
        evaluation.nlargest(30, "anomaly_score")[columns],
        use_container_width=True,
        hide_index=True,
    )


def show_settings(config: dict) -> None:
    st.title("파라미터 설정")
    st.caption(
        "여기서 저장한 값은 config/settings.yaml에 반영됩니다. "
        "모델 구조나 윈도우 길이를 변경하면 다시 학습해야 합니다."
    )
    with st.form("settings_form"):
        st.subheader("윈도우")
        col1, col2, col3 = st.columns(3)
        sequence_length = col1.number_input(
            "시퀀스 길이(분)",
            min_value=5,
            value=int(config["window"]["sequence_length"]),
        )
        stride = col2.number_input(
            "Stride",
            min_value=1,
            value=int(config["window"]["stride"]),
        )
        label_modes = ["last", "any_anomaly", "majority", "all"]
        label_mode = col3.selectbox(
            "STATUS 윈도우 기준",
            label_modes,
            index=label_modes.index(config["status"]["window_label_mode"]),
        )
        max_train = st.number_input(
            "최대 학습 윈도우 수(0=전체)",
            min_value=0,
            value=int(config["window"]["max_train_windows"] or 0),
            step=10000,
        )
        col1, col2 = st.columns(2)
        max_validation = col1.number_input(
            "최대 검증 윈도우 수(0=전체)",
            min_value=0,
            value=int(config["window"]["max_validation_windows"] or 0),
            step=5000,
        )
        max_test = col2.number_input(
            "최대 테스트 윈도우 수(0=전체)",
            min_value=0,
            value=int(config["window"]["max_test_windows"] or 0),
            step=10000,
        )

        st.subheader("모델")
        col1, col2, col3, col4 = st.columns(4)
        hidden_size = col1.number_input(
            "Hidden size",
            min_value=8,
            value=int(config["model"]["hidden_size"]),
        )
        latent_size = col2.number_input(
            "Latent size",
            min_value=2,
            value=int(config["model"]["latent_size"]),
        )
        num_layers = col3.number_input(
            "LSTM layers",
            min_value=1,
            value=int(config["model"]["num_layers"]),
        )
        dropout = col4.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.9,
            value=float(config["model"]["dropout"]),
            step=0.05,
        )

        st.subheader("학습 및 임계값")
        col1, col2, col3, col4 = st.columns(4)
        batch_size = col1.number_input(
            "Batch size",
            min_value=1,
            value=int(config["training"]["batch_size"]),
        )
        epochs = col2.number_input(
            "Epochs",
            min_value=1,
            value=int(config["training"]["epochs"]),
        )
        learning_rate = col3.number_input(
            "Learning rate",
            min_value=0.000001,
            value=float(config["training"]["learning_rate"]),
            format="%.6f",
        )
        percentile = col4.number_input(
            "Threshold percentile",
            min_value=90.0,
            max_value=99.99,
            value=float(config["threshold"]["percentile"]),
            step=0.05,
            format="%.2f",
        )
        patience = st.number_input(
            "Early stopping patience",
            min_value=1,
            value=int(config["training"]["patience"]),
        )
        col1, col2, col3 = st.columns(3)
        weight_decay = col1.number_input(
            "Weight decay",
            min_value=0.0,
            value=float(config["training"]["weight_decay"]),
            format="%.6f",
        )
        gradient_clip = col2.number_input(
            "Gradient clip",
            min_value=0.1,
            value=float(config["training"]["gradient_clip_norm"]),
            step=0.1,
        )
        device = col3.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=["auto", "cpu", "cuda"].index(
                config["training"]["device"]
            ),
        )
        col1, col2, col3 = st.columns(3)
        scaling_methods = ["robust", "standard", "minmax", "none"]
        scaling_method = col1.selectbox(
            "Scaler",
            scaling_methods,
            index=scaling_methods.index(config["scaling"]["method"]),
        )
        threshold_methods = [
            "validation_percentile",
            "train_percentile",
            "mean_std",
        ]
        threshold_method = col2.selectbox(
            "Threshold 방식",
            threshold_methods,
            index=threshold_methods.index(config["threshold"]["method"]),
        )
        purge_gap_steps = col3.number_input(
            "Split purge gap(step)",
            min_value=0,
            value=int(config["split"]["purge_gap_steps"]),
        )

        st.subheader("STATUS 의미 설정")
        use_status_filter = st.checkbox(
            "설정한 정상 STATUS만 학습에 사용",
            value=bool(config["status"]["use_for_training_filter"]),
        )
        normal_values_text = st.text_input(
            "정상 STATUS 값(쉼표 구분, 모르면 비움)",
            value=", ".join(map(str, config["status"]["normal_values"])),
        )
        anomaly_values_text = st.text_input(
            "이상 STATUS 값(쉼표 구분, 모르면 비움)",
            value=", ".join(map(str, config["status"]["anomaly_values"])),
        )
        submitted = st.form_submit_button("설정 저장", type="primary")

    if submitted:
        config["window"]["sequence_length"] = int(sequence_length)
        config["window"]["stride"] = int(stride)
        config["status"]["window_label_mode"] = label_mode
        config["window"]["max_train_windows"] = (
            None if int(max_train) == 0 else int(max_train)
        )
        config["window"]["max_validation_windows"] = (
            None if int(max_validation) == 0 else int(max_validation)
        )
        config["window"]["max_test_windows"] = (
            None if int(max_test) == 0 else int(max_test)
        )
        config["model"].update(
            {
                "hidden_size": int(hidden_size),
                "latent_size": int(latent_size),
                "num_layers": int(num_layers),
                "dropout": float(dropout),
            }
        )
        config["training"].update(
            {
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "patience": int(patience),
                "weight_decay": float(weight_decay),
                "gradient_clip_norm": float(gradient_clip),
                "device": device,
            }
        )
        config["scaling"]["method"] = scaling_method
        config["threshold"]["method"] = threshold_method
        config["threshold"]["percentile"] = float(percentile)
        config["split"]["purge_gap_steps"] = int(purge_gap_steps)
        config["status"]["use_for_training_filter"] = bool(use_status_filter)
        config["status"]["normal_values"] = [
            value.strip()
            for value in normal_values_text.split(",")
            if value.strip()
        ]
        config["status"]["anomaly_values"] = [
            value.strip()
            for value in anomaly_values_text.split(",")
            if value.strip()
        ]
        save_config(config, DEFAULT_CONFIG_PATH)
        st.success("config/settings.yaml에 저장했습니다.")

    st.download_button(
        "현재 설정 다운로드",
        data=DEFAULT_CONFIG_PATH.read_bytes(),
        file_name="settings.yaml",
        mime="application/x-yaml",
    )


def show_operations(config: dict) -> None:
    st.title("실행 관리")
    st.warning(
        "학습은 서버 자원을 오래 사용할 수 있습니다. GPU 서버에서는 "
        "config의 training.device를 auto 또는 cuda로 설정하세요."
    )
    left, right = st.columns(2)
    with left:
        if st.button("모델 학습 시작", type="primary"):
            pid = _start_background(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(DEFAULT_CONFIG_PATH),
                ],
                "train.log",
            )
            st.success(f"학습을 시작했습니다. PID={pid}")
        if st.button("저장 모델 재평가"):
            pid = _start_background(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--config",
                    str(DEFAULT_CONFIG_PATH),
                ],
                "evaluate.log",
            )
            st.success(f"평가를 시작했습니다. PID={pid}")

    with right:
        uploaded = st.file_uploader(
            "추론용 전처리 CSV",
            type=["csv"],
        )
        input_is_scaled = st.checkbox(
            "과거 전체 데이터 scaler로 이미 변환된 입력",
            value=False,
        )
        if uploaded is not None and st.button("업로드 데이터 추론 시작"):
            incoming_dir = PROJECT_ROOT / "data/incoming"
            incoming_dir.mkdir(parents=True, exist_ok=True)
            input_path = incoming_dir / uploaded.name
            input_path.write_bytes(uploaded.getvalue())
            command = [
                sys.executable,
                "scripts/infer.py",
                "--config",
                str(DEFAULT_CONFIG_PATH),
                "--input",
                str(input_path),
            ]
            command.append(
                "--input-is-scaled"
                if input_is_scaled
                else "--no-input-is-scaled"
            )
            pid = _start_background(command, "inference.log")
            st.success(f"추론을 시작했습니다. PID={pid}")

    st.subheader("실행 로그")
    selected_log = st.selectbox(
        "로그 선택",
        ["train.log", "evaluate.log", "inference.log"],
    )
    st.code(_tail(PROJECT_ROOT / "logs" / selected_log), language="text")
    st.caption("새 결과를 보려면 브라우저를 새로고침하세요.")


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    st.sidebar.title("Duksan LSTM-AE")
    page = st.sidebar.radio(
        "메뉴",
        ["대시보드", "파라미터 설정", "실행 관리"],
    )
    st.sidebar.caption(f"설정: {DEFAULT_CONFIG_PATH.name}")

    if page == "대시보드":
        show_dashboard(config)
    elif page == "파라미터 설정":
        show_settings(config)
    else:
        show_operations(config)


if __name__ == "__main__":
    main()
