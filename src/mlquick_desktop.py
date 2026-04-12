from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from PySide6.QtCore import QEvent, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QFrame,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from mlquick_core.io_utils import load_dataset
from mlquick_core.models import TrainingConfig, TrainingResult
from mlquick_core.prediction import PredictionService
from mlquick_core.profiling import profile_dataset
from mlquick_core.registry import ModelRegistry
from mlquick_core.training import TrainingService



def resolve_app_dir() -> Path:
    candidates = [
        Path.cwd() / "workspace",
        Path.home() / "MLquickWorkspace",
        Path(tempfile.gettempdir()) / "MLquickWorkspace",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path.cwd()


APP_DIR = resolve_app_dir()


def dataframe_to_table(table: QTableWidget, data: pd.DataFrame | None) -> None:
    if data is None or data.empty:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
        return

    frame = data.copy()
    table.setRowCount(len(frame))
    table.setColumnCount(len(frame.columns))
    table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
    for row_index, (_, row) in enumerate(frame.iterrows()):
        for column_index, value in enumerate(row):
            table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()


def configure_table(table: QTableWidget) -> None:
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


def create_section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("sectionLabel", True)
    return label


def create_metric_value_label() -> QLabel:
    label = QLabel("--")
    label.setProperty("metricValue", True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return label


def create_hint_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("hintLabel", True)
    return label


def create_detail_value_label() -> QLabel:
    label = QLabel("--")
    label.setProperty("detailValue", True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    label.setWordWrap(True)
    return label


def create_stat_value_label() -> QLabel:
    label = QLabel("--")
    label.setProperty("statValue", True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return label


class TrainingWorker(QThread):
    log_emitted = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, service: TrainingService, data: pd.DataFrame, config: TrainingConfig):
        super().__init__()
        self.service = service
        self.data = data
        self.config = config

    def run(self) -> None:
        try:
            result = self.service.train(self.data, self.config, progress=self.log_emitted.emit)
            self.completed.emit(result)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class PredictionWorker(QThread):
    progress_changed = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, service: PredictionService, model_name: str, data: pd.DataFrame):
        super().__init__()
        self.service = service
        self.model_name = model_name
        self.data = data

    def run(self) -> None:
        try:
            self.progress_changed.emit(10, "正在加载模型...")
            self.progress_changed.emit(35, "正在执行批量预测...")
            result = self.service.predict(self.model_name, self.data)
            self.progress_changed.emit(85, "正在整理预测结果...")
            self.completed.emit(result)
            self.progress_changed.emit(100, "预测完成")
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class DesktopMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.registry = ModelRegistry(APP_DIR)
        self.training_service = TrainingService(self.registry)
        self.prediction_service = PredictionService(self.registry)
        self.current_data: pd.DataFrame | None = None
        self.current_data_path: Path | None = None
        self.current_profile = None
        self.training_worker: TrainingWorker | None = None
        self.prediction_worker: PredictionWorker | None = None
        self.pending_prediction_model_name: str | None = None
        self.last_export_frame: pd.DataFrame | None = None
        self.all_model_metadata = []
        self.model_metadata_lookup: dict[str, object] = {}
        self.history_records: list[str] = []
        self._no_wheel_widgets: set[QWidget] = set()
        self._current_plot_path: Path | None = None
        self.app_mode: str = "train"

        self.setWindowTitle("MLquick 桌面版")
        self.resize(1180, 700)
        self.setMinimumSize(1080, 660)
        self._apply_styles()
        self._build_ui()
        self.refresh_model_list()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI", Arial;
                font-size: 15px;
                color: #4d4c48;
            }
            QMainWindow {
                background: #f5f4ed;
            }
            QGroupBox {
                font-weight: 500;
                font-family: Georgia;
                border: 1px solid #f0eee6;
                border-radius: 12px;
                margin-top: 14px;
                padding-top: 14px;
                background: #faf9f5;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
                color: #141413;
            }
            QPushButton {
                min-height: 36px;
                min-width: 120px;
                padding: 8px 14px;
                border-radius: 10px;
                background: #e8e6dc;
                border: 1px solid #d1cfc5;
                color: #4d4c48;
            }
            QPushButton:hover {
                background: #f0eee6;
            }
            QPushButton[variant="primary"] {
                background: #c96442;
                color: #faf9f5;
                border: 1px solid #c96442;
                font-weight: 600;
            }
            QPushButton[variant="primary"]:hover {
                background: #d97757;
            }
            QPushButton[variant="secondary"] {
                background: #e8e6dc;
                color: #4d4c48;
                border: 1px solid #d1cfc5;
            }
            QPushButton[variant="danger"] {
                background: #f5d9d6;
                color: #b42318;
                border: 1px solid #d8a9a4;
            }
            QPushButton[variant="danger"]:hover {
                background: #efc9c5;
            }
            QPushButton[modeSwitch="true"] {
                min-height: 30px;
                min-width: 76px;
                padding: 6px 12px;
                border-radius: 16px;
                background: #ece9df;
                border: 1px solid #d7d1c2;
                color: #5b5a54;
                font-weight: 600;
            }
            QPushButton[modeSwitch="true"]:hover {
                background: #f2efe6;
            }
            QPushButton[modeSwitch="true"]:checked {
                background: #c96442;
                border: 1px solid #c96442;
                color: #faf9f5;
            }
            QPushButton[modeSwitch="true"]:checked:hover {
                background: #d97757;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #e8e6dc;
                border-radius: 12px;
                background: #faf9f5;
                padding: 3px 10px;
                color: #141413;
            }
            QPlainTextEdit, QListWidget, QTableWidget {
                border: 1px solid #e8e6dc;
                border-radius: 12px;
                background: #faf9f5;
                selection-background-color: #e8e6dc;
                selection-color: #141413;
                padding: 4px 6px;
            }
            QLabel {
                color: #5e5d59;
            }
            QLabel[sectionLabel="true"] {
                font-family: Georgia;
                font-size: 18px;
                font-weight: 500;
                color: #141413;
                padding: 2px 0 0 0;
            }
            QLabel[metricTitle="true"] {
                font-size: 12px;
                color: #87867f;
            }
            QLabel[metricValue="true"] {
                font-family: Georgia;
                font-size: 24px;
                font-weight: 500;
                color: #141413;
                padding-top: 2px;
            }
            QLabel[detailTitle="true"] {
                font-size: 12px;
                color: #87867f;
            }
            QLabel[detailValue="true"] {
                font-size: 14px;
                font-weight: 600;
                color: #3d3d3a;
            }
            QLabel[statValue="true"] {
                font-family: Georgia;
                font-size: 22px;
                font-weight: 500;
                color: #141413;
            }
            QLabel[hintLabel="true"] {
                color: #5e5d59;
                line-height: 1.4;
            }
            QLabel[brandTitle="true"] {
                font-family: "Georgia";
                font-size: 34px;
                font-weight: 700;
                color: #141413;
                letter-spacing: 1px;
                padding: 0;
            }
            QLabel[brandTagline="true"] {
                font-family: "KaiTi", "STKaiti", "Georgia";
                font-size: 12px;
                color: #87867f;
                font-style: italic;
                padding: 0 0 2px 0;
            }
            QLabel[brandLogo="true"] {
                background: #faf9f5;
                border: 1px solid #ece9df;
                border-radius: 12px;
                padding: 3px;
            }
            QFrame[card="true"] {
                background: #faf9f5;
                border: 1px solid #f0eee6;
                border-radius: 16px;
            }
            QTabWidget::pane {
                border: 1px solid #f0eee6;
                border-radius: 12px;
                background: #faf9f5;
                margin-top: 8px;
            }
            QTabBar::tab {
                background: #e8e6dc;
                color: #5e5d59;
                border: 1px solid #e8e6dc;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 14px;
                margin-right: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: #faf9f5;
                color: #141413;
                border-color: #f0eee6;
            }
            QHeaderView::section {
                background: #f0eee6;
                color: #4d4c48;
                border: none;
                border-right: 1px solid #e8e6dc;
                padding: 6px 8px;
                font-weight: 500;
            }
            """
        )

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        layout.addWidget(splitter)
        self.setCentralWidget(root)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(400)

        brand_header = QWidget()
        brand_header_layout = QHBoxLayout(brand_header)
        brand_header_layout.setContentsMargins(2, 0, 2, 2)
        brand_header_layout.setSpacing(8)
        brand_logo = QLabel()
        brand_logo.setProperty("brandLogo", True)
        brand_logo.setFixedSize(42, 42)
        brand_logo.setAlignment(Qt.AlignCenter)
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "mlquick-logo.svg"
        if logo_path.exists():
            logo_pixmap = QPixmap(str(logo_path))
            if not logo_pixmap.isNull():
                brand_logo.setPixmap(logo_pixmap.scaled(34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        brand_header_layout.addWidget(brand_logo, 0, Qt.AlignTop)

        brand_text = QWidget()
        brand_text_layout = QVBoxLayout(brand_text)
        brand_text_layout.setContentsMargins(0, 0, 0, 0)
        brand_text_layout.setSpacing(5)
        brand_title = QLabel("MLquick")
        brand_title.setProperty("brandTitle", True)
        brand_text_layout.addWidget(brand_title)
        brand_tagline = QLabel("且将新火试新茶，诗酒趁年华。")
        brand_tagline.setProperty("brandTagline", True)
        brand_text_layout.addWidget(brand_tagline)
        brand_header_layout.addWidget(brand_text, 1)
        left_layout.addWidget(brand_header)

        mode_row = QWidget()
        mode_row_layout = QHBoxLayout(mode_row)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)
        mode_row_layout.setSpacing(8)
        mode_row_layout.addWidget(QLabel("工作模式"))
        mode_switch = QWidget()
        mode_switch_layout = QHBoxLayout(mode_switch)
        mode_switch_layout.setContentsMargins(0, 0, 0, 0)
        mode_switch_layout.setSpacing(6)
        self.mode_train_button = QPushButton("训练")
        self.mode_train_button.setCheckable(True)
        self.mode_train_button.setProperty("modeSwitch", "true")
        self.mode_predict_button = QPushButton("预测")
        self.mode_predict_button.setCheckable(True)
        self.mode_predict_button.setProperty("modeSwitch", "true")
        mode_switch_layout.addWidget(self.mode_train_button)
        mode_switch_layout.addWidget(self.mode_predict_button)
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.mode_train_button)
        self.mode_button_group.addButton(self.mode_predict_button)
        self.mode_train_button.clicked.connect(lambda checked: checked and self.set_app_mode("train"))
        self.mode_predict_button.clicked.connect(lambda checked: checked and self.set_app_mode("predict"))
        mode_row_layout.addWidget(mode_switch)
        mode_row_layout.addStretch(1)
        left_layout.addWidget(mode_row)

        self.load_button = QPushButton("导入数据集")
        self.load_button.setProperty("variant", "primary")
        self.load_button.clicked.connect(self.load_dataset_file)
        left_layout.addWidget(self.load_button)

        self.dataset_label = QLabel("尚未加载数据集")
        self.dataset_label.setWordWrap(True)
        left_layout.addWidget(self.dataset_label)

        workflow_box = QGroupBox("操作步骤")
        workflow_layout = QVBoxLayout(workflow_box)
        workflow_layout.setContentsMargins(12, 16, 12, 12)
        workflow_layout.setSpacing(8)
        self.workflow_text = QPlainTextEdit()
        self.workflow_text.setReadOnly(True)
        self.workflow_text.setMaximumHeight(130)
        self.workflow_text.setPlainText(
            "\n".join(
                [
                    "1. 导入业务数据文件",
                    "2. 查看系统给出的任务建议",
                    "3. 选择目标列和参与训练字段",
                    "4. 点击开始训练生成模型",
                    "5. 在模型管理中执行批量预测",
                ]
            )
        )
        workflow_layout.addWidget(self.workflow_text)
        left_layout.addWidget(workflow_box)
        workflow_box.setVisible(False)

        self.profile_box = QGroupBox("数据体检")
        profile_layout = QVBoxLayout(self.profile_box)
        profile_layout.setContentsMargins(12, 16, 12, 12)
        profile_stats_grid = QGridLayout()
        profile_stats_grid.setContentsMargins(0, 0, 0, 0)
        profile_stats_grid.setHorizontalSpacing(10)
        profile_stats_grid.setVerticalSpacing(10)
        self.profile_stat_cards: dict[str, QLabel] = {}
        profile_titles = [
            ("rows", "总行数"),
            ("columns", "总列数"),
            ("numeric", "数值列"),
            ("categorical", "文本/类别列"),
            ("duplicates", "重复行"),
            ("targets", "候选目标列"),
        ]
        for index, (key, title) in enumerate(profile_titles):
            card = QFrame()
            card.setProperty("card", "true")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(2)
            title_label = QLabel(title)
            title_label.setProperty("detailTitle", True)
            card_layout.addWidget(title_label)
            value_label = create_stat_value_label()
            self.profile_stat_cards[key] = value_label
            card_layout.addWidget(value_label)
            profile_stats_grid.addWidget(card, index // 3, index % 3)
        profile_layout.addLayout(profile_stats_grid)
        self.profile_text = QPlainTextEdit()
        self.profile_text.setReadOnly(True)
        self.profile_text.setMinimumHeight(170)
        self.profile_text.setVisible(False)
        profile_layout.addWidget(self.profile_text)
        left_layout.addWidget(self.profile_box)

        self.config_box = QGroupBox("训练配置")
        config_layout = QFormLayout(self.config_box)
        config_layout.setContentsMargins(12, 18, 12, 12)
        config_layout.setSpacing(10)
        config_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.config_layout = config_layout
        self.config_row_index: dict[str, int] = {}
        self.recommendation_label = QLabel("请先导入数据集，系统会给出任务建议。")
        self.recommendation_label.setWordWrap(True)
        config_layout.addRow("任务建议", self.recommendation_label)
        self.config_row_index["recommendation"] = config_layout.rowCount() - 1

        self.task_combo = QComboBox()
        self.task_combo.addItem("分类", "classification")
        self.task_combo.addItem("回归", "regression")
        self.task_combo.addItem("聚类", "clustering")
        self.task_combo.currentIndexChanged.connect(self.toggle_task_inputs)
        config_layout.addRow("任务类型", self.task_combo)

        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self.refresh_feature_options)
        self.config_row_index["target"] = config_layout.rowCount()
        config_layout.addRow("目标列", self.target_combo)

        self.train_size_spin = QDoubleSpinBox()
        self.train_size_spin.setRange(0.1, 0.95)
        self.train_size_spin.setSingleStep(0.05)
        self.train_size_spin.setValue(0.7)
        self.config_row_index["train_size"] = config_layout.rowCount()
        config_layout.addRow("训练集比例", self.train_size_spin)

        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItem("自动对比候选模型", "auto_compare")
        self.training_mode_combo.addItem("手动指定模型", "manual_single")
        self.training_mode_combo.currentIndexChanged.connect(self.toggle_task_inputs)
        self.config_row_index["training_mode"] = config_layout.rowCount()
        config_layout.addRow("训练策略", self.training_mode_combo)

        self.manual_model_combo = QComboBox()
        self.config_row_index["manual_model"] = config_layout.rowCount()
        config_layout.addRow("指定模型", self.manual_model_combo)

        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 20)
        self.cluster_spin.setValue(3)
        self.config_row_index["cluster_count"] = config_layout.rowCount()
        config_layout.addRow("聚类数量", self.cluster_spin)

        self.preprocess_text_check = QCheckBox("启用文本预处理")
        self.preprocess_text_check.toggled.connect(self.toggle_task_inputs)
        self.config_row_index["preprocess_text"] = config_layout.rowCount()
        config_layout.addRow("", self.preprocess_text_check)

        self.remove_stopwords_check = QCheckBox("移除停用词")
        self.remove_stopwords_check.setChecked(True)
        self.config_row_index["remove_stopwords"] = config_layout.rowCount()
        config_layout.addRow("", self.remove_stopwords_check)

        self.min_word_length_spin = QSpinBox()
        self.min_word_length_spin.setRange(1, 5)
        self.min_word_length_spin.setValue(2)
        self.config_row_index["min_word_length"] = config_layout.rowCount()
        config_layout.addRow("最小词长度", self.min_word_length_spin)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.feature_list.setMinimumHeight(150)
        self.feature_list.setMaximumHeight(180)
        config_layout.addRow("参与训练字段", self.feature_list)

        feature_actions = QWidget()
        feature_actions_layout = QHBoxLayout(feature_actions)
        feature_actions_layout.setContentsMargins(0, 0, 0, 0)
        feature_actions_layout.setSpacing(8)
        self.select_all_features_button = QPushButton("全选")
        self.select_all_features_button.setProperty("variant", "secondary")
        self.select_all_features_button.clicked.connect(lambda: self.set_all_features_checked(True))
        feature_actions_layout.addWidget(self.select_all_features_button)
        self.clear_features_button = QPushButton("清空")
        self.clear_features_button.setProperty("variant", "secondary")
        self.clear_features_button.clicked.connect(lambda: self.set_all_features_checked(False))
        feature_actions_layout.addWidget(self.clear_features_button)
        feature_actions_layout.addStretch(1)
        config_layout.addRow("", feature_actions)

        self.advanced_box = QGroupBox("高级选项")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setChecked(False)
        advanced_layout = QFormLayout(self.advanced_box)
        advanced_layout.setContentsMargins(12, 18, 12, 12)
        advanced_layout.setSpacing(10)
        self.use_recommended_button = QPushButton("应用推荐配置")
        self.use_recommended_button.setProperty("variant", "secondary")
        self.use_recommended_button.clicked.connect(self.apply_recommended_config)
        self.use_recommended_button.setVisible(False)
        advanced_layout.addRow("", self.use_recommended_button)
        config_layout.addRow("", self.advanced_box)
        self.advanced_box.setVisible(False)

        train_button = QPushButton("开始训练")
        train_button.setProperty("variant", "primary")
        train_button.clicked.connect(self.start_training)
        self.train_button = train_button
        config_layout.addRow("", train_button)

        self.training_status_label = QLabel("当前状态: 待命")
        config_layout.addRow("训练状态", self.training_status_label)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setFormat("%p%")
        config_layout.addRow("训练进度", self.training_progress)
        left_layout.addWidget(self.config_box)

        self.models_box = QGroupBox("模型管理")
        models_layout = QVBoxLayout(self.models_box)
        models_layout.setContentsMargins(12, 16, 12, 12)
        self.model_search_input = QLineEdit()
        self.model_search_input.setPlaceholderText("搜索模型名称")
        self.model_search_input.textChanged.connect(self.refresh_model_list)
        models_layout.addWidget(self.model_search_input)

        self.model_filter_combo = QComboBox()
        self.model_filter_combo.addItem("全部任务", "all")
        self.model_filter_combo.addItem("分类", "classification")
        self.model_filter_combo.addItem("回归", "regression")
        self.model_filter_combo.addItem("聚类", "clustering")
        self.model_filter_combo.currentIndexChanged.connect(self.refresh_model_list)
        models_layout.addWidget(self.model_filter_combo)
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(200)
        self.model_list.itemSelectionChanged.connect(self.show_selected_model_details)
        models_layout.addWidget(self.model_list)

        predict_button = QPushButton("使用选中模型批量预测")
        predict_button.setProperty("variant", "secondary")
        predict_button.clicked.connect(self.run_prediction_for_selected_model)
        self.predict_button = predict_button
        models_layout.addWidget(predict_button)

        self.prediction_progress = QProgressBar()
        self.prediction_progress.setRange(0, 100)
        self.prediction_progress.setValue(0)
        self.prediction_progress.setFormat("%p%")
        models_layout.addWidget(self.prediction_progress)

        delete_button = QPushButton("删除选中模型")
        delete_button.setProperty("variant", "danger")
        delete_button.clicked.connect(self.delete_selected_model)
        models_layout.addWidget(delete_button)
        for button in left_panel.findChildren(QPushButton):
            button.setMinimumWidth(88)
        left_layout.addWidget(self.models_box)
        left_layout.addStretch(1)

        self.right_tabs = QTabWidget()
        splitter.addWidget(self.right_tabs)

        self.preview_tab = QWidget()
        preview_layout = QVBoxLayout(self.preview_tab)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        self.preview_empty_label = create_hint_label("导入数据集后，这里会显示前 100 条数据预览。")
        preview_layout.addWidget(self.preview_empty_label)
        self.preview_table = QTableWidget()
        configure_table(self.preview_table)
        preview_layout.addWidget(self.preview_table)
        self.right_tabs.addTab(self.preview_tab, "数据预览")

        self.results_tab = QWidget()
        results_tab_layout = QVBoxLayout(self.results_tab)
        results_tab_layout.setContentsMargins(8, 8, 8, 8)
        results_tab_layout.setSpacing(0)
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        results_tab_layout.addWidget(results_scroll)
        results_content = QWidget()
        results_layout = QVBoxLayout(results_content)
        results_layout.setContentsMargins(8, 8, 8, 8)
        results_layout.setSpacing(10)
        results_scroll.setWidget(results_content)

        self.summary_card = QFrame()
        self.summary_card.setProperty("card", "true")
        summary_card_layout = QVBoxLayout(self.summary_card)
        summary_card_layout.setContentsMargins(12, 12, 12, 12)
        summary_card_layout.setSpacing(10)
        summary_card_layout.addWidget(create_section_label("训练摘要"))

        metrics_grid = QGridLayout()
        metrics_grid.setContentsMargins(0, 0, 0, 0)
        metrics_grid.setHorizontalSpacing(10)
        metrics_grid.setVerticalSpacing(10)
        self.metric_cards: dict[str, QLabel] = {}
        metric_titles = [
            ("metric_a", "核心指标 A"),
            ("metric_b", "核心指标 B"),
            ("metric_c", "核心指标 C"),
            ("metric_d", "核心指标 D"),
        ]
        for index, (metric_key, title) in enumerate(metric_titles):
            card = QFrame()
            card.setProperty("card", "true")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(2)
            title_label = QLabel(title)
            title_label.setProperty("metricTitle", True)
            card_layout.addWidget(title_label)
            value_label = create_metric_value_label()
            self.metric_cards[metric_key] = value_label
            card_layout.addWidget(value_label)
            metrics_grid.addWidget(card, index // 2, index % 2)
        summary_card_layout.addLayout(metrics_grid)

        result_actions = QWidget()
        result_actions_layout = QGridLayout(result_actions)
        result_actions_layout.setContentsMargins(0, 0, 0, 0)
        result_actions_layout.setHorizontalSpacing(8)
        result_actions_layout.setVerticalSpacing(8)
        self.export_result_button = QPushButton("导出当前结果")
        self.export_result_button.setProperty("variant", "secondary")
        self.export_result_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.export_result_button.clicked.connect(self.export_current_results)
        result_actions_layout.addWidget(self.export_result_button, 0, 0)
        self.open_workspace_button = QPushButton("打开工作区目录")
        self.open_workspace_button.setProperty("variant", "secondary")
        self.open_workspace_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.open_workspace_button.clicked.connect(self.open_workspace_directory)
        result_actions_layout.addWidget(self.open_workspace_button, 1, 0)
        result_actions_layout.setColumnStretch(0, 1)
        summary_card_layout.addWidget(result_actions)

        next_actions = QFrame()
        next_actions.setProperty("card", "true")
        next_actions_layout = QVBoxLayout(next_actions)
        next_actions_layout.setContentsMargins(12, 10, 12, 10)
        next_actions_layout.setSpacing(8)
        next_actions_layout.addWidget(create_section_label("推荐下一步"))
        self.next_action_label = create_hint_label("训练完成后，系统会在这里推荐下一步动作。")
        next_actions_layout.addWidget(self.next_action_label)

        next_button_row = QWidget()
        next_button_layout = QGridLayout(next_button_row)
        next_button_layout.setContentsMargins(0, 0, 0, 0)
        next_button_layout.setHorizontalSpacing(8)
        next_button_layout.setVerticalSpacing(8)
        self.goto_model_button = QPushButton("导出模型文件")
        self.goto_model_button.setProperty("variant", "secondary")
        self.goto_model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.goto_model_button.clicked.connect(self.export_selected_model_file)
        next_button_layout.addWidget(self.goto_model_button, 0, 0)
        self.goto_prediction_button = QPushButton("去做批量预测")
        self.goto_prediction_button.setProperty("variant", "secondary")
        self.goto_prediction_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.goto_prediction_button.clicked.connect(self.run_prediction_for_selected_model)
        next_button_layout.addWidget(self.goto_prediction_button, 1, 0)
        next_button_layout.setColumnStretch(0, 1)
        next_actions_layout.addWidget(next_button_row)
        summary_card_layout.addWidget(next_actions)

        self.summary_empty_label = create_hint_label("训练完成后，这里会展示关键指标、摘要和推荐阅读顺序。")
        summary_card_layout.addWidget(self.summary_empty_label)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(120)
        summary_card_layout.addWidget(self.summary_text)
        results_layout.addWidget(self.summary_card)

        self.comparison_card = QFrame()
        self.comparison_card.setProperty("card", "true")
        comparison_layout = QVBoxLayout(self.comparison_card)
        comparison_layout.setContentsMargins(12, 12, 12, 12)
        comparison_layout.setSpacing(8)
        comparison_layout.addWidget(create_section_label("模型对比明细"))
        self.comparison_empty_label = create_hint_label("开始训练后，这里会显示候选模型对比结果。")
        comparison_layout.addWidget(self.comparison_empty_label)
        self.result_table = QTableWidget()
        configure_table(self.result_table)
        comparison_layout.addWidget(self.result_table)
        results_layout.addWidget(self.comparison_card, 2)

        self.detail_card = QFrame()
        self.detail_card.setProperty("card", "true")
        detail_layout = QVBoxLayout(self.detail_card)
        detail_layout.setContentsMargins(12, 12, 12, 12)
        detail_layout.setSpacing(8)
        self.detail_section_label = create_section_label("辅助结果")
        detail_layout.addWidget(self.detail_section_label)
        self.detail_empty_label = create_hint_label("聚类统计、样本预览或预测附加结果会显示在这里。")
        detail_layout.addWidget(self.detail_empty_label)
        self.cluster_table = QTableWidget()
        configure_table(self.cluster_table)
        detail_layout.addWidget(self.cluster_table)
        results_layout.addWidget(self.detail_card, 2)

        self.plot_card = QFrame()
        self.plot_card.setProperty("card", "true")
        plot_layout = QVBoxLayout(self.plot_card)
        plot_layout.setContentsMargins(12, 12, 12, 12)
        plot_layout.setSpacing(8)
        plot_layout.addWidget(create_section_label("训练图表"))
        self.plot_empty_label = create_hint_label("训练完成后会展示 PyCaret 生成的评估图表。")
        plot_layout.addWidget(self.plot_empty_label)
        self.plot_selector = QComboBox()
        self.plot_selector.currentIndexChanged.connect(self.show_selected_training_plot)
        self.plot_selector.setEnabled(False)
        plot_layout.addWidget(self.plot_selector)
        plot_height_row = QWidget()
        plot_height_row_layout = QHBoxLayout(plot_height_row)
        plot_height_row_layout.setContentsMargins(0, 0, 0, 0)
        plot_height_row_layout.setSpacing(8)
        plot_height_row_layout.addWidget(QLabel("图表高度"))
        self.plot_height_spin = QSpinBox()
        self.plot_height_spin.setRange(320, 1400)
        self.plot_height_spin.setSingleStep(40)
        self.plot_height_spin.setValue(520)
        self.plot_height_spin.valueChanged.connect(self.apply_plot_height)
        plot_height_row_layout.addWidget(self.plot_height_spin)
        plot_height_row_layout.addStretch(1)
        plot_layout.addWidget(plot_height_row)
        self.plot_image_scroll = QScrollArea()
        self.plot_image_scroll.setWidgetResizable(True)
        self.plot_image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_image_label = QLabel("暂无图表")
        self.plot_image_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.plot_image_label.setMinimumHeight(520)
        self.plot_image_label.setWordWrap(True)
        self.plot_image_scroll.setWidget(self.plot_image_label)
        plot_layout.addWidget(self.plot_image_scroll)
        results_layout.addWidget(self.plot_card, 2)
        self.right_tabs.addTab(self.results_tab, "训练结果")

        self.predict_results_tab = QWidget()
        predict_results_layout = QVBoxLayout(self.predict_results_tab)
        predict_results_layout.setContentsMargins(12, 12, 12, 12)
        predict_results_layout.setSpacing(10)

        predict_input_card = QFrame()
        predict_input_card.setProperty("card", "true")
        predict_input_layout = QVBoxLayout(predict_input_card)
        predict_input_layout.setContentsMargins(12, 12, 12, 12)
        predict_input_layout.setSpacing(8)
        predict_input_layout.addWidget(create_section_label("预测输入预览"))
        self.predict_input_empty_label = create_hint_label("选择预测文件后，这里会显示输入数据预览。")
        predict_input_layout.addWidget(self.predict_input_empty_label)
        self.predict_input_table = QTableWidget()
        configure_table(self.predict_input_table)
        predict_input_layout.addWidget(self.predict_input_table)
        predict_results_layout.addWidget(predict_input_card, 1)

        predict_output_card = QFrame()
        predict_output_card.setProperty("card", "true")
        predict_output_layout = QVBoxLayout(predict_output_card)
        predict_output_layout.setContentsMargins(12, 12, 12, 12)
        predict_output_layout.setSpacing(8)
        predict_output_layout.addWidget(create_section_label("预测结果表"))
        self.predict_output_empty_label = create_hint_label("执行预测后，这里会显示预测结果。")
        predict_output_layout.addWidget(self.predict_output_empty_label)
        self.predict_output_table = QTableWidget()
        configure_table(self.predict_output_table)
        predict_output_layout.addWidget(self.predict_output_table)
        predict_results_layout.addWidget(predict_output_card, 1)

        predict_export_card = QFrame()
        predict_export_card.setProperty("card", "true")
        predict_export_layout = QVBoxLayout(predict_export_card)
        predict_export_layout.setContentsMargins(12, 12, 12, 12)
        predict_export_layout.setSpacing(8)
        predict_export_layout.addWidget(create_section_label("导出记录"))
        self.predict_export_text = QPlainTextEdit()
        self.predict_export_text.setReadOnly(True)
        self.predict_export_text.setPlainText("暂无导出记录")
        predict_export_layout.addWidget(self.predict_export_text)
        predict_results_layout.addWidget(predict_export_card, 1)

        self.right_tabs.addTab(self.predict_results_tab, "预测结果")

        self.model_tab = QWidget()
        model_layout = QVBoxLayout(self.model_tab)
        model_layout.setContentsMargins(12, 12, 12, 12)
        detail_grid = QGridLayout()
        detail_grid.setContentsMargins(0, 0, 0, 0)
        detail_grid.setHorizontalSpacing(10)
        detail_grid.setVerticalSpacing(10)
        self.model_detail_cards: dict[str, QLabel] = {}
        detail_titles = [
            ("model_name", "模型名称"),
            ("task_type", "任务类型"),
            ("target_column", "目标列"),
            ("created_at", "创建时间"),
            ("feature_count", "字段数量"),
            ("metric_count", "指标数量"),
        ]
        for index, (detail_key, title) in enumerate(detail_titles):
            card = QFrame()
            card.setProperty("card", "true")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(2)
            title_label = QLabel(title)
            title_label.setProperty("detailTitle", True)
            card_layout.addWidget(title_label)
            value_label = create_detail_value_label()
            self.model_detail_cards[detail_key] = value_label
            card_layout.addWidget(value_label)
            detail_grid.addWidget(card, index // 3, index % 3)
        model_layout.addLayout(detail_grid)
        self.model_empty_label = create_hint_label("选择一个已训练模型后，这里会显示模型详情和指标摘要。")
        model_layout.addWidget(self.model_empty_label)
        self.model_details_text = QPlainTextEdit()
        self.model_details_text.setReadOnly(True)
        model_layout.addWidget(self.model_details_text)
        self.right_tabs.addTab(self.model_tab, "模型详情")

        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout(self.logs_tab)
        logs_layout.setContentsMargins(12, 12, 12, 12)
        logs_layout.addWidget(create_section_label("最近操作历史"))
        self.history_text = QPlainTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(140)
        logs_layout.addWidget(self.history_text)
        logs_layout.addWidget(create_section_label("运行日志"))
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)
        self.right_tabs.addTab(self.logs_tab, "运行日志")

        splitter.setSizes([420, 780])
        self.disable_config_wheel_adjustment()
        self._set_config_row_visible("recommendation", False)
        self.toggle_task_inputs(self.task_combo.currentIndex())
        self.update_workflow_status("idle")
        self.update_next_actions("idle")
        self.update_profile_stat_cards()
        self.update_training_plots([])
        self.apply_plot_height(self.plot_height_spin.value())
        self.update_model_detail_cards()
        self.update_history_panel()
        self.set_app_mode("train")
        self.update_empty_states()

    def current_app_mode(self) -> str:
        return self.app_mode

    def set_app_mode(self, mode: str) -> None:
        next_mode = mode if mode in {"train", "predict"} else "train"
        self.app_mode = next_mode

        if hasattr(self, "mode_train_button") and hasattr(self, "mode_predict_button"):
            self.mode_train_button.blockSignals(True)
            self.mode_predict_button.blockSignals(True)
            self.mode_train_button.setChecked(next_mode == "train")
            self.mode_predict_button.setChecked(next_mode == "predict")
            self.mode_train_button.blockSignals(False)
            self.mode_predict_button.blockSignals(False)

        self.apply_mode_ui()

    def apply_mode_ui(self) -> None:
        is_train = self.current_app_mode() == "train"

        self.config_box.setVisible(is_train)
        self.models_box.setVisible(not is_train)
        self.load_button.setVisible(is_train)
        self.profile_box.setVisible(is_train)

        if is_train:
            self.summary_card.setVisible(True)
            self.plot_card.setVisible(True)
            self.comparison_card.setVisible(self.current_task_type() != "clustering")
            self.detail_card.setVisible(True)
        else:
            # Prediction mode is fully decoupled from training result panels.
            self.summary_card.setVisible(False)
            self.plot_card.setVisible(False)
            self.comparison_card.setVisible(False)
            self.detail_card.setVisible(False)

        self.right_tabs.setTabVisible(self.right_tabs.indexOf(self.preview_tab), is_train)
        self.right_tabs.setTabVisible(self.right_tabs.indexOf(self.results_tab), is_train)
        self.right_tabs.setTabVisible(self.right_tabs.indexOf(self.model_tab), is_train)
        self.right_tabs.setTabVisible(self.right_tabs.indexOf(self.predict_results_tab), not is_train)
        self.right_tabs.setCurrentWidget(self.results_tab if is_train else self.predict_results_tab)

    def append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def handle_training_log_progress(self, message: str) -> None:
        self.append_log(message)
        next_value = min(90, self.training_progress.value() + 10)
        if next_value > self.training_progress.value():
            self.training_progress.setValue(next_value)

    def update_prediction_progress(self, value: int, message: str) -> None:
        self.prediction_progress.setValue(max(0, min(100, int(value))))
        if message:
            self.append_log(message)

    def add_history_record(self, title: str, detail: str) -> None:
        entry = f"{title}: {detail}"
        self.history_records.insert(0, entry)
        self.history_records = self.history_records[:12]
        self.update_history_panel()

    def update_history_panel(self) -> None:
        if not self.history_records:
            self.history_text.setPlainText(
                "\n".join(
                    [
                        "最近操作会显示在这里。",
                        "包括：导入数据、训练模型、批量预测、导出结果、删除模型。",
                    ]
                )
            )
            return
        self.history_text.setPlainText("\n".join(self.history_records))

    def append_predict_export_record(self, record: str) -> None:
        current_text = self.predict_export_text.toPlainText().strip()
        if not current_text or current_text == "暂无导出记录":
            self.predict_export_text.setPlainText(record)
            return
        self.predict_export_text.setPlainText(f"{record}\n{current_text}")

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Wheel and obj in self._no_wheel_widgets:
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_plot_path is not None:
            self.show_selected_training_plot()

    def apply_plot_height(self, height: int) -> None:
        self.plot_image_label.setMinimumHeight(max(200, int(height)))
        self.plot_image_scroll.setMinimumHeight(max(220, int(height) + 24))
        if self._current_plot_path is not None:
            self.show_selected_training_plot()

    def disable_config_wheel_adjustment(self) -> None:
        widgets: list[QWidget] = []
        widgets.extend(self.findChildren(QComboBox))
        widgets.extend(self.findChildren(QSpinBox))
        widgets.extend(self.findChildren(QDoubleSpinBox))
        for widget in widgets:
            widget.installEventFilter(self)
            self._no_wheel_widgets.add(widget)

    def update_workflow_status(self, state: str = "idle") -> None:
        steps = {
            "idle": [
                "1. 导入业务数据文件",
                "2. 查看系统给出的任务建议",
                "3. 选择目标列和参与训练字段",
                "4. 点击开始训练生成模型",
                "5. 在模型管理中执行批量预测",
            ],
            "loaded": [
                "1. 数据已导入",
                "2. 请确认任务类型、目标列和训练字段",
                "3. 可按需展开高级选项",
                "4. 点击开始训练生成模型",
                "5. 完成后可用模型执行批量预测",
            ],
            "trained": [
                "1. 数据与训练已完成",
                "2. 查看训练摘要和模型对比结果",
                "3. 在模型管理中选择目标模型",
                "4. 导入新数据执行批量预测",
                "5. 输出结果会保存到工作区目录",
            ],
            "predicted": [
                "1. 模型预测已完成",
                "2. 在训练结果页查看输出预览",
                "3. 到输出文件路径获取完整结果",
                "4. 如需复训，可调整字段和参数",
                "5. 如需切换模型，请在左侧模型管理中选择",
            ],
        }
        if hasattr(self, "workflow_text") and self.workflow_text is not None:
            self.workflow_text.setPlainText("\n".join(steps.get(state, steps["idle"])))

    def update_next_actions(self, state: str = "idle", model_name: str | None = None) -> None:
        texts = {
            "idle": "先导入数据，系统会根据数据结构推荐适合的任务类型。",
            "loaded": "建议先确认目标列和字段，再开始训练。",
            "trained": f"模型 {model_name or ''} 已训练完成。建议先导出模型文件，再导入新数据做批量预测。",
            "predicted": "预测结果已生成。建议先导出结果，或切换模型重新预测。",
        }
        self.next_action_label.setText(texts.get(state, texts["idle"]))

    def update_empty_states(self) -> None:
        has_preview = self.preview_table.rowCount() > 0
        has_summary = bool(self.summary_text.toPlainText().strip())
        has_comparison = self.result_table.rowCount() > 0
        has_detail = self.cluster_table.rowCount() > 0
        has_model = bool(self.model_details_text.toPlainText().strip())
        has_plots = self.plot_selector.count() > 0
        has_predict_input = self.predict_input_table.rowCount() > 0
        has_predict_output = self.predict_output_table.rowCount() > 0

        self.preview_empty_label.setVisible(not has_preview)
        self.summary_empty_label.setVisible(not has_summary)
        self.comparison_empty_label.setVisible(not has_comparison)
        self.detail_empty_label.setVisible(not has_detail)
        self.model_empty_label.setVisible(not has_model)
        self.plot_empty_label.setVisible(not has_plots)
        self.predict_input_empty_label.setVisible(not has_predict_input)
        self.predict_output_empty_label.setVisible(not has_predict_output)
        has_export = self.last_export_frame is not None and not self.last_export_frame.empty
        self.export_result_button.setEnabled(has_export)
        has_selected_model = len(self.model_list.selectedItems()) > 0
        self.goto_model_button.setEnabled(has_selected_model)
        self.goto_prediction_button.setEnabled(has_selected_model)

    def load_dataset_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据集",
            "",
            "Data Files (*.csv *.xlsx *.xls)",
        )
        if not file_path:
            return
        try:
            self.current_data = load_dataset(file_path)
            self.current_data_path = Path(file_path)
            self.dataset_label.setText(f"当前数据集: {self.current_data_path.name}")
            dataframe_to_table(self.preview_table, self.current_data.head(100))
            self.right_tabs.setCurrentIndex(0)
            self.refresh_target_options()
            self.render_profile()
            self.refresh_feature_options()
            self.update_workflow_status("loaded")
            self.update_next_actions("loaded")
            self.update_empty_states()
            self.add_history_record("导入数据", self.current_data_path.name)
            self.append_log(f"已加载数据集: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def refresh_target_options(self) -> None:
        self.target_combo.clear()
        if self.current_data is not None:
            self.target_combo.addItems(self.current_data.columns.tolist())

    def render_profile(self) -> None:
        if self.current_data is None:
            self.profile_text.setPlainText("")
            self.update_profile_stat_cards()
            return
        profile = profile_dataset(self.current_data)
        self.current_profile = profile
        lines = [
            f"总行数: {profile.rows}",
            f"总列数: {profile.columns}",
            f"数值列: {', '.join(profile.numeric_columns) or '无'}",
            f"类别/文本列: {', '.join(profile.categorical_columns) or '无'}",
            f"重复行: {profile.duplicate_rows}",
            f"建议优先检查的目标列: {', '.join(profile.candidate_targets[:8]) or '无'}",
            f"推荐任务: {', '.join(profile.recommended_tasks) or '无'}",
        ]
        if profile.warnings:
            lines.append("")
            lines.append("风险提示:")
            lines.extend(profile.warnings)
        self.profile_text.setPlainText("\n".join(lines))
        if profile.recommended_tasks:
            self.recommendation_label.setText(
                "系统建议优先尝试: " + " / ".join(self._task_label(task) for task in profile.recommended_tasks)
            )
        else:
            self.recommendation_label.setText("系统暂时无法给出明确任务建议，请结合业务目标手动选择。")
        self.update_profile_stat_cards(profile)

    def update_profile_stat_cards(self, profile=None) -> None:
        values = {
            "rows": "--",
            "columns": "--",
            "numeric": "--",
            "categorical": "--",
            "duplicates": "--",
            "targets": "--",
        }
        if profile is not None:
            values = {
                "rows": str(profile.rows),
                "columns": str(profile.columns),
                "numeric": str(len(profile.numeric_columns)),
                "categorical": str(len(profile.categorical_columns)),
                "duplicates": str(profile.duplicate_rows),
                "targets": str(len(profile.candidate_targets)),
            }
        for key, label in self.profile_stat_cards.items():
            label.setText(values[key])

    def _set_config_row_visible(self, row_key: str, visible: bool) -> None:
        row = self.config_row_index.get(row_key)
        if row is None:
            return
        try:
            self.config_layout.setRowVisible(row, visible)
            return
        except AttributeError:
            pass

        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
            item = self.config_layout.itemAt(row, role)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setVisible(visible)

    def toggle_task_inputs(self, _: int | bool | None = None) -> None:
        task_type = self.current_task_type()
        clustering = task_type == "clustering"
        is_supervised = task_type in {"classification", "regression"}
        manual_mode = self.training_mode_combo.currentData() == "manual_single"
        preprocess_enabled = (not clustering) and self.preprocess_text_check.isChecked()

        self._set_config_row_visible("target", not clustering)
        self._set_config_row_visible("train_size", not clustering)
        self._set_config_row_visible("training_mode", is_supervised)
        self._set_config_row_visible("manual_model", is_supervised and manual_mode)
        self._set_config_row_visible("cluster_count", clustering)
        self._set_config_row_visible("preprocess_text", not clustering)
        self._set_config_row_visible("remove_stopwords", preprocess_enabled)
        self._set_config_row_visible("min_word_length", preprocess_enabled)

        self.refresh_manual_model_options(task_type)
        self.cluster_spin.setEnabled(clustering)
        self.refresh_feature_options()

    def refresh_feature_options(self) -> None:
        self.feature_list.clear()
        if self.current_data is None:
            return

        task_type = self.current_task_type()
        target_column = self.target_combo.currentText()
        for column in self.current_data.columns:
            if task_type != "clustering" and column == target_column:
                continue
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, column)
            self.feature_list.addItem(item)

    def should_recommend_feature(self, column: str, task_type: str) -> bool:
        if self.current_data is None:
            return True
        series = self.current_data[column]
        row_count = len(series)
        unique_count = series.nunique(dropna=True)
        missing_ratio = float(series.isna().mean()) if row_count else 0.0

        if missing_ratio > 0.4:
            return False
        if row_count > 20 and unique_count == row_count:
            return False
        if task_type == "clustering" and not pd.api.types.is_numeric_dtype(series):
            return False
        return True

    def feature_quality_tags(self, column: str, task_type: str) -> list[str]:
        if self.current_data is None:
            return []
        series = self.current_data[column]
        row_count = len(series)
        unique_count = series.nunique(dropna=True)
        missing_ratio = float(series.isna().mean()) if row_count else 0.0

        tags: list[str] = []
        if row_count > 20 and unique_count == row_count:
            tags.append("疑似ID")
        if missing_ratio > 0.4:
            tags.append("高缺失")
        if task_type == "clustering" and not pd.api.types.is_numeric_dtype(series):
            tags.append("聚类不推荐")
        if not tags:
            tags.append("推荐")
        return tags

    def available_manual_model_options(self, task_type: str) -> list[tuple[str, str]]:
        if task_type == "classification":
            return [
                ("逻辑回归 (lr)", "lr"),
                ("随机森林 (rf)", "rf"),
                ("极端随机树 (et)", "et"),
                ("决策树 (dt)", "dt"),
                ("KNN (knn)", "knn"),
                ("朴素贝叶斯 (nb)", "nb"),
                ("支持向量机 (svm)", "svm"),
                ("梯度提升 (gbc)", "gbc"),
                ("AdaBoost (ada)", "ada"),
                ("XGBoost (xgboost)", "xgboost"),
                ("LightGBM (lightgbm)", "lightgbm"),
            ]
        if task_type == "regression":
            return [
                ("线性回归 (lr)", "lr"),
                ("岭回归 (ridge)", "ridge"),
                ("Lasso (lasso)", "lasso"),
                ("ElasticNet (en)", "en"),
                ("随机森林 (rf)", "rf"),
                ("极端随机树 (et)", "et"),
                ("决策树 (dt)", "dt"),
                ("KNN (knn)", "knn"),
                ("梯度提升 (gbr)", "gbr"),
                ("AdaBoost (ada)", "ada"),
                ("XGBoost (xgboost)", "xgboost"),
                ("LightGBM (lightgbm)", "lightgbm"),
            ]
        return []

    def refresh_manual_model_options(self, task_type: str) -> None:
        current_value = self.manual_model_combo.currentData()
        self.manual_model_combo.blockSignals(True)
        self.manual_model_combo.clear()
        for label, model_id in self.available_manual_model_options(task_type):
            self.manual_model_combo.addItem(label, model_id)
        if current_value is not None:
            index = self.manual_model_combo.findData(current_value)
            if index >= 0:
                self.manual_model_combo.setCurrentIndex(index)
        self.manual_model_combo.blockSignals(False)

    def current_task_type(self) -> str:
        return self.task_combo.currentData()

    def _task_label(self, task_type: str) -> str:
        return {
            "classification": "分类",
            "regression": "回归",
            "clustering": "聚类",
        }.get(task_type, task_type)

    def selected_feature_columns(self) -> list[str]:
        columns: list[str] = []
        for index in range(self.feature_list.count()):
            item = self.feature_list.item(index)
            if item.checkState() == Qt.Checked:
                columns.append(item.data(Qt.UserRole) or item.text())
        return columns

    def set_all_features_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for index in range(self.feature_list.count()):
            self.feature_list.item(index).setCheckState(state)

    def update_metric_cards(self, task_type: str | None = None, metrics: dict[str, object] | None = None) -> None:
        labels = {
            "metric_a": ("核心指标 A", "--"),
            "metric_b": ("核心指标 B", "--"),
            "metric_c": ("核心指标 C", "--"),
            "metric_d": ("核心指标 D", "--"),
        }

        if task_type == "classification" and metrics:
            keys = ["Accuracy", "AUC", "Recall", "Prec."]
            titles = ["准确率", "AUC", "召回率", "精确率"]
            for slot, key, title in zip(labels.keys(), keys, titles):
                labels[slot] = (title, str(metrics.get(key, "--")))
        elif task_type == "regression" and metrics:
            keys = ["R2", "RMSE", "MAE", "MAPE"]
            titles = ["R2", "RMSE", "MAE", "MAPE"]
            for slot, key, title in zip(labels.keys(), keys, titles):
                labels[slot] = (title, str(metrics.get(key, "--")))
        elif task_type == "clustering" and metrics:
            mapping = [
                ("metric_a", "聚类数量", metrics.get("cluster_count", "--")),
                ("metric_b", "样本数", metrics.get("sample_count", "--")),
                ("metric_c", "轮廓系数", metrics.get("silhouette", "--")),
                ("metric_d", "状态", "已完成"),
            ]
            for slot, title, value in mapping:
                labels[slot] = (title, str(value))

        for index, slot in enumerate(["metric_a", "metric_b", "metric_c", "metric_d"]):
            card = self.metric_cards[slot].parentWidget()
            title_label = card.layout().itemAt(0).widget()
            title_label.setText(labels[slot][0])
            self.metric_cards[slot].setText(labels[slot][1])

    def update_training_plots(self, plot_paths: list[str] | None) -> None:
        self.plot_selector.blockSignals(True)
        self.plot_selector.clear()
        self._current_plot_path = None

        valid_paths: list[Path] = []
        for path in plot_paths or []:
            candidate = Path(path)
            if candidate.exists():
                valid_paths.append(candidate)

        for path in valid_paths:
            self.plot_selector.addItem(path.stem, str(path))

        self.plot_selector.setEnabled(bool(valid_paths))
        self.plot_selector.blockSignals(False)
        if valid_paths:
            self.plot_selector.setCurrentIndex(0)
            self.show_selected_training_plot()
        else:
            self.plot_image_label.setPixmap(QPixmap())
            self.plot_image_label.setText("暂无图表")
        self.update_empty_states()

    def show_selected_training_plot(self) -> None:
        if self.plot_selector.count() == 0:
            self.plot_image_label.setPixmap(QPixmap())
            self.plot_image_label.setText("暂无图表")
            return

        path_text = self.plot_selector.currentData()
        if not path_text:
            self.plot_image_label.setPixmap(QPixmap())
            self.plot_image_label.setText("暂无图表")
            return
        plot_path = Path(path_text)
        if not plot_path.exists():
            self.plot_image_label.setPixmap(QPixmap())
            self.plot_image_label.setText(f"图表文件不存在:\n{plot_path}")
            return

        pixmap = QPixmap(str(plot_path))
        if pixmap.isNull():
            self.plot_image_label.setPixmap(QPixmap())
            self.plot_image_label.setText(f"无法加载图表:\n{plot_path.name}")
            return

        target_width = max(360, self.plot_image_scroll.viewport().width() - 24)
        scaled = pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
        self.plot_image_label.setText("")
        self.plot_image_label.setPixmap(scaled)
        self._current_plot_path = plot_path

    def apply_recommended_config(self) -> None:
        if self.current_data is None:
            return

        profile = self.current_profile or profile_dataset(self.current_data)
        if profile.recommended_tasks:
            recommended_task = profile.recommended_tasks[0]
            index = self.task_combo.findData(recommended_task)
            if index >= 0:
                self.task_combo.setCurrentIndex(index)

        if profile.candidate_targets:
            preferred_target = profile.candidate_targets[0]
            target_index = self.target_combo.findText(preferred_target)
            if target_index >= 0:
                self.target_combo.setCurrentIndex(target_index)

        self.refresh_feature_options()

    def build_training_config(self) -> TrainingConfig:
        if self.current_data is None:
            raise ValueError("请先导入数据集。")

        task_type = self.current_task_type()
        target_column = self.target_combo.currentText() if task_type != "clustering" else None
        feature_columns = self.selected_feature_columns() or None
        preprocess_text = self.preprocess_text_check.isChecked() and task_type != "clustering"

        text_columns: list[str] = []
        if preprocess_text:
            text_columns = self.current_data.select_dtypes(include="object").columns.tolist()
            text_columns = [column for column in text_columns if column != target_column]
            text_columns = [column for column in text_columns if column in (feature_columns or [])]

        return TrainingConfig(
            task_type=task_type,
            target_column=target_column,
            train_size=float(self.train_size_spin.value()),
            feature_columns=feature_columns,
            training_mode=str(self.training_mode_combo.currentData() or "auto_compare"),
            manual_model_id=str(self.manual_model_combo.currentData() or "") if self.training_mode_combo.currentData() == "manual_single" else None,
            cluster_count=int(self.cluster_spin.value()),
            preprocess_text=preprocess_text,
            text_columns=text_columns,
            remove_stopwords=self.remove_stopwords_check.isChecked() if preprocess_text else False,
            min_word_length=int(self.min_word_length_spin.value()) if preprocess_text else 1,
        )

    def validate_training_config(self, config: TrainingConfig) -> list[str]:
        issues: list[str] = []
        if self.current_data is None:
            issues.append("请先导入数据集。")
            return issues

        selected_features = config.feature_columns or []
        if not selected_features:
            issues.append("至少需要选择一个参与训练的字段。")

        if config.task_type != "clustering":
            if not config.target_column:
                issues.append("分类和回归任务必须选择目标列。")
            elif config.target_column not in self.current_data.columns:
                issues.append("所选目标列不存在，请重新选择。")
            if config.training_mode == "manual_single" and not config.manual_model_id:
                issues.append("手动指定模型模式下，必须选择一个模型。")

        return issues

    def start_training(self) -> None:
        if self.current_data is None:
            QMessageBox.warning(self, "缺少数据", "请先导入数据集。")
            return
        try:
            config = self.build_training_config()
        except Exception as exc:
            QMessageBox.warning(self, "配置错误", str(exc))
            return

        issues = self.validate_training_config(config)
        if issues:
            confirm = QMessageBox.question(
                self,
                "训练前检查",
                "\n".join(["检测到以下配置提示：", *issues, "", "仍然继续训练吗？"]),
            )
            if confirm != QMessageBox.Yes:
                return

        self.summary_text.clear()
        self.result_table.clear()
        self.cluster_table.clear()
        self.update_training_plots([])
        self.detail_section_label.setText("辅助结果")
        self.last_export_frame = None
        self.update_metric_cards()
        self.update_empty_states()
        self.append_log("开始训练模型...")
        self.training_status_label.setText("当前状态: 训练中")
        self.training_progress.setValue(5)
        self.train_button.setEnabled(False)
        self.predict_button.setEnabled(False)

        self.training_worker = TrainingWorker(self.training_service, self.current_data, config)
        self.training_worker.log_emitted.connect(self.handle_training_log_progress)
        self.training_worker.completed.connect(self.handle_training_completed)
        self.training_worker.failed.connect(self.handle_training_failed)
        self.training_worker.start()

    def handle_training_completed(self, result: TrainingResult) -> None:
        is_clustering = result.metadata.task_type == "clustering"
        self.comparison_card.setVisible(not is_clustering)
        self.update_metric_cards(result.metadata.task_type, result.metadata.metrics)
        self.summary_text.setPlainText("\n".join(result.summary_lines))
        self.update_training_plots(result.plot_paths)
        dataframe_to_table(self.result_table, result.comparison)
        cluster_data = result.cluster_stats if result.cluster_stats is not None else result.predictions_preview
        if is_clustering:
            self.detail_section_label.setText("聚类统计与样本预览")
        elif result.predictions_preview is not None and not result.predictions_preview.empty:
            self.detail_section_label.setText("测试集预测预览")
        elif result.comparison is not None and not result.comparison.empty:
            self.detail_section_label.setText("当前任务附加结果")
        else:
            self.detail_section_label.setText("结果预览")
        dataframe_to_table(self.cluster_table, cluster_data)
        self.last_export_frame = result.comparison if result.comparison is not None and not result.comparison.empty else cluster_data
        self.append_log(f"训练完成: {result.metadata.model_name}")
        self.training_status_label.setText("当前状态: 训练完成")
        self.training_progress.setValue(100)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.refresh_model_list()
        self.select_model(result.metadata.model_name)
        self.right_tabs.setCurrentIndex(1)
        self.update_workflow_status("trained")
        self.update_next_actions("trained", result.metadata.model_name)
        self.update_empty_states()
        self.add_history_record("训练模型", result.metadata.model_name)

    def handle_training_failed(self, error_message: str) -> None:
        self.append_log(f"训练失败: {error_message}")
        self.training_status_label.setText("当前状态: 训练失败")
        self.training_progress.setValue(0)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        QMessageBox.critical(self, "训练失败", error_message)

    def refresh_model_list(self) -> None:
        self.model_list.clear()
        self.all_model_metadata = self.registry.list_models()
        self.model_metadata_lookup = {metadata.model_name: metadata for metadata in self.all_model_metadata}
        keyword = self.model_search_input.text().strip().lower() if hasattr(self, "model_search_input") else ""
        task_filter = self.model_filter_combo.currentData() if hasattr(self, "model_filter_combo") else "all"

        for metadata in self.all_model_metadata:
            if task_filter != "all" and metadata.task_type != task_filter:
                continue
            if keyword and keyword not in metadata.model_name.lower():
                continue
            summary = self.build_model_list_summary(metadata)
            item = QListWidgetItem(f"{metadata.model_name} [{self._task_label(metadata.task_type)}]")
            if summary:
                item.setToolTip(summary)
                item.setText(f"{metadata.model_name} [{self._task_label(metadata.task_type)}]  {summary}")
            item.setData(256, metadata.model_name)
            self.model_list.addItem(item)
        self.update_empty_states()

    def build_model_list_summary(self, metadata) -> str:
        if not metadata.metrics:
            return ""

        if metadata.task_type == "classification":
            preferred_keys = ["Accuracy", "AUC", "Recall"]
        elif metadata.task_type == "regression":
            preferred_keys = ["R2", "RMSE", "MAE"]
        elif metadata.task_type == "clustering":
            preferred_keys = ["silhouette", "cluster_count"]
        else:
            preferred_keys = list(metadata.metrics.keys())[:3]

        parts = []
        for key in preferred_keys:
            if key in metadata.metrics:
                parts.append(f"{key}: {metadata.metrics[key]}")
        return " | ".join(parts[:2])

    def select_model(self, model_name: str) -> None:
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item.data(256) == model_name:
                self.model_list.setCurrentItem(item)
                return

    def export_selected_model_file(self) -> None:
        items = self.model_list.selectedItems()
        if not items:
            QMessageBox.information(self, "未选择模型", "请先在左侧模型列表中选择一个模型。")
            return

        model_name = items[0].data(256)
        metadata = self.registry.load_metadata(model_name)
        source_path = Path(metadata.model_path)
        if not source_path.exists():
            QMessageBox.warning(self, "导出失败", f"模型文件不存在:\n{source_path}")
            return

        default_path = APP_DIR / source_path.name
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出模型文件",
            str(default_path),
            "Model Files (*.pkl)",
        )
        if not file_path:
            return

        export_path = Path(file_path)
        shutil.copy2(source_path, export_path)
        self.append_log(f"已导出模型文件: {export_path}")
        self.add_history_record("导出模型", export_path.name)
        QMessageBox.information(self, "导出成功", f"模型文件已导出到:\n{export_path}")

    def show_selected_model_details(self) -> None:
        items = self.model_list.selectedItems()
        if not items:
            self.model_details_text.clear()
            self.update_metric_cards()
            self.update_model_detail_cards()
            self.update_empty_states()
            return
        model_name = items[0].data(256)
        metadata = self.registry.load_metadata(model_name)
        self.update_metric_cards(metadata.task_type, metadata.metrics)
        self.update_model_detail_cards(metadata)
        lines = [
            f"模型名称: {metadata.model_name}",
            f"任务类型: {metadata.task_type}",
            f"创建时间: {metadata.created_at}",
            f"目标列: {metadata.target_column or '无'}",
            f"特征列: {', '.join(metadata.feature_columns) or '无'}",
            f"文本列: {', '.join(metadata.text_columns) or '无'}",
            "指标摘要:",
        ]
        for key, value in metadata.metrics.items():
            lines.append(f"- {key}: {value}")
        if metadata.notes:
            lines.append("")
            lines.append(metadata.notes)
        self.model_details_text.setPlainText("\n".join(lines))
        self.right_tabs.setCurrentIndex(2)
        self.update_empty_states()

    def update_model_detail_cards(self, metadata=None) -> None:
        values = {
            "model_name": "--",
            "task_type": "--",
            "target_column": "--",
            "created_at": "--",
            "feature_count": "--",
            "metric_count": "--",
        }
        if metadata is not None:
            values = {
                "model_name": metadata.model_name,
                "task_type": self._task_label(metadata.task_type),
                "target_column": metadata.target_column or "无",
                "created_at": metadata.created_at,
                "feature_count": str(len(metadata.feature_columns)),
                "metric_count": str(len(metadata.metrics)),
            }
        for key, label in self.model_detail_cards.items():
            label.setText(values[key])

    def run_prediction_for_selected_model(self) -> None:
        if self.current_app_mode() != "predict":
            self.set_app_mode("predict")

        items = self.model_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "未选择模型", "请先从模型列表中选择一个模型。")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待预测数据",
            "",
            "Data Files (*.csv *.xlsx *.xls)",
        )
        if not file_path:
            return

        model_name = items[0].data(256)
        try:
            prediction_data = load_dataset(file_path)
            dataframe_to_table(self.predict_input_table, prediction_data.head(100))
            self.pending_prediction_model_name = model_name
            self.prediction_progress.setValue(5)
            self.train_button.setEnabled(False)
            self.predict_button.setEnabled(False)

            self.prediction_worker = PredictionWorker(self.prediction_service, model_name, prediction_data)
            self.prediction_worker.progress_changed.connect(self.update_prediction_progress)
            self.prediction_worker.completed.connect(self.handle_prediction_completed)
            self.prediction_worker.failed.connect(self.handle_prediction_failed)
            self.prediction_worker.start()
        except Exception as exc:
            QMessageBox.critical(self, "预测失败", str(exc))

    def handle_prediction_completed(self, result) -> None:
        model_name = self.pending_prediction_model_name or result.metadata.model_name
        self.update_metric_cards(result.metadata.task_type, result.metadata.metrics)
        dataframe_to_table(self.predict_output_table, result.predictions.head(100))
        self.last_export_frame = result.predictions
        self.append_predict_export_record(
            f"[预测输出 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {Path(result.output_path).name}"
        )
        self.prediction_progress.setValue(100)
        self.append_log(f"批量预测完成: {result.output_path}")
        self.right_tabs.setCurrentWidget(self.predict_results_tab)
        self.update_workflow_status("predicted")
        self.update_next_actions("predicted", model_name)
        self.update_empty_states()
        self.add_history_record("批量预测", f"{model_name} -> {Path(result.output_path).name}")
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.pending_prediction_model_name = None

    def handle_prediction_failed(self, error_message: str) -> None:
        self.prediction_progress.setValue(0)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.pending_prediction_model_name = None
        self.append_log(f"批量预测失败: {error_message}")
        QMessageBox.critical(self, "预测失败", error_message)

    def delete_selected_model(self) -> None:
        items = self.model_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "未选择模型", "请先从模型列表中选择一个模型。")
            return

        model_name = items[0].data(256)
        confirm = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除模型 {model_name} 吗？",
        )
        if confirm != QMessageBox.Yes:
            return

        metadata_path = self.registry.metadata_path(model_name)
        model_path = self.registry.model_path(model_name).with_suffix(".pkl")
        cluster_path = self.registry.models_dir / f"{model_name}_clusters.csv"

        for path in [metadata_path, model_path, cluster_path]:
            if path.exists():
                path.unlink()

        self.append_log(f"已删除模型: {model_name}")
        self.add_history_record("删除模型", model_name)
        self.refresh_model_list()
        self.model_details_text.clear()
        self.update_empty_states()

    def export_current_results(self) -> None:
        if self.last_export_frame is None or self.last_export_frame.empty:
            QMessageBox.information(self, "暂无结果", "当前没有可导出的训练或预测结果。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前结果",
            str(APP_DIR / "export_results.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        export_path = Path(file_path)
        self.last_export_frame.to_csv(export_path, index=False, encoding="utf-8-sig")
        self.append_log(f"已导出结果: {export_path}")
        self.add_history_record("导出结果", export_path.name)
        if self.current_app_mode() == "predict":
            self.append_predict_export_record(
                f"[手动导出 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {export_path.name}"
            )
        QMessageBox.information(self, "导出成功", f"结果已导出到:\n{export_path}")

    def open_workspace_directory(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(APP_DIR)))


def main() -> int:
    app = QApplication(sys.argv)
    window = DesktopMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
