from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mlquick_core.io_utils import load_dataset
from mlquick_core.models import TrainingConfig
from mlquick_core.prediction import PredictionService
from mlquick_core.profiling import profile_dataset
from mlquick_core.registry import ModelRegistry
from mlquick_core.training import TrainingService


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="mlquick_test_"))
        cls.mpl_dir = cls.temp_dir / "mplconfig"
        cls.mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cls.mpl_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self) -> None:
        self.workspace = self.temp_dir / self.id().replace(".", "_")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(self.workspace)
        self.trainer = TrainingService(self.registry)
        self.predictor = PredictionService(self.registry)

    def test_profile_dataset(self) -> None:
        data = load_dataset(ROOT / "data" / "samples" / "classification_sample.csv")
        profile = profile_dataset(data)
        self.assertEqual(profile.rows, len(data))
        self.assertIn("classification", profile.recommended_tasks)

    def test_classification_train_predict(self) -> None:
        data = load_dataset(ROOT / "data" / "samples" / "classification_sample.csv")
        target = "target"
        config = TrainingConfig(
            task_type="classification",
            target_column=target,
            train_size=0.7,
            feature_columns=[column for column in data.columns if column != target],
        )
        result = self.trainer.train(data, config)
        prediction_result = self.predictor.predict(result.metadata.model_name, data.drop(columns=[target]).head(10))

        self.assertTrue(result.metadata.metrics)
        self.assertEqual(prediction_result.predictions.shape[0], 10)
        self.assertIn("prediction_label", prediction_result.predictions.columns)

    def test_regression_train_predict(self) -> None:
        data = load_dataset(ROOT / "data" / "samples" / "regression_sample.csv")
        target = "price_in_thousands"
        config = TrainingConfig(
            task_type="regression",
            target_column=target,
            train_size=0.7,
            feature_columns=[column for column in data.columns if column != target],
        )
        result = self.trainer.train(data, config)
        prediction_result = self.predictor.predict(result.metadata.model_name, data.drop(columns=[target]).head(10))

        self.assertTrue(result.metadata.metrics)
        self.assertEqual(prediction_result.predictions.shape[0], 10)
        self.assertIn("prediction_label", prediction_result.predictions.columns)

    def test_clustering_train_predict(self) -> None:
        data = load_dataset(ROOT / "data" / "samples" / "clustering_sample.csv")
        features = data.select_dtypes(include="number").columns.tolist()
        config = TrainingConfig(
            task_type="clustering",
            cluster_count=3,
            feature_columns=features,
        )
        result = self.trainer.train(data, config)
        prediction_result = self.predictor.predict(result.metadata.model_name, data[features].head(10))

        self.assertIn("silhouette", result.metadata.metrics)
        self.assertEqual(prediction_result.predictions.shape[0], 10)
        self.assertIn("Cluster", prediction_result.predictions.columns)

    def test_desktop_initialization(self) -> None:
        from PySide6.QtWidgets import QApplication
        import mlquick_desktop

        app = QApplication.instance() or QApplication([])
        previous_app_dir = mlquick_desktop.APP_DIR
        mlquick_desktop.APP_DIR = self.workspace / "desktop_workspace"
        try:
            window = mlquick_desktop.DesktopMainWindow()
            self.assertEqual(window.windowTitle(), "MLquick 桌面版")
            self.assertEqual(window.right_tabs.count(), 5)
            self.assertEqual(window.model_filter_combo.count(), 4)
            window.close()
        finally:
            mlquick_desktop.APP_DIR = previous_app_dir
            app.quit()


if __name__ == "__main__":
    unittest.main()
