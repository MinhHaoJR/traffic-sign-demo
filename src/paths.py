"""
Common paths for the traffic sign detection project.
Uses pathlib to support running on different machines and environments.
"""
from pathlib import Path

# Thư mục mã nguồn và thư mục gốc của repository
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

# Thư mục cấu hình
CONFIGS_DIR = ROOT_DIR / "configs"
CLASSES_PATH = CONFIGS_DIR / "classes.json"
DATA_YAML_PATH = CONFIGS_DIR / "data.yaml"
DATA_EXAMPLE_YAML_PATH = CONFIGS_DIR / "data.example.yaml"

# Thư mục dữ liệu
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DATA_DIR / "images"
LABELS_DIR = PROCESSED_DATA_DIR / "labels"

# Thư mục mô hình
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best.pt"

# Thư mục kết quả đầu ra
OUTPUT_DIR = ROOT_DIR / "outputs"
PREPROCESSING_OUTPUT_DIR = OUTPUT_DIR / "preprocessing"
LABEL_VERIFICATION_DIR = OUTPUT_DIR / "label_verification"
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"
PREDICTIONS_OUTPUT_DIR = OUTPUT_DIR / "predictions"

# Thư mục tài liệu và báo cáo
DOCS_DIR = ROOT_DIR / "docs"
DATA_SUMMARY_PATH = DOCS_DIR / "data_summary.csv"
SPLIT_MANIFEST_PATH = DOCS_DIR / "split_manifest.csv"
EXPERIMENT_LOG_PATH = DOCS_DIR / "experiment_log.csv"
TEST_METRICS_PATH = DOCS_DIR / "test_metrics.csv"
DEMO_CHECKLIST_PATH = DOCS_DIR / "demo_checklist.md"
REPORT_OUTLINE_PATH = DOCS_DIR / "report_outline.md"
FIGURES_DIR = DOCS_DIR / "figures"
REPORT_DIR = DOCS_DIR / "report"
