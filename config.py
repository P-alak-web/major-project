from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Dataset paths
FEATURES_FILE = (
    ROOT / "processed" / "user_daily_features_advanced_v2.csv"
)

GROUND_TRUTH_FILE = (
    ROOT / "outputs" / "ground_truth" / "ground_truth_user_day.csv"
)

# Chronological split
VALIDATION_DATE = pd.Timestamp("2010-09-01")
TEST_DATE = pd.Timestamp("2010-11-01")

# Experiment settings
RANDOM_SEED = 42
MIN_TRAIN_DAYS = 30
ALERT_BUDGETS = [10, 50, 100]

# Common behavioral features
FEATURE_COLUMNS = [
    "total_logon_events",
    "logon_count",
    "logoff_count",
    "after_hours_logons",
    "http_requests",
    "after_hours_http",
    "email_count",
    "after_hours_email",
    "usb_events",
    "usb_connect_count",
    "usb_disconnect_count",
    "after_hours_usb",
    "file_operations",
    "after_hours_file",
    "external_email_count",
    "external_attachment_count",
    "new_pc_count",
    "unique_pc_count",
    "unique_domains",
    "new_domain_count",
    "cloud_storage_visits",
    "job_site_visits"
]