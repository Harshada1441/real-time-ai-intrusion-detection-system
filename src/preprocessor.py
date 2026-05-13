import argparse
import glob
import json
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - runtime environment dependency
    SMOTE = None

CICIDS_FEATURES = [
    "dst_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_fwd_packets",
    "total_length_bwd_packets",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "packet_length_mean",
    "packet_length_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "protocol",
]

LABEL_COLUMN_CANDIDATES = {"label", "attack", "class"}

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

NSL_KDD_DOS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "apache2",
    "udpstorm",
    "processtable",
    "worm",
}
NSL_KDD_PROBE = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
NSL_KDD_R2L = {
    "guess_passwd",
    "ftp_write",
    "imap",
    "phf",
    "multihop",
    "warezmaster",
    "warezclient",
    "spy",
    "xlock",
    "xsnoop",
    "snmpguess",
    "snmpgetattack",
    "httptunnel",
    "sendmail",
    "named",
}
NSL_KDD_U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"}

NSL_KDD_ATTACKS = {
    "DOS": NSL_KDD_DOS,
    "Probe": NSL_KDD_PROBE,
    "R2L": NSL_KDD_R2L,
    "U2R": NSL_KDD_U2R,
}


def normalize_column(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


CICIDS_COLUMN_RENAMES = {
    "destination_port": "dst_port",
    "flow_duration": "flow_duration",
    "total_fwd_packets": "total_fwd_packets",
    "total_backward_packets": "total_backward_packets",
    "total_length_of_fwd_packets": "total_length_fwd_packets",
    "total_length_of_bwd_packets": "total_length_bwd_packets",
    "fwd_packet_length_max": "fwd_packet_length_max",
    "fwd_packet_length_min": "fwd_packet_length_min",
    "fwd_packet_length_mean": "fwd_packet_length_mean",
    "fwd_packet_length_std": "fwd_packet_length_std",
    "bwd_packet_length_max": "bwd_packet_length_max",
    "bwd_packet_length_min": "bwd_packet_length_min",
    "bwd_packet_length_mean": "bwd_packet_length_mean",
    "bwd_packet_length_std": "bwd_packet_length_std",
    "flow_bytes_s": "flow_bytes_per_s",
    "flow_bytes_per_s": "flow_bytes_per_s",
    "flow_packets_s": "flow_packets_per_s",
    "flow_packets_per_s": "flow_packets_per_s",
    "packet_length_mean": "packet_length_mean",
    "packet_length_std": "packet_length_std",
    "fwd_iat_mean": "fwd_iat_mean",
    "bwd_iat_mean": "bwd_iat_mean",
    "protocol": "protocol",
}

CICIDS_COLUMN_MAP = {normalize_column(k): v for k, v in CICIDS_COLUMN_RENAMES.items()}


@dataclass
class PreprocessArtifacts:
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_columns: List[str]
    profile: str
    include_benign: bool


def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def discover_csv_files(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def discover_nsl_files(data_dir: str) -> List[str]:
    patterns = ["*KDD*.txt", "*KDD*.csv", "*nsl*.txt", "*nsl*.csv"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))
    return sorted(set(files))


def is_cicids_frame(df: pd.DataFrame) -> bool:
    cols = {normalize_column(c) for c in df.columns}
    return "label" in cols and "flow_duration" in cols and ("destination_port" in cols or "dst_port" in cols)


def normalize_cicids_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        norm = normalize_column(col)
        if norm in CICIDS_COLUMN_MAP:
            rename_map[col] = CICIDS_COLUMN_MAP[norm]
        elif norm in LABEL_COLUMN_CANDIDATES:
            rename_map[col] = "label"
    return df.rename(columns=rename_map)


def map_cicids_label(raw: str, include_benign: bool) -> Optional[str]:
    label = str(raw).strip().lower()
    if label == "benign":
        return "BENIGN" if include_benign else None
    if "ddos" in label or "dos" in label:
        return "DOS"
    if "portscan" in label or "scan" in label:
        return "Probe"
    if any(k in label for k in ["brute", "ssh", "ftp", "sql", "xss", "web attack", "patator"]):
        return "R2L"
    if any(k in label for k in ["infiltration", "heartbleed"]):
        return "U2R"
    if "bot" in label:
        return "R2L"
    return None


def map_nsl_kdd_label(raw: str, include_benign: bool) -> Optional[str]:
    label = str(raw).strip().lower()
    if label == "normal":
        return "BENIGN" if include_benign else None
    for category, labels in NSL_KDD_ATTACKS.items():
        if label in labels:
            return category
    return None


def read_nsl_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".txt"):
        return pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    df = read_csv_safe(path)
    normalized = [normalize_column(c) for c in df.columns]
    if "label" not in normalized and df.shape[1] == len(NSL_KDD_COLUMNS):
        return pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    df.columns = normalized
    return df


def load_cicids(
    data_dir: str,
    include_benign: bool,
    sample_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, List[str]]:
    files = discover_csv_files(data_dir)
    if not files:
        raise ValueError("No CSV files found for CICIDS2017")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = read_csv_safe(path)
        if not is_cicids_frame(df):
            continue
        df = normalize_cicids_frame(df)
        if "label" not in df.columns:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("No CICIDS2017 files matched expected columns")

    df = pd.concat(frames, ignore_index=True)
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].apply(lambda x: map_cicids_label(x, include_benign))
    df = df.dropna(subset=["label"])

    if "protocol" not in df.columns:
        df["protocol"] = 0

    missing = [c for c in CICIDS_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CICIDS columns: {missing}")

    df = df[CICIDS_FEATURES + ["label"]]
    for col in CICIDS_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, CICIDS_FEATURES


def load_nsl_kdd(
    data_dir: str,
    include_benign: bool,
    sample_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, List[str]]:
    files = discover_nsl_files(data_dir)
    if not files:
        raise ValueError("No NSL-KDD files found")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = read_nsl_file(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    if "label" not in df.columns:
        raise ValueError("NSL-KDD data missing label column")

    df["label"] = df["label"].apply(lambda x: map_nsl_kdd_label(x, include_benign))
    df = df.dropna(subset=["label"])
    df = df.drop(columns=["difficulty"], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    categorical = [c for c in ["protocol_type", "service", "flag"] if c in df.columns]
    if categorical:
        df = pd.get_dummies(df, columns=categorical)

    feature_columns = [c for c in df.columns if c != "label"]
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_columns = sorted([c for c in df.columns if c != "label"])
    df = df[feature_columns + ["label"]]
    return df, feature_columns


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if SMOTE is None:
        warnings.warn("imbalanced-learn not available; proceeding without SMOTE")
        return X, y
    counts = Counter(y)
    if not counts:
        return X, y

    min_count = min(counts.values())
    if min_count < 2:
        return X, y

    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X, y)


def load_profile(
    profile: str,
    data_dir: str,
    include_benign: bool,
    sample_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, List[str]]:
    if profile == "cicids":
        return load_cicids(data_dir, include_benign, sample_frac, random_state)
    if profile == "nsl_kdd":
        return load_nsl_kdd(data_dir, include_benign, sample_frac, random_state)
    if profile == "auto":
        try:
            return load_cicids(data_dir, include_benign, sample_frac, random_state)
        except Exception:
            return load_nsl_kdd(data_dir, include_benign, sample_frac, random_state)
    raise ValueError(f"Unknown profile: {profile}")


def prepare_datasets(
    data_dir: str,
    profile: str,
    include_benign: bool,
    test_size: float,
    random_state: int,
    sample_frac: float,
    use_smote: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessArtifacts]:
    if sample_frac <= 0 or sample_frac > 1.0:
        raise ValueError("sample_frac must be in (0, 1]")

    df, feature_columns = load_profile(profile, data_dir, include_benign, sample_frac, random_state)

    X = df[feature_columns].astype(np.float32).values
    y = df["label"].astype(str).values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    artifacts = PreprocessArtifacts(
        scaler=scaler,
        label_encoder=label_encoder,
        feature_columns=feature_columns,
        profile=profile,
        include_benign=include_benign,
    )

    return X_train, X_test, y_train, y_test, artifacts


def save_artifacts(
    artifacts: PreprocessArtifacts,
    output_dir: str,
    stats: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(
        {
            "scaler": artifacts.scaler,
            "label_encoder": artifacts.label_encoder,
            "feature_columns": artifacts.feature_columns,
            "profile": artifacts.profile,
            "include_benign": artifacts.include_benign,
        },
        os.path.join(output_dir, "preprocessor.joblib"),
    )

    metadata: Dict[str, object] = {
        "profile": artifacts.profile,
        "include_benign": artifacts.include_benign,
        "feature_columns": artifacts.feature_columns,
        "class_names": artifacts.label_encoder.classes_.tolist(),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    if stats:
        metadata.update(stats)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NSL-KDD or CICIDS2017 datasets")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--profile", choices=["cicids", "nsl_kdd", "auto"], default="cicids")
    parser.add_argument("--include-benign", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, artifacts = prepare_datasets(
        data_dir=args.data_dir,
        profile=args.profile,
        include_benign=args.include_benign,
        test_size=args.test_size,
        random_state=args.random_state,
        sample_frac=args.sample_frac,
        use_smote=not args.no_smote,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "train_data.npz"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    stats = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "num_features": int(X_train.shape[1]),
    }

    save_artifacts(artifacts, args.output_dir, stats=stats)

    print("Preprocessing complete")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


if __name__ == "__main__":
    main()
