import argparse
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from scapy.all import AsyncSniffer, ICMP, IP, TCP, UDP

SNIFFER_FEATURES = [
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


class OnlineStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_value = float("inf")
        self.max_value = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / self.count)

    def min(self) -> float:
        return 0.0 if self.count == 0 else self.min_value

    def max(self) -> float:
        return 0.0 if self.count == 0 else self.max_value


class FlowState:
    def __init__(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: int,
        first_ts: float,
    ) -> None:
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.start_ts = first_ts
        self.last_ts = first_ts

        self.total_fwd_packets = 0
        self.total_bwd_packets = 0
        self.total_fwd_bytes = 0
        self.total_bwd_bytes = 0

        self.fwd_stats = OnlineStats()
        self.bwd_stats = OnlineStats()
        self.all_stats = OnlineStats()
        self.fwd_iat_stats = OnlineStats()
        self.bwd_iat_stats = OnlineStats()
        self.last_fwd_ts: Optional[float] = None
        self.last_bwd_ts: Optional[float] = None

    def update(self, pkt_len: int, ts: float, direction: str) -> None:
        self.last_ts = ts
        self.all_stats.update(float(pkt_len))

        if direction == "fwd":
            self.total_fwd_packets += 1
            self.total_fwd_bytes += pkt_len
            self.fwd_stats.update(float(pkt_len))
            if self.last_fwd_ts is not None:
                self.fwd_iat_stats.update((ts - self.last_fwd_ts) * 1e6)
            self.last_fwd_ts = ts
        else:
            self.total_bwd_packets += 1
            self.total_bwd_bytes += pkt_len
            self.bwd_stats.update(float(pkt_len))
            if self.last_bwd_ts is not None:
                self.bwd_iat_stats.update((ts - self.last_bwd_ts) * 1e6)
            self.last_bwd_ts = ts

    def to_features(self) -> Dict[str, float]:
        duration_us = max((self.last_ts - self.start_ts) * 1e6, 0.0)
        duration_s = max(self.last_ts - self.start_ts, 0.0)
        total_bytes = self.total_fwd_bytes + self.total_bwd_bytes
        total_packets = self.total_fwd_packets + self.total_bwd_packets

        flow_bytes_per_s = total_bytes / duration_s if duration_s > 0 else 0.0
        flow_packets_per_s = total_packets / duration_s if duration_s > 0 else 0.0

        return {
            "dst_port": float(self.dst_port),
            "flow_duration": float(duration_us),
            "total_fwd_packets": float(self.total_fwd_packets),
            "total_backward_packets": float(self.total_bwd_packets),
            "total_length_fwd_packets": float(self.total_fwd_bytes),
            "total_length_bwd_packets": float(self.total_bwd_bytes),
            "fwd_packet_length_max": float(self.fwd_stats.max()),
            "fwd_packet_length_min": float(self.fwd_stats.min()),
            "fwd_packet_length_mean": float(self.fwd_stats.mean),
            "fwd_packet_length_std": float(self.fwd_stats.std()),
            "bwd_packet_length_max": float(self.bwd_stats.max()),
            "bwd_packet_length_min": float(self.bwd_stats.min()),
            "bwd_packet_length_mean": float(self.bwd_stats.mean),
            "bwd_packet_length_std": float(self.bwd_stats.std()),
            "flow_bytes_per_s": float(flow_bytes_per_s),
            "flow_packets_per_s": float(flow_packets_per_s),
            "packet_length_mean": float(self.all_stats.mean),
            "packet_length_std": float(self.all_stats.std()),
            "fwd_iat_mean": float(self.fwd_iat_stats.mean),
            "bwd_iat_mean": float(self.bwd_iat_stats.mean),
            "protocol": float(self.protocol),
        }


@dataclass
class RuntimeArtifacts:
    scaler: object
    feature_columns: List[str]
    class_names: List[str]
    profile: str


@dataclass
class Alert:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    prediction: str
    score: float
    flow_duration_us: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": datetime.utcfromtimestamp(self.timestamp).isoformat() + "Z",
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": int(self.src_port),
            "dst_port": int(self.dst_port),
            "protocol": int(self.protocol),
            "prediction": self.prediction,
            "score": float(self.score),
            "flow_duration_us": float(self.flow_duration_us),
        }


def load_runtime(model_dir: str) -> Tuple[tf.keras.Model, RuntimeArtifacts]:
    model_path = os.path.join(model_dir, "ids_model.keras")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    metadata: Dict[str, object] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    bundle = joblib.load(preprocessor_path)
    feature_columns = bundle.get("feature_columns") or metadata.get("feature_columns")
    class_names = bundle.get("label_encoder").classes_.tolist() if bundle.get("label_encoder") else metadata.get("class_names")
    profile = bundle.get("profile") or metadata.get("profile") or "unknown"

    if not feature_columns:
        raise ValueError("Missing feature columns in preprocessor metadata")
    if not class_names:
        raise ValueError("Missing class names in preprocessor metadata")

    missing = [f for f in feature_columns if f not in SNIFFER_FEATURES]
    if missing:
        raise ValueError(f"Model expects unsupported features: {missing}")

    model = tf.keras.models.load_model(model_path)
    artifacts = RuntimeArtifacts(
        scaler=bundle["scaler"],
        feature_columns=feature_columns,
        class_names=class_names,
        profile=profile,
    )
    return model, artifacts


class PacketSniffer:
    def __init__(
        self,
        model_dir: str,
        interface: Optional[str] = None,
        window_seconds: int = 5,
        prediction_threshold: float = 0.6,
        cleanup_interval: float = 1.0,
    ) -> None:
        self.model, self.artifacts = load_runtime(model_dir)
        self.interface = interface
        self.window_seconds = window_seconds
        self.prediction_threshold = prediction_threshold
        self.cleanup_interval = cleanup_interval

        self._flows: Dict[Tuple[str, str, int, int, int], FlowState] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Alert]" = queue.Queue()
        self._sniffer: Optional[AsyncSniffer] = None
        self._last_cleanup = time.time()

    @property
    def is_running(self) -> bool:
        return self._sniffer is not None

    def start(self) -> None:
        if self._sniffer is not None:
            return
        self._sniffer = AsyncSniffer(prn=self._handle_packet, store=False, iface=self.interface)
        self._sniffer.start()

    def stop(self) -> None:
        if self._sniffer is None:
            return
        self._sniffer.stop()
        self._sniffer = None
        self._flush_all()

    def _handle_packet(self, pkt) -> None:
        parsed = self._parse_packet(pkt)
        if not parsed:
            return
        flow_key, reverse_key, pkt_len, ts, direction = parsed

        with self._lock:
            if direction == "fwd":
                flow = self._flows.get(flow_key)
            else:
                flow = self._flows.get(reverse_key)

            if flow is None:
                src_ip, dst_ip, src_port, dst_port, protocol = flow_key
                flow = FlowState(src_ip, dst_ip, src_port, dst_port, protocol, ts)
                self._flows[flow_key] = flow
                direction = "fwd"

            flow.update(pkt_len, ts, direction)

            now = time.time()
            if now - self._last_cleanup >= self.cleanup_interval:
                self._flush_expired(now)
                self._last_cleanup = now

    def _parse_packet(self, pkt) -> Optional[Tuple[Tuple[str, str, int, int, int], Tuple[str, str, int, int, int], int, float, str]]:
        if IP not in pkt:
            return None

        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        protocol = int(ip.proto)

        if TCP in pkt:
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
            protocol = 6
        elif UDP in pkt:
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
            protocol = 17
        elif ICMP in pkt:
            src_port = 0
            dst_port = 0
            protocol = 1
        else:
            return None

        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        reverse_key = (dst_ip, src_ip, dst_port, src_port, protocol)

        direction = "fwd" if flow_key in self._flows or reverse_key not in self._flows else "bwd"
        pkt_len = int(len(pkt))
        ts = float(time.time())
        return flow_key, reverse_key, pkt_len, ts, direction

    def _flush_expired(self, now: float) -> None:
        expired_keys = [
            key for key, flow in self._flows.items() if now - flow.last_ts >= self.window_seconds
        ]
        for key in expired_keys:
            flow = self._flows.pop(key)
            alert = self._predict_flow(flow)
            if alert:
                self._queue.put(alert)

    def _flush_all(self) -> None:
        now = time.time()
        for key in list(self._flows.keys()):
            flow = self._flows.pop(key)
            alert = self._predict_flow(flow)
            if alert:
                self._queue.put(alert)
        self._last_cleanup = now

    def _predict_flow(self, flow: FlowState) -> Optional[Alert]:
        features = flow.to_features()
        row = np.array([features[col] for col in self.artifacts.feature_columns], dtype=np.float32).reshape(1, -1)
        scaled = self.artifacts.scaler.transform(row)
        probabilities = self.model.predict(scaled, verbose=0)[0]
        idx = int(np.argmax(probabilities))
        score = float(probabilities[idx])
        label = self.artifacts.class_names[idx]
        if score < self.prediction_threshold:
            label = "Unknown"

        return Alert(
            timestamp=time.time(),
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            src_port=flow.src_port,
            dst_port=flow.dst_port,
            protocol=flow.protocol,
            prediction=label,
            score=score,
            flow_duration_us=features["flow_duration"],
        )

    def drain_alerts(self, limit: int = 200) -> List[Dict[str, object]]:
        alerts: List[Dict[str, object]] = []
        while len(alerts) < limit:
            try:
                alert = self._queue.get_nowait()
            except queue.Empty:
                break
            alerts.append(alert.to_dict())
        return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Live packet sniffer for IDS")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    sniffer = PacketSniffer(
        model_dir=args.model_dir,
        interface=args.interface,
        window_seconds=args.window,
        prediction_threshold=args.threshold,
    )

    print("Starting sniffer...")
    sniffer.start()
    try:
        while True:
            alerts = sniffer.drain_alerts(limit=50)
            for alert in alerts:
                print(alert)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
        print("Sniffer stopped")


if __name__ == "__main__":
    main()
