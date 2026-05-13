import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocessor import prepare_datasets, save_artifacts


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDS classifier")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--profile", choices=["cicids", "nsl_kdd", "auto"], default="cicids")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--include-benign", action="store_true")
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    X_train, X_test, y_train, y_test, artifacts = prepare_datasets(
        data_dir=args.data_dir,
        profile=args.profile,
        include_benign=args.include_benign,
        test_size=0.2,
        random_state=args.random_state,
        sample_frac=args.sample_frac,
        use_smote=not args.no_smote,
    )

    class_names = artifacts.label_encoder.classes_.tolist()
    num_classes = len(class_names)
    if not args.include_benign and num_classes != 4:
        print(f"Warning: expected 4 classes, got {num_classes}: {class_names}")

    model = build_model(X_train.shape[1], num_classes)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "ids_model.keras")
    model.save(model_path)

    stats = {
        "num_features": int(X_train.shape[1]),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "model_path": "ids_model.keras",
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    save_artifacts(artifacts, args.model_dir, stats=stats)

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs": len(history.history.get("loss", [])),
        "class_names": class_names,
    }
    with open(os.path.join(args.model_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
