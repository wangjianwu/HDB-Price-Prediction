"""Command-line entry point for the HDB resale-price pipeline."""
import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
REQUIRED_CONFIG = {
    "file_path", "target_column", "val_test_size", "val_size", "param_grid",
    "cv", "scoring", "random_state", "lasso_param_grid", "numerical_features", "nominal_features",
    "ordinal_features", "passthrough_features", "flat_type_categories",
    "output_dir", "model_dir",
}


def load_config(path: Path) -> dict:
    """Load and validate pipeline configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    missing = sorted(REQUIRED_CONFIG.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    for key in ("val_test_size", "val_size"):
        if not 0 < config[key] < 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    if config["cv"] < 2:
        raise ValueError("cv must be at least 2.")
    return config


def load_data(path: Path) -> pd.DataFrame:
    """Read the source CSV with useful logging and errors."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse dataset {path}: {exc}") from exc
    logging.info("Loaded %d rows and %d columns from %s", *frame.shape, path)
    return frame


def data_quality_report(frame: pd.DataFrame, target: str) -> dict:
    """Return JSON-serializable quality checks before cleaning."""
    report = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items()},
        "categorical_cardinality": {
            c: int(frame[c].nunique(dropna=True))
            for c in ("town_name", "flat_type", "flatm_name") if c in frame
        },
    }
    if target in frame:
        report["target_summary"] = {
            k: float(v) for k, v in frame[target].describe().items()
        }
        report["non_positive_target_rows"] = int((frame[target] <= 0).sum())
    return report


def run(config_path: Path) -> dict:
    """Run training, persist artifacts, and return final test metrics."""
    config_path = config_path.resolve()
    project_dir = config_path.parent.parent
    config = load_config(config_path)
    data_path = Path(config["file_path"])
    if not data_path.is_absolute():
        data_path = project_dir / data_path

    outputs = project_dir / config["output_dir"]
    models = project_dir / config["model_dir"]
    outputs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    frame = load_data(data_path)
    (outputs / "data_quality_report.json").write_text(
        json.dumps(data_quality_report(frame, config["target_column"]), indent=2),
        encoding="utf-8",
    )

    preparation = DataPreparation(config)
    cleaned = preparation.clean_data(frame)
    trainer = ModelTraining(config, preparation.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(cleaned)

    baseline_models, baseline_metrics = trainer.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )
    tuned_models, tuned_metrics = trainer.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    best_name = min(all_metrics, key=lambda name: all_metrics[name]["RMSE"])
    logging.info("Best validation model by RMSE: %s", best_name)
    final_model = trainer.refit_model(
        all_models[best_name], X_train, y_train, X_val, y_val
    )
    final_metrics = trainer.evaluate_final_model(
        final_model, X_test, y_test, best_name
    )

    result = {
        "selected_model": best_name,
        "validation_metrics": all_metrics,
        "test_metrics": final_metrics,
    }
    (outputs / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trainer.make_predictions(final_model, X_test, y_test).to_csv(
        outputs / "test_predictions.csv", index=False
    )
    joblib.dump(final_model, models / "hdb_resale_price_pipeline.joblib")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HDB resale-price model.")
    parser.add_argument(
        "--config", type=Path, default=Path("src/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    try:
        result = run(args.config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logging.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result["test_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
