"""
Main pipeline orchestrator.

Usage:
    python pipeline.py               # collect → normalize → tag → publish
    python pipeline.py --dry-run     # skip Zenodo publish step
    python pipeline.py --train       # (re)train the ML rating predictor
    python pipeline.py --tag-only    # only run taggers on existing dataset
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from collectors import CodeforcesCollector, AtCoderCollector, KattisCollector
from normalizers import RatingScaler
from normalizers.rating_predictor import MLRatingPredictor
from normalizers.rating_predictor.base import BaseRatingPredictor
from taggers import SimilarityTagger, EnsembleTagger
from taggers.base import BaseTagger
from publishers import ZenodoPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("pipeline")

DATASET_PATH = Path("data/c0d3collective.csv")
CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_existing_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


def build_taggers(cfg: dict) -> list[BaseTagger]:
    taggers: list[BaseTagger] = []
    for tagger_cfg in cfg.get("taggers", []):
        name = tagger_cfg["name"]
        if name == "similarity_minilm":
            taggers.append(SimilarityTagger())
        # Future: "openai_gpt4o", "claude_sonnet", etc.
        else:
            logger.warning(f"Unknown tagger '{name}', skipping.")
    return taggers or [SimilarityTagger()]


def build_rating_predictors(cfg: dict) -> list[BaseRatingPredictor]:
    predictors: list[BaseRatingPredictor] = []
    for pred_cfg in cfg.get("rating_predictors", []):
        name = pred_cfg["name"]
        if name == "gradient_boosting":
            predictors.append(MLRatingPredictor())
        # Future: "openai_gpt4o", "claude_sonnet", etc.
        else:
            logger.warning(f"Unknown predictor '{name}', skipping.")
    return predictors or [MLRatingPredictor()]


def run(dry_run: bool = False, train: bool = False, tag_only: bool = False) -> None:
    cfg = load_config()
    scaler = RatingScaler()
    existing_df = load_existing_dataset()
    existing_ids = set(existing_df["id"].tolist()) if not existing_df.empty else set()

    # ── 1. Collect ──────────────────────────────────────────────────────────
    new_problems = []
    if not tag_only:
        collectors = [CodeforcesCollector(), AtCoderCollector(), KattisCollector()]
        for collector in collectors:
            if cfg.get("collectors", {}).get(collector.platform_name, {}).get("enabled", True):
                problems = collector.collect()
                new = [p for p in problems if p.id not in existing_ids]
                logger.info(f"{collector.platform_name}: {len(new)} new problems")
                new_problems.extend(new)

    # ── 2. Normalize ratings ─────────────────────────────────────────────────
    for p in new_problems:
        if p.cf_scaled_difficulty is None and p.rating is not None:
            p.cf_scaled_difficulty = scaler.scale(p.platform, p.rating)

    # ── 3. Predict ratings for unrated problems ───────────────────────────────
    predictors = build_rating_predictors(cfg)
    ml_predictor = predictors[0]  # baseline; ensemble when multiple configured
    if train:
        if hasattr(ml_predictor, "_train") and not existing_df.empty:
            ml_predictor._train(DATASET_PATH)

    for p in new_problems:
        if p.cf_scaled_difficulty is None:
            try:
                result = ml_predictor.predict(p.name, p.tags, p.platform, p.solved_count)
                p.cf_scaled_difficulty = result.cf_scaled_difficulty
                p.estimated = True
            except FileNotFoundError:
                logger.warning(f"Sem modelo ML treinado — pulando predição de rating para {p.id}")

    # ── 4. Tag untagged problems ─────────────────────────────────────────────
    tagger_list = build_taggers(cfg)
    ensemble = EnsembleTagger(tagger_list, majority_threshold=cfg.get("tag_majority_threshold", 0.5))

    # Build similarity corpus from existing tagged problems
    if any(isinstance(t, SimilarityTagger) for t in tagger_list):
        sim_tagger = next(t for t in tagger_list if isinstance(t, SimilarityTagger))
        if not existing_df.empty:
            tagged_df = existing_df[existing_df["tags"].notna() & (existing_df["tags"] != "")]
            corpus = [{"name": r["name"], "tags": str(r["tags"]).split("|")}
                      for _, r in tagged_df.iterrows()]
            sim_tagger.build_corpus(corpus)

    target_problems = new_problems if not tag_only else []
    if tag_only and not existing_df.empty:
        # Re-tag all untagged rows in existing dataset
        logger.info("Tag-only mode: re-tagging all untagged problems in existing dataset.")

    for p in target_problems:
        if not p.tags:
            result = ensemble.tag(p.id, p.name, p.platform, p.tags)
            p.tags = result.tags
            p.estimated = True

    # ── 5. Merge and save ────────────────────────────────────────────────────
    if new_problems:
        new_rows = [vars(p) for p in new_problems]
        for row in new_rows:
            row["tags"] = "|".join(row["tags"])
            row["fetched_at"] = row["fetched_at"].isoformat()

        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["id"], keep="last", inplace=True)

        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(DATASET_PATH, index=False)
        logger.info(f"Dataset saved: {len(combined)} total problems → {DATASET_PATH}")
    else:
        combined = existing_df
        logger.info("No new problems collected.")

    # ── 6. Generate stats.json ───────────────────────────────────────────────
    _generate_stats(combined, new_problems, existing_df)

    # ── 7. Publish to Zenodo ─────────────────────────────────────────────────
    if not dry_run and new_problems:
        from datetime import date
        try:
            publisher = ZenodoPublisher()
            record_url = publisher.publish_new_version(
                DATASET_PATH,
                metadata_update={"version": f"{date.today().isoformat()}-auto"},
            )
            logger.info(f"Zenodo record published: {record_url}")
        except Exception as e:
            logger.error(f"Zenodo publish falhou (CSV foi salvo localmente): {e}")
    elif dry_run:
        logger.info("Dry-run mode: Zenodo publish skipped.")


def _generate_stats(combined: pd.DataFrame, new_problems: list, existing_df: pd.DataFrame) -> None:
    """Write data/stats.json with current dataset stats and append this run to history."""
    import json
    from datetime import date, timezone

    stats_path = Path("data/stats.json")

    # Load existing stats to preserve run history
    existing_stats: dict = {}
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                existing_stats = json.load(f)
        except Exception:
            existing_stats = {}

    runs_history: list = existing_stats.get("runs", [])

    def platform_stats(df: pd.DataFrame, platform: str) -> dict:
        sub = df[df["platform"] == platform] if not df.empty else pd.DataFrame()
        total = len(sub)
        with_rating = int(sub["cf_scaled_difficulty"].notna().sum()) if total else 0
        estimated = int(sub["estimated"].astype(str).str.lower().isin(["true", "1"]).sum()) if total else 0
        with_tags = int((sub["tags"].notna() & (sub["tags"].astype(str).str.strip() != "")).sum()) if total else 0
        return {"total": total, "with_rating": with_rating, "estimated": estimated, "with_tags": with_tags}

    platforms = ["codeforces", "atcoder", "kattis"]
    by_platform = {p: platform_stats(combined, p) for p in platforms}

    added_by_platform = {}
    for p in platforms:
        added_by_platform[p] = len([x for x in new_problems if x.platform == p])

    total = len(combined)
    with_rating = int(combined["cf_scaled_difficulty"].notna().sum()) if total else 0
    with_tags = int((combined["tags"].notna() & (combined["tags"].astype(str).str.strip() != "")).sum()) if total else 0

    today = date.today().isoformat()
    run_entry = {
        "date": today,
        "total_after": total,
        "added": {"total": len(new_problems), **added_by_platform},
    }
    # Avoid duplicate entries for the same date
    runs_history = [r for r in runs_history if r.get("date") != today]
    runs_history.append(run_entry)

    stats = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "with_rating": with_rating,
        "rating_pct": round(with_rating / total * 100, 1) if total else 0,
        "with_tags": with_tags,
        "tags_pct": round(with_tags / total * 100, 1) if total else 0,
        "by_platform": by_platform,
        "runs": runs_history,
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved → {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C0D3Collective pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip Zenodo publish")
    parser.add_argument("--train", action="store_true", help="Retrain ML predictor")
    parser.add_argument("--tag-only", action="store_true", help="Only re-tag existing dataset")
    args = parser.parse_args()
    run(dry_run=args.dry_run, train=args.train, tag_only=args.tag_only)
