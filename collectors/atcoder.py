import logging
import requests
from datetime import datetime

from .base import BaseCollector, Problem

logger = logging.getLogger(__name__)

PROBLEMS_API = "https://kenkoooo.com/atcoder/resources/merged-problems.json"
MODELS_API = "https://kenkoooo.com/atcoder/resources/problem-models.json"


class AtCoderCollector(BaseCollector):
    @property
    def platform_name(self) -> str:
        return "atcoder"

    def collect(self) -> list[Problem]:
        logger.info("Fetching AtCoder problems via Kenkoooo API...")
        problems_resp = requests.get(PROBLEMS_API, timeout=30)
        problems_resp.raise_for_status()

        models_resp = requests.get(MODELS_API, timeout=30)
        models_resp.raise_for_status()

        problems = problems_resp.json()
        models = models_resp.json()  # pid → {"difficulty": float, ...}

        result = []
        for p in problems:
            pid = p.get("id", "")
            contest_id = p.get("contest_id", "")
            difficulty = models.get(pid, {}).get("difficulty")  # IRT-based

            result.append(Problem(
                id=f"atcoder_{pid}",
                platform="atcoder",
                problem_id=pid,
                name=p.get("title", ""),
                url=f"https://atcoder.jp/contests/{contest_id}/tasks/{pid}",
                tags=[],  # AtCoder has no native tags — filled by tagger module
                rating=difficulty,
                cf_scaled_difficulty=None,  # computed by normalizer
                solved_count=p.get("solver_count"),
                contest_id=contest_id,
                fetched_at=datetime.utcnow(),
            ))

        logger.info(f"Collected {len(result)} AtCoder problems.")
        return result
