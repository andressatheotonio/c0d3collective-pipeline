import logging
import requests
from datetime import datetime

from .base import BaseCollector, Problem

logger = logging.getLogger(__name__)

CF_API = "https://codeforces.com/api/problemset.problems"


class CodeforcesCollector(BaseCollector):
    @property
    def platform_name(self) -> str:
        return "codeforces"

    def collect(self) -> list[Problem]:
        logger.info("Fetching Codeforces problemset...")
        resp = requests.get(CF_API, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "OK":
            raise RuntimeError(f"Codeforces API error: {data}")

        problems = data["result"]["problems"]
        stats_by_key = {
            (s["contestId"], s["index"]): s["solvedCount"]
            for s in data["result"]["problemStatistics"]
        }

        result = []
        for p in problems:
            contest_id = str(p.get("contestId", ""))
            index = p.get("index", "")
            rating = p.get("rating")

            result.append(Problem(
                id=f"codeforces_{contest_id}{index}",
                platform="codeforces",
                problem_id=f"{contest_id}{index}",
                name=p.get("name", ""),
                url=f"https://codeforces.com/problemset/problem/{contest_id}/{index}",
                tags=p.get("tags", []),
                rating=float(rating) if rating else None,
                cf_scaled_difficulty=float(rating) if rating else None,
                solved_count=stats_by_key.get((p.get("contestId"), index)),
                contest_id=contest_id,
                fetched_at=datetime.utcnow(),
            ))

        logger.info(f"Collected {len(result)} Codeforces problems.")
        return result
