import logging
import requests
from datetime import datetime

from .base import BaseCollector, Problem

logger = logging.getLogger(__name__)

# Kattis does not have an official public API.
# This collector uses the community-maintained open.kattis.com listing.
# Fragile: may break if Kattis changes their HTML structure.
KATTIS_PROBLEMS_URL = "https://open.kattis.com/problems?show_solved=on&show_tried=on&show_untried=on"


class KattisCollector(BaseCollector):
    @property
    def platform_name(self) -> str:
        return "kattis"

    def collect(self) -> list[Problem]:
        logger.info("Fetching Kattis problems (web scraping)...")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise RuntimeError("Install beautifulsoup4 to use KattisCollector: pip install beautifulsoup4")

        result = []
        page = 0
        while True:
            url = f"{KATTIS_PROBLEMS_URL}&page={page}"
            resp = requests.get(url, timeout=30, headers={"User-Agent": "c0d3collective-pipeline/1.0"})
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table.table-problems tbody tr")
            if not rows:
                break

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue

                pid = cols[0].find("a")
                if not pid:
                    continue
                problem_id = pid["href"].split("/")[-1]
                name = pid.text.strip()

                difficulty_text = cols[-1].text.strip()
                try:
                    difficulty = float(difficulty_text)
                except ValueError:
                    difficulty = None

                result.append(Problem(
                    id=f"kattis_{problem_id}",
                    platform="kattis",
                    problem_id=problem_id,
                    name=name,
                    url=f"https://open.kattis.com/problems/{problem_id}",
                    tags=[],
                    rating=difficulty,
                    cf_scaled_difficulty=None,  # computed by normalizer
                    solved_count=None,
                    contest_id=None,
                    fetched_at=datetime.utcnow(),
                ))

            page += 1

        logger.info(f"Collected {len(result)} Kattis problems.")
        return result
