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
        # Kattis paginates from page=1; page=0 redirects to page=1
        page = 1
        while True:
            url = f"{KATTIS_PROBLEMS_URL}&page={page}"
            resp = requests.get(url, timeout=30, headers={"User-Agent": "c0d3collective-pipeline/1.0"})
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            # Kattis uses two <table class="table2">: first is a filter widget,
            # second is the problems listing. The class may have a trailing space.
            tables = soup.find_all("table", class_="table2")
            problems_table = tables[-1] if tables else None
            if not problems_table:
                logger.warning(f"Kattis: tabela não encontrada na página {page}.")
                break

            rows = problems_table.find_all("tr")[1:]  # skip header row
            if not rows:
                break

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 7:
                    continue

                link = cols[0].find("a", href=lambda h: h and "/problems/" in h)
                if not link:
                    continue
                problem_id = link["href"].rstrip("/").split("/")[-1]
                name = link.text.strip()

                # Difficulty column (index 6): "6.3\n    Hard" — take first token
                difficulty_text = cols[6].text.strip().split()[0] if cols[6].text.strip() else ""
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
                    cf_scaled_difficulty=None,
                    solved_count=None,
                    contest_id=None,
                    fetched_at=datetime.utcnow(),
                ))

            page += 1

        logger.info(f"Collected {len(result)} Kattis problems.")
        return result
