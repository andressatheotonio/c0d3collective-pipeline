"""
Publishes a new dataset version to Zenodo via REST API.

Flow:
1. newversion  → creates a draft copy of the latest published version
2. Upload file → PUT the new CSV to the draft's bucket
3. Publish     → POST to make the draft publicly available

Requires environment variable: ZENODO_TOKEN (store as GitHub Actions secret).
The concept DOI (10.5281/zenodo.16294115) always points to the latest version.
"""

import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ZENODO_BASE = "https://zenodo.org/api"
CONCEPT_RECORD_ID = "16294115"  # versão atual; newversion cria a próxima a partir desta


class ZenodoPublisher:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("ZENODO_TOKEN")
        if not self.token:
            raise EnvironmentError("ZENODO_TOKEN not set.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _get_latest_deposition_id(self) -> str:
        """Resolve concept record → latest published deposition id."""
        resp = self.session.get(f"{ZENODO_BASE}/records/{CONCEPT_RECORD_ID}")
        resp.raise_for_status()
        return str(resp.json()["id"])

    def publish_new_version(self, csv_path: Path, metadata_update: dict | None = None) -> str:
        """Upload csv_path as a new Zenodo version. Returns the new record URL."""
        dep_id = self._get_latest_deposition_id()
        logger.info(f"Latest deposition id: {dep_id}")

        # 1. Create new version draft
        draft_resp = self.session.post(f"{ZENODO_BASE}/deposit/depositions/{dep_id}/actions/newversion")
        draft_resp.raise_for_status()
        new_dep_url = draft_resp.json()["links"]["latest_draft"]
        new_dep_id = new_dep_url.split("/")[-1]
        logger.info(f"Draft version created: {new_dep_id}")

        # 2. Remove old files from draft
        draft = self.session.get(new_dep_url)
        draft.raise_for_status()
        bucket_url = draft.json()["links"]["bucket"]
        for f in draft.json().get("files", []):
            self.session.delete(f"{ZENODO_BASE}/deposit/depositions/{new_dep_id}/files/{f['id']}")

        # 3. Upload new CSV
        with open(csv_path, "rb") as fh:
            upload_resp = self.session.put(
                f"{bucket_url}/{csv_path.name}",
                data=fh,
            )
        upload_resp.raise_for_status()
        logger.info(f"Uploaded {csv_path.name} ({csv_path.stat().st_size:,} bytes)")

        # 4. Optionally update metadata (e.g., version string, description)
        if metadata_update:
            meta_resp = self.session.put(
                f"{ZENODO_BASE}/deposit/depositions/{new_dep_id}",
                json={"metadata": metadata_update},
            )
            meta_resp.raise_for_status()

        # 5. Publish draft
        pub_resp = self.session.post(
            f"{ZENODO_BASE}/deposit/depositions/{new_dep_id}/actions/publish"
        )
        pub_resp.raise_for_status()
        record_url = pub_resp.json()["links"]["record_html"]
        logger.info(f"Published: {record_url}")
        return record_url
