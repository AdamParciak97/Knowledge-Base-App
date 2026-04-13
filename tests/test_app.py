from __future__ import annotations

import asyncio
import io
import os
import shutil
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

os.environ["KB_TEST_MODE"] = "1"
os.environ["KB_PASSWORD"] = "change-me"
os.environ["KB_RATE_LIMIT_PER_MINUTE"] = "1000"

from fastapi.testclient import TestClient

from app.main import app
from app.services.kb import kb_service


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = Path(tempfile.mkdtemp(prefix="kb-tests-"))
        kb_service.base_dir = cls.tempdir
        kb_service.data_dir = cls.tempdir / "data"
        kb_service.uploads_dir = kb_service.data_dir / "uploads"
        kb_service.texts_dir = kb_service.data_dir / "texts"
        kb_service.gguf_dir = kb_service.data_dir / "gguf"
        kb_service.state_path = kb_service.data_dir / "kb_state.json"
        kb_service.admin_path = kb_service.data_dir / "admin_state.json"
        kb_service.model_dir = cls.tempdir / "models" / "all-MiniLM-L6-v2"
        kb_service.template_path = Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html"
        kb_service._startup_done = False
        kb_service.state = kb_service.state.__class__()
        kb_service.admin_state = kb_service.admin_state.__class__()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        if kb_service._build_loop_task and not kb_service._build_loop_task.done():
            kb_service._build_loop_task.cancel()
            try:
                asyncio.run(asyncio.wait_for(kb_service._build_loop_task, timeout=1))
            except Exception:
                pass
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    def login(self) -> None:
        response = self.client.post("/auth/login", data={"password": "change-me"})
        self.assertEqual(response.status_code, 200)

    def wait_for_build(self) -> dict:
        deadline = time.time() + 10
        while time.time() < deadline:
            response = self.client.get("/status")
            if response.status_code == 200:
                payload = response.json()
                if not payload["build"]["in_progress"] and payload["queue_depth"] == 0:
                    return payload
            time.sleep(0.1)
        self.fail("build did not finish in time")

    def test_health_and_auth_flow(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/status").status_code, 401)
        self.login()
        self.assertEqual(self.client.get("/status").status_code, 200)

    def test_upload_build_search_and_download(self) -> None:
        self.login()
        upload = self.client.post(
            "/upload",
            data={"source": "tests", "tags": "alpha,beta"},
            files=[("files", ("notes.txt", b"alpha beta gamma\nsemantic search text", "text/plain"))],
        )
        self.assertEqual(upload.status_code, 200)
        self.assertTrue(upload.json()["changed"])

        status = self.wait_for_build()
        self.assertEqual(status["documents"], 1)
        self.assertGreaterEqual(status["chunks"], 1)
        self.assertTrue(status["build_jobs"][0]["partial_rebuild"])

        documents = self.client.get("/documents").json()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["source"], "tests")
        self.assertEqual(documents[0]["active_version"], 1)

        duplicate = self.client.post(
            "/upload",
            data={"source": "tests", "tags": "alpha"},
            files=[("files", ("copy.txt", b"alpha beta gamma\nsemantic search text", "text/plain"))],
        )
        self.assertEqual(duplicate.status_code, 200)
        self.assertFalse(duplicate.json()["changed"])

        updated = self.client.post(
            "/upload",
            data={"source": "tests", "tags": "alpha,beta"},
            files=[("files", ("notes.txt", b"alpha beta gamma\nsemantic search text\nversion two", "text/plain"))],
        )
        self.assertEqual(updated.status_code, 200)
        self.assertTrue(updated.json()["changed"])
        status = self.wait_for_build()
        self.assertGreaterEqual(status["build_jobs"][0]["reused_chunks"], 0)
        self.assertGreaterEqual(status["build_jobs"][0]["rebuilt_chunks"], 1)

        documents = self.client.get("/documents").json()
        self.assertEqual(documents[0]["active_version"], 2)
        self.assertEqual(len(documents[0]["versions"]), 2)

        search = self.client.get("/search", params={"q": "semantic", "source": "tests", "tags": ["alpha"]})
        self.assertEqual(search.status_code, 200)
        self.assertGreaterEqual(len(search.json()["results"]), 1)

        chunks = self.client.get(f"/document/{documents[0]['id']}/chunks")
        self.assertEqual(chunks.status_code, 200)
        self.assertGreaterEqual(len(chunks.json()), 1)

        download = self.client.get("/download")
        self.assertEqual(download.status_code, 200)
        self.assertGreater(len(download.content), 0)

    def test_zip_upload_and_admin_endpoints(self) -> None:
        self.login()
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("docs/readme.txt", "zip alpha beta")
            archive.writestr("docs/config.json", '{"hello": "world"}')
            archive.writestr("docs/image.bin", b"\x00\x01\x02")

        response = self.client.post(
            "/upload",
            data={"source": "zip-source", "tags": "ziptag"},
            files=[("files", ("bundle.zip", payload.getvalue(), "application/zip"))],
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["changed"])
        self.assertEqual(len(body["uploaded"]), 2)
        self.assertEqual(len(body["skipped"]), 1)

        diagnostics = self.client.get("/admin/diagnostics")
        self.assertEqual(diagnostics.status_code, 200)
        self.assertTrue(diagnostics.json()["model_loaded"])

        clear_builds = self.client.post("/admin/clear-builds")
        self.assertEqual(clear_builds.status_code, 200)

        reset_limits = self.client.post("/admin/reset-rate-limits")
        self.assertEqual(reset_limits.status_code, 200)

        change_password = self.client.post(
            "/admin/password",
            data={"current_password": "change-me", "new_password": "new-secret-123"},
        )
        self.assertEqual(change_password.status_code, 200)

        status_after_password_change = self.client.get("/status")
        self.assertEqual(status_after_password_change.status_code, 401)

        relogin = self.client.post("/auth/login", data={"password": "new-secret-123"})
        self.assertEqual(relogin.status_code, 200)


if __name__ == "__main__":
    unittest.main()
