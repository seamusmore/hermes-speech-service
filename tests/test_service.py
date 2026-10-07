"""HTTP contract and namespace isolation tests, without loading model weights."""
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hermes_speech_service.app import create_app
from hermes_speech_service.stt import app as stt
from hermes_speech_service.tts import app as tts


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(token="test-secret"))
        self.headers = {"authorization": "Bearer test-secret"}

    def tearDown(self):
        self.client.close()

    def test_liveness_is_separate_from_model_readiness(self):
        self.assertEqual(self.client.get("/health").json()["service"], "hermes-speech-service")
        with patch.object(stt, "_get_engine", side_effect=AssertionError("loaded STT")), \
             patch.object(tts, "_get_engine", side_effect=AssertionError("loaded TTS")):
            response = self.client.get("/ready", headers=self.headers)
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json()["ready"])
            for path in ("/stt/ready", "/tts/ready"):
                self.assertEqual(self.client.get(path, headers=self.headers).status_code, 503)

    def test_authentication_applies_to_both_components(self):
        for path in ("/capabilities", "/ready", "/stt/openapi.json", "/tts/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 401)
            self.assertEqual(self.client.get(path, headers={"authorization": "Bearer wrong"}).status_code, 401)

    def test_existing_audio_and_cancellation_contracts_are_preserved(self):
        stt_routes = self.client.get("/stt/openapi.json", headers=self.headers).json()["paths"]
        tts_routes = self.client.get("/tts/openapi.json", headers=self.headers).json()["paths"]
        self.assertIn("/transcribe", stt_routes)
        for path in ("/synthesize", "/synthesize-stream", "/cancel/{request_id}"):
            self.assertIn(path, tts_routes)
        self.assertNotIn("/transcribe", tts_routes)
        self.assertNotIn("/synthesize", stt_routes)

    def test_engine_namespaces_are_independent(self):
        from hermes_speech_service.stt import engines as se
        from hermes_speech_service.tts import engines as te
        self.assertIsNot(se._ENGINE_CLASSES, te._ENGINE_CLASSES)
        self.assertIsNot(stt._runtime, tts._runtime)

    def test_local_service_declares_its_capabilities(self):
        caps = self.client.get("/capabilities", headers=self.headers).json()
        self.assertTrue(caps["stt"] and caps["tts"] and caps["tts_streaming"])
        self.assertFalse(caps["realtime"])

    def test_unauthenticated_remote_clients_are_rejected(self):
        with TestClient(create_app(token="", components={}), client=("203.0.113.1", 80)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/capabilities").status_code, 403)

    def test_component_lifecycles_and_warmup_failure(self):
        events = []
        def component(name, failing=False):
            @asynccontextmanager
            async def lifespan(app):
                events.append("start:" + name)
                try:
                    yield
                finally:
                    events.append("stop:" + name)
            def warmup():
                if failing:
                    raise RuntimeError("private backend detail")
                return {"ready": True}
            return SimpleNamespace(app=FastAPI(), lifespan=lifespan, warmup=warmup,
                                   readiness_state=lambda: {"ready": True})
        components = {"stt": component("stt"), "tts": component("tts", True)}
        with TestClient(create_app(token="test-secret", components=components)) as client:
            self.assertEqual(events, ["start:stt", "start:tts"])
            response = client.post("/warmup", headers=self.headers).json()
            self.assertFalse(response["ready"])
            self.assertEqual(response["components"]["tts"]["error"], "RuntimeError")
            self.assertNotIn("private", str(response))
        self.assertEqual(events[-2:], ["stop:tts", "stop:stt"])


if __name__ == "__main__":
    unittest.main()
