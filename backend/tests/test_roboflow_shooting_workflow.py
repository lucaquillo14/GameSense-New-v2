import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.roboflow_shooting_workflow import (
    WORKFLOW_OUTPUT_KEYS,
    parse_shooting_technique_output,
    run_soccer_shooting_technique_analyzer,
)

FIXTURE_IMAGE = BACKEND_ROOT / "tests" / "fixtures" / "soccer_kick_sample.jpg"
WORKFLOW_FIXTURE = BACKEND_ROOT / "tests" / "fixtures" / "workflow_shooting_response.json"
ENV_PATH = BACKEND_ROOT / ".env"


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


class RoboflowShootingWorkflowTests(unittest.TestCase):
    def test_parse_workflow_output_uses_dynamic_keys(self):
        raw = json.loads(WORKFLOW_FIXTURE.read_text(encoding="utf-8"))
        with self.subTest("fixture covers declared workflow outputs"):
            self.assertTrue(WORKFLOW_OUTPUT_KEYS.issubset(raw.keys()))

        result = parse_shooting_technique_output(raw)
        self.assertEqual(result.technique_score, 6.2)
        self.assertEqual(result.phase, "approach")
        self.assertGreaterEqual(len(result.feedback), 1)
        self.assertIn("knee_angle_deg", result.metrics)
        self.assertIsNotNone(result.detections)
        self.assertIn("person", result.detections.class_names)
        self.assertEqual(set(result.raw_output_keys), set(raw.keys()))

    def test_run_function_returns_expected_keys_from_fixture_image(self):
        raw = json.loads(WORKFLOW_FIXTURE.read_text(encoding="utf-8"))
        self.assertTrue(FIXTURE_IMAGE.exists(), "Missing tests/fixtures/soccer_kick_sample.jpg")

        class _FakeClient:
            def run_workflow(self, **kwargs):
                self.last_images = kwargs.get("images")
                return [raw]

        fake_client = _FakeClient()
        fixture_result = parse_shooting_technique_output(raw)
        with patch(
            "app.services.roboflow_shooting_workflow._get_serverless_client",
            return_value=fake_client,
        ), patch(
            "app.services.shooting_technique_metrics.analyze_frame_with_local_detections",
            return_value=(
                {
                    "technique_score": fixture_result.technique_score,
                    "shot_power_kmh": fixture_result.shot_power_kmh,
                    "phase": fixture_result.phase,
                    "feedback": fixture_result.feedback,
                    "metrics": fixture_result.metrics,
                },
                fixture_result.detections,
                None,
            ),
        ):
            result = run_soccer_shooting_technique_analyzer(
                FIXTURE_IMAGE,
                api_key="test-key",
                max_retries=0,
            )

        self.assertIsNotNone(fake_client.last_images)
        missing = WORKFLOW_OUTPUT_KEYS - set(result.raw_output_keys)
        self.assertFalse(missing, f"Missing keys: {sorted(missing)}")
        self.assertEqual(result.technique_score, 6.2)

    @unittest.skipUnless(
        os.environ.get("ROBOFLOW_API_KEY", "").strip()
        and os.environ.get("RUN_LIVE_WORKFLOW_TESTS", "").strip(),
        "Set ROBOFLOW_API_KEY and RUN_LIVE_WORKFLOW_TESTS=1 for the live smoke test "
        "(legacy workflow — the app now uses the RF-DETR pipeline)",
    )
    def test_run_soccer_shooting_technique_analyzer_smoke(self):
        self.assertTrue(FIXTURE_IMAGE.exists(), "Missing tests/fixtures/soccer_kick_sample.jpg")
        try:
            result = run_soccer_shooting_technique_analyzer(FIXTURE_IMAGE, max_retries=0)
        except Exception as exc:
            self.skipTest(f"Live workflow unavailable: {exc}")

        missing = WORKFLOW_OUTPUT_KEYS - set(result.raw_output_keys)
        self.assertFalse(
            missing,
            f"Workflow response missing expected keys: {sorted(missing)}",
        )
        self.assertIsInstance(result.feedback, list)


if __name__ == "__main__":
    unittest.main()
