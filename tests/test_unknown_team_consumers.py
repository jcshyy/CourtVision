import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


class UnknownTeamConsumerTests(unittest.TestCase):
    def test_ball_control_keeps_unknown_assignment_unknown(self):
        fake_cv2 = types.ModuleType("cv2")
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.array = lambda values: values
        module_path = (
            ROOT
            / "backend"
            / "app"
            / "visualization"
            / "team_ball_control_drawer.py"
        )
        spec = importlib.util.spec_from_file_location(
            "courtvision_team_ball_control",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, {"cv2": fake_cv2, "numpy": fake_numpy}):
            spec.loader.exec_module(module)

        result = module.TeamBallControlDrawer().get_team_ball_control(
            [{22: -1}, {22: None}, {22: 2}],
            [22, 22, 22],
        )

        self.assertEqual(result, [-1, -1, 2])

    def test_pass_detection_ignores_unknown_team_transition(self):
        module_path = (
            ROOT
            / "backend"
            / "app"
            / "analytics"
            / "pass_interception.py"
        )
        spec = importlib.util.spec_from_file_location(
            "courtvision_pass_interception",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        detector = module.PassInterceptionDetector()
        acquisitions = [10, 22]
        assignments = [{10: 1}, {22: -1}]

        self.assertEqual(
            detector.detect_passes(acquisitions, assignments),
            [-1, -1],
        )
        self.assertEqual(
            detector.detect_interceptions(acquisitions, assignments),
            [-1, -1],
        )


if __name__ == "__main__":
    unittest.main()
