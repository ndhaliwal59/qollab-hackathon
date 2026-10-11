"""Acceptance checks for the live service and measured Qiskit path."""

import json
from datetime import datetime
import unittest

import numpy as np
from qiskit.providers.basic_provider import BasicSimulator

from signal_rescue.live import _decode_counts, _make_observation, _normalize_spec, dispatch


def spec(**overrides):
    return {"request_id": "live-test-1", "scenario_index": 0, "seed": 7,
            "users": 8, "snr_db": 12, "shots_per_round": 64, **overrides}


def invoke(action, payload=None, backend=None):
    return json.loads(dispatch(backend, action, json.dumps(payload or {})))


class FailedBackend:
    name = "broken_backend"

    def run(self, *_args, **_kwargs):
        raise OSError("simulated circuit failure")


class RecordingBackend:
    name = "basic_simulator"

    def __init__(self):
        self.calls = []
        self.simulator = BasicSimulator()

    def run(self, circuit, **options):
        self.calls.append((circuit, options))
        return self.simulator.run(circuit, **options)


class LiveExperimentTests(unittest.TestCase):
    def test_preview_exports_every_actual_user_to_antenna_connection(self):
        for users in (8, 12):
            with self.subTest(users=users):
                request = spec(users=users)
                backend = RecordingBackend()
                data = invoke("preview", request, backend)["result"]
                observation = _make_observation(_normalize_spec(request))
                powers = np.asarray(data["recipe"]["channel_link_powers"])
                self.assertEqual(powers.shape, (8, users))
                np.testing.assert_allclose(powers, np.abs(observation.H) ** 2)
                # The dataset uses complex64; its original column reduction
                # rounds in float32 before JSON promotes each link to float64.
                np.testing.assert_allclose(powers.sum(axis=0), data["recipe"]["channel_column_powers"], rtol=5e-7)
                self.assertEqual(data["truth_bits"], data["recipe"]["truth_bits"])
                self.assertEqual(backend.calls, [])

    def test_repeated_requests_execute_new_circuits_with_distinct_receipts(self):
        backend = RecordingBackend()
        first = invoke("run", spec(request_id="first-click"), backend)["result"]
        second = invoke("run", spec(request_id="second-click"), backend)["result"]
        self.assertEqual(len(backend.calls), 4)
        self.assertIsNot(backend.calls[0][0], backend.calls[2][0])
        self.assertEqual(first["qaoa_rounds"], second["qaoa_rounds"])
        self.assertEqual(first["execution"]["request_id"], "first-click")
        self.assertEqual(second["execution"]["request_id"], "second-click")
        for data in (first, second):
            receipt = data["execution"]
            self.assertTrue(receipt["sampling_seeded"])
            self.assertEqual(receipt["reproducible_seed"], 7)
            self.assertLessEqual(datetime.fromisoformat(receipt["started_at"]),
                                 datetime.fromisoformat(receipt["completed_at"]))

    def test_bootstrap_and_invalid_control_values(self):
        loaded = invoke("bootstrap")
        self.assertEqual(loaded["status"], "ready")
        self.assertEqual(loaded["supported"]["users"], [8, 12])
        self.assertEqual(loaded["dataset"]["sha256"], loaded["dataset"]["provenance"]["sha256"])
        self.assertTrue(loaded["model"]["freeze_sha256"])
        invalid = invoke("preview", spec(snr_db=10))
        self.assertEqual(invalid["status"], "error")
        self.assertIn("SNR", invalid["error"])

    def test_little_endian_measurements_keep_qubit_zero_as_first_block_bit(self):
        h = np.zeros(8)
        J = np.zeros((8, 8))
        candidates, histogram = _decode_counts({"00000101": 2, "00000010": 1}, 8, 3, h, J)
        self.assertEqual(candidates.shape, (3, 8))
        np.testing.assert_array_equal(candidates[0], [-1, 1, -1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(candidates[2], [1, -1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(histogram[0]["bits"], "00000101")
        with self.assertRaisesRegex(RuntimeError, "returned 2 samples"):
            _decode_counts({"00000101": 2}, 8, 3, h, J)

    def test_actual_qiskit_runs_execute_two_matched_adaptive_rounds(self):
        for users, shots in [(8, 64), (12, 128)]:
            with self.subTest(users=users, shots=shots):
                result = invoke("run", spec(users=users, shots_per_round=shots), BasicSimulator())
                self.assertEqual(result["status"], "ok", result)
                data = result["result"]
                self.assertEqual(data["execution"]["backend"], "BasicSimulator")
                self.assertEqual(data["execution"]["status"], "complete")
                self.assertEqual(data["execution"]["circuit_invocations"], 2)
                self.assertEqual(data["execution"]["shots"], 2 * shots)
                self.assertTrue(data["comparison"]["budgets_matched"])
                self.assertEqual(data["comparison"]["uniform_total_candidates"], 2 * shots)
                self.assertEqual(len(data["qaoa_rounds"]), 2)
                self.assertTrue(all(sum(row["count"] for row in counts) == shots
                                    for counts in data["qaoa_rounds"]))
                self.assertEqual([row["id"] for row in data["methods"]],
                                 ["mmse", "mmse_sic", "classical", "uniform", "qaoa", "ml_reference"])
                self.assertEqual(data["model"]["config"]["k"], 8)
                self.assertEqual(data["model"]["config"]["p"], 1)
                self.assertEqual(data["recipe"]["users"], users)
                self.assertEqual(len(data["recipe"]["user_uncertainty"]), users)
                self.assertTrue(all(0.0 <= value <= 1.0 for value in data["recipe"]["user_uncertainty"]))
                self.assertTrue(all(len(row["bits"]) == users for row in data["methods"]))
                trace = data["qaoa_replay"]
                np.testing.assert_array_equal(trace["rounds"][0]["incumbent_before"], trace["classical_incumbent"])
                np.testing.assert_array_equal(trace["rounds"][0]["incumbent_after"], trace["rounds"][1]["incumbent_before"])
                np.testing.assert_array_equal(trace["rounds"][-1]["incumbent_after"], trace["final_bits"])

    def test_identical_experiment_recipe_is_reproducible(self):
        first = invoke("run", spec(), BasicSimulator())["result"]
        second = invoke("run", spec(), BasicSimulator())["result"]
        self.assertEqual(first["qaoa_rounds"], second["qaoa_rounds"])
        self.assertEqual([(row["id"], row["bits"]) for row in first["methods"]], [
            (row["id"], row["bits"]) for row in second["methods"]
        ])

    def test_failed_circuits_preserve_classical_incumbent_and_are_labeled_partial(self):
        result = invoke("run", spec(), FailedBackend())
        self.assertEqual(result["status"], "ok", result)
        data = result["result"]
        self.assertEqual(data["execution"]["status"], "partial")
        self.assertEqual(data["execution"]["fallbacks"], 2)
        self.assertIn("failed", data["execution"]["warning"])
        self.assertEqual(data["methods"][2]["bits"], data["methods"][4]["bits"])
        self.assertEqual(data["execution"]["shots"], 0)
