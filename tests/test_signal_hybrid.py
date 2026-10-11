"""Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -v"""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from signal_rescue.channels import (ChannelBank, ExperimentBank, DOWNLOAD_CAP,
                                    complex_normal, model_channel, split_groups)
from signal_rescue.core import (cost, energies, exact_ml_reference, ising, mmse,
                                ordered_sic, quadratic, refine, select_block, spin_table)
from signal_rescue.dataset import acquire, synthesize_ula
from signal_rescue.hybrid import detect, prepare
from signal_rescue.metrics import paired_bootstrap
from signal_rescue.samplers import Config, SampleResult, objective_value, probabilities, sample
from signal_rescue.training import candidate_configs

ROOT = Path(__file__).resolve().parents[1]


class ComplexMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.H = complex_normal(rng, (4, 5))
        self.y = complex_normal(rng, 4)

    def test_all_ising_residual_differences_with_fixed_bits(self):
        G, b = quadratic(self.H, self.y)
        incumbent = np.array([1., -1., 1., 1., -1.])
        for idx in (np.array([3, 0, 2]), np.arange(5)):
            h, J = ising(G, b, incumbent, idx)
            spins = spin_table(len(idx))
            expanded = np.tile(incumbent, (len(spins), 1))
            expanded[:, idx] = spins
            residuals = np.array([cost(self.H, self.y, x) for x in expanded])
            e = energies(spins, h, J)
            np.testing.assert_allclose(e - e[0], residuals - residuals[0], atol=1e-11)

    def test_complex_real_equivalent_mmse(self):
        hard, soft, posterior = mmse(self.H, self.y, .7)
        Hr = np.concatenate((self.H.real, self.H.imag))
        yr = np.concatenate((self.y.real, self.y.imag))
        hard_r, soft_r, posterior_r = mmse(Hr, yr, .35)
        np.testing.assert_array_equal(hard, hard_r)
        np.testing.assert_allclose(soft, soft_r, atol=1e-12)
        np.testing.assert_allclose(posterior, posterior_r, atol=1e-12)

    def test_ml_matches_direct_exhaustive_residual(self):
        table = spin_table(5)
        best = table[np.argmin([cost(self.H, self.y, x) for x in table])]
        np.testing.assert_array_equal(exact_ml_reference(self.H, self.y), best)

    def test_mask_order_is_little_endian_spin_minus_one(self):
        np.testing.assert_array_equal(spin_table(3)[1], [-1, 1, 1])
        np.testing.assert_array_equal(spin_table(3)[4], [1, 1, -1])

    def test_refinement_never_increases_cost(self):
        G, b = quadratic(self.H, self.y)
        candidates = spin_table(5)
        for pair in (False, True):
            final, _ = refine(G, b, candidates, steps=8, pair=pair)
            for x, original in zip(final, candidates):
                self.assertLessEqual(cost(self.H, self.y, x), cost(self.H, self.y, original) + 1e-9)

    def test_refinement_step_matches_direct_single_pair_search(self):
        G, b = quadratic(self.H, self.y)
        for x in spin_table(5):
            trials = [x.copy()]
            for i in range(5):
                trial = x.copy(); trial[i] *= -1; trials.append(trial)
                for j in range(i + 1, 5):
                    trial = x.copy(); trial[[i, j]] *= -1; trials.append(trial)
            expected = min(cost(self.H, self.y, trial) for trial in trials)
            final, _ = refine(G, b, x, steps=1)
            self.assertAlmostEqual(cost(self.H, self.y, final[0]), expected, places=10)

    def test_noiseless_ordered_sic(self):
        H = np.eye(4, dtype=complex) * (1 + 1j)
        truth = np.array([1., -1., -1., 1.])
        np.testing.assert_array_equal(ordered_sic(H, H @ truth, 0), truth)

    def test_imaginary_channel_contains_real_detection_information(self):
        H = 1j * np.eye(4)
        truth = np.array([1., -1., 1., -1.])
        np.testing.assert_array_equal(mmse(H, H @ truth, .01)[0], truth)
        self.assertFalse(np.array_equal(mmse(H.real, (H @ truth).real, .005)[0], truth))

    def test_lower_residual_is_not_a_ber_guarantee(self):
        H, y = np.array([[1j]]), np.array([-.2j])
        correct, wrong = np.array([1.]), np.array([-1.])
        self.assertLess(cost(H, y, wrong), cost(H, y, correct))

    def test_malformed_inputs(self):
        for H, y, variance in ((np.ones((2, 3)), np.ones(3), 1),
                               (np.ones((2, 3)), np.ones(2), -1),
                               (np.array([[np.nan]]), np.ones(1), 1)):
            with self.assertRaises(ValueError):
                mmse(H, y, variance)
        with self.assertRaises(ValueError):
            ising(*quadratic(self.H, self.y), np.ones(5), [1, 1])


class ChannelTests(unittest.TestCase):
    def test_training_examples_cover_each_family_without_test_calls(self):
        from signal_rescue.training import examples
        bank = ExperimentBank(7, dataset="controlled")
        with patch.object(bank, "instance", wraps=bank.instance) as called:
            rows = examples([bank], "train", ("rayleigh", "correlated", "rician"), 24)
        self.assertEqual(len(rows), 24)
        self.assertEqual({call.args[0] for call in called.call_args_list}, {"train"})
        self.assertEqual({call.args[1] for call in called.call_args_list}, {"rayleigh", "correlated", "rician"})

    def test_split_disjoint_and_repeatable(self):
        split = split_groups(range(100), 7)
        self.assertEqual([len(split[x]) for x in ("train", "validation", "test")], [60, 20, 20])
        self.assertEqual(split, split_groups(range(100), 7))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["validation"]) & set(split["test"]))

    def test_models_are_complex_and_retain_power_imbalance(self):
        for family in ("rayleigh", "correlated", "rician"):
            H = model_channel(family, 8, 12, 3)
            self.assertTrue(np.iscomplexobj(H))
            self.assertAlmostEqual(np.vdot(H, H).real, 12, places=10)
            self.assertGreater(np.std(np.sum(np.abs(H) ** 2, axis=0)), .1)
            np.testing.assert_array_equal(H, model_channel(family, 8, 12, 3))

    def test_circular_complex_noise_variance(self):
        noise = complex_normal(np.random.default_rng(7), 200_000) * np.sqrt(.6)
        self.assertAlmostEqual(float(np.mean(np.abs(noise) ** 2)), .6, delta=.006)
        self.assertAlmostEqual(float(np.var(noise.real)), .3, delta=.004)
        self.assertAlmostEqual(float(np.var(noise.imag)), .3, delta=.004)
        self.assertAlmostEqual(float(np.mean(noise ** 2).real), 0, delta=.005)

    def test_snr_and_csi_share_observation_not_estimate(self):
        bank = ChannelBank(seed=7)
        inst = bank.instance("test", "rayleigh", "square", 12, 2)
        imperfect = bank.instance("test", "rayleigh", "square", 12, 2, csi="-10")
        expected = np.vdot(inst.H, inst.H).real / 8 / 10 ** 1.2
        self.assertAlmostEqual(inst.sigma2, expected)
        np.testing.assert_array_equal(inst.y, imperfect.y)
        np.testing.assert_array_equal(inst.truth, imperfect.truth)
        ratio = np.vdot(imperfect.H - inst.H, imperfect.H - inst.H).real / np.vdot(inst.H, inst.H).real
        self.assertAlmostEqual(ratio, .1)

    def test_all_snr_repeats_stay_in_same_group(self):
        bank = ChannelBank()
        a = bank.instance("test", "rician", "square", 8, 3)
        b = bank.instance("test", "rician", "square", 20, 3)
        self.assertEqual(a.group, b.group)
        np.testing.assert_array_equal(a.H, b.H)

    def test_deepmimo_missing_rejected(self):
        with self.assertRaises(ValueError):
            ChannelBank(dataset="deepmimo", path="/nonexistent/channel.npz")

    def test_dataset_cap_is_100mb(self):
        self.assertEqual(DOWNLOAD_CAP, 100_000_000)

    def test_oversized_download_not_started(self):
        response = BytesIO(json.dumps({"fileSize": DOWNLOAD_CAP + 1}).encode())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "large.zip"
            with patch("urllib.request.urlopen", return_value=response) as opened:
                with self.assertRaises(ValueError):
                    acquire("asu_campus_3p5", target)
                self.assertEqual(opened.call_count, 1)
                self.assertFalse(target.exists())

    def test_specialized_ula_single_path(self):
        # 0 dBW, 90-degree ray phase, polar=90, azimuth=90 -> alternating j/sqrt(512).
        H = synthesize_ula(np.array([[0.]]), np.array([[90.]]), np.array([[90.]]),
                           np.array([[90.]]), np.array([[0.]]))
        expected = 1j * (-1.) ** np.arange(8) / np.sqrt(512)
        np.testing.assert_allclose(H[0], expected, atol=1e-8)

    @unittest.skipUnless((ROOT / "data/deepmimo/asu_slice.npz").exists(), "optional bounded DeepMIMO slice absent")
    def test_real_spatial_split_stable_across_noise_seeds(self):
        a = ChannelBank(seed=7, dataset="deepmimo", path=ROOT / "data/deepmimo/asu_slice.npz")
        b = ChannelBank(seed=8, dataset="deepmimo", path=ROOT / "data/deepmimo/asu_slice.npz")
        self.assertEqual(a.splits, b.splits)
        inst_a = a.instance("test", "deepmimo", "overloaded", 12, 0)
        inst_b = b.instance("test", "deepmimo", "overloaded", 12, 0)
        np.testing.assert_array_equal(inst_a.H, inst_b.H)
        self.assertEqual(inst_a.group, inst_b.group)
        self.assertFalse(np.array_equal(inst_a.y, inst_b.y))

    @unittest.skipUnless((ROOT / "data/deepmimo/asu_slice.npz").exists(), "optional bounded DeepMIMO slice absent")
    def test_real_positions_are_disjoint_with_guard_gaps(self):
        with np.load(ROOT / "data/deepmimo/asu_slice.npz", allow_pickle=False) as data:
            ids, groups, positions = data["user_ids"], data["group_ids"], data["user_positions"]
        self.assertEqual(len(np.unique(ids)), ids.size)
        all_positions = positions.reshape(-1, 3)[:, :2]
        all_groups = np.repeat(groups, 12)
        for group in np.unique(groups):
            inside, outside = all_positions[all_groups == group], all_positions[all_groups != group]
            distances = np.linalg.norm(inside[:, None] - outside[None, :], axis=2)
            self.assertGreaterEqual(float(distances.min()), 16 - 1e-3)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.h = np.array([.3, -.7, .1])
        self.J = np.array([[0, .2, -.4], [.2, 0, .5], [-.4, .5, 0]])
        self.params = np.array([.4, .8, .3, .6])
        self.q = np.array([.1, .7, .25])

    def test_qiskit_probability_equivalence_standard_and_warm(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        scale = max(np.max(np.abs(self.h)), np.max(np.abs(self.J)))
        theta = 2 * np.arcsin(np.sqrt(self.q))
        for warm in (False, True):
            qc = QuantumCircuit(3)
            for i in range(3):
                qc.ry(theta[i], i) if warm else qc.h(i)
            for layer in range(2):
                for i in range(3):
                    qc.rz(2 * self.params[layer] * self.h[i] / scale, i)
                    for j in range(i + 1, 3):
                        qc.rzz(2 * self.params[layer] * self.J[i, j] / scale, i, j)
                for i in range(3):
                    if warm:
                        qc.ry(-theta[i], i); qc.rz(-2 * self.params[2 + layer], i); qc.ry(theta[i], i)
                    else:
                        qc.rx(2 * self.params[2 + layer], i)
            expected = Statevector.from_instruction(qc).probabilities()
            actual = probabilities(self.h, self.J, self.params, p=2, warm_prob=self.q if warm else None)
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_probability_normalization_zero_cost(self):
        probs = probabilities(np.zeros(3), np.zeros((3, 3)), self.params, p=2, warm_prob=self.q)
        self.assertAlmostEqual(probs.sum(), 1)
        bits = (1 - spin_table(3)) / 2
        expected = np.prod(np.where(bits, self.q, 1 - self.q), axis=1)
        np.testing.assert_allclose(probs, expected, atol=1e-12)

    def test_finite_shot_counts_and_repeatability(self):
        for kind in ("qaoa", "uniform", "biased", "sa"):
            config = Config(k=4)
            a = sample(kind, self.h, self.J, self.q, config, self.params, 9)
            b = sample(kind, self.h, self.J, self.q, config, self.params, 9)
            self.assertEqual(a.candidates.shape, (64, 3))
            np.testing.assert_array_equal(a.candidates, b.candidates)
            self.assertEqual(a.telemetry["shots"], 64 if kind == "qaoa" else 0)

    def test_cvar_fractional_boundary(self):
        self.assertAlmostEqual(objective_value(np.array([1., 4., 10.]), np.array([.1, .2, .7]),
                                               objective="cvar", alpha=.25), 2.8)

    def test_budget_validation(self):
        for kwargs in ({"k": 10}, {"rounds": 3}, {"shots": 1024}, {"p": 4},
                       {"refine_steps": 33}, {"temperature": np.nan}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        configs = candidate_configs()
        self.assertEqual(len(configs), 24)
        self.assertEqual({c.p for c in configs}, {1, 2, 3})
        self.assertEqual({c.k for c in configs}, {4, 6, 8})
        self.assertEqual({c.warm for c in configs}, {False, True})


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.instance = ChannelBank().instance("test", "correlated", "overloaded", 8, 3)
        self.config = Config(k=6)
        self.params = np.array([.4, .7, .3, .6])

    def test_matched_candidate_and_refinement_budgets(self):
        inst = self.instance
        for kind in ("qaoa", "uniform", "biased", "sa"):
            result = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params, kind=kind)
            self.assertEqual(result.telemetry["candidates"], 128)
            self.assertEqual(result.telemetry["candidate_refinements"], 8)
            self.assertTrue(result.telemetry["residual_nonincreasing"])
            self.assertEqual(result.telemetry["fallbacks"], 0)

    def test_failure_falls_back_to_incumbent(self):
        inst = self.instance
        initial = prepare(inst.H, inst.y, inst.sigma2, self.config)[-1]
        def fail(*args):
            raise RuntimeError("sampler unavailable")
        result = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params, sampler_fn=fail)
        np.testing.assert_array_equal(result.x, initial)
        self.assertEqual(result.telemetry["fallbacks"], 2)

    def test_malformed_candidates_fall_back(self):
        inst = self.instance
        def malformed(*args):
            return SampleResult(np.zeros((64, 6)), {})
        result = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params, sampler_fn=malformed)
        self.assertEqual(result.telemetry["fallbacks"], 2)
        self.assertTrue(result.telemetry["residual_nonincreasing"])

    def test_rotating_block_can_include_previously_excluded_bits(self):
        G = np.eye(12)
        soft, posterior = np.linspace(.01, 1, 12), np.ones(12)
        a = select_block(soft, posterior, G, 4)
        b = select_block(soft, posterior, G, 4, round_index=1, previous=a)
        self.assertTrue(set(b) - set(a))

    def test_runtime_never_calls_exact_ml_or_optimizes(self):
        inst = self.instance
        with patch("signal_rescue.core.exact_ml_reference", side_effect=AssertionError("oracle called")), \
             patch("scipy.optimize.minimize", side_effect=AssertionError("optimizer called")):
            result = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params)
        self.assertEqual(result.telemetry["circuit_invocations"], 2)

    def test_hybrid_repeatable(self):
        inst = self.instance
        a = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params, seed=5)
        b = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params, seed=5)
        np.testing.assert_array_equal(a.x, b.x)

    def test_duplicate_samples_use_incumbent_padding_and_keep_budget(self):
        inst = self.instance
        def duplicate(*args):
            return SampleResult(np.ones((64, 6)), {"shots": 64, "circuit_invocations": 1,
                                "candidates": 64, "sa_spin_proposals": 0, "seconds": 0., "unique_candidates": 1})
        result = detect(inst.H, inst.y, inst.sigma2, config=self.config, params=self.params,
                        sampler_fn=duplicate, replay=True)
        self.assertEqual(result.telemetry["fallbacks"], 0)
        self.assertEqual(result.telemetry["candidate_refinements"], 8)
        self.assertEqual(len(result.replay["rounds"][0]["sampled_candidates"]), 1)
        self.assertEqual(len(result.replay["rounds"][0]["refined_candidates"]), 4)


class MetricTests(unittest.TestCase):
    def test_confirmation_test_rejects_unrequested_rerun(self):
        from signal_rescue.benchmark import run_test
        import hashlib
        frozen = {"params": []}
        frozen["freeze_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "test_started.json").write_text("{}")
            with self.assertRaises(ValueError):
                run_test(path, [], [], frozen, {})

    def test_old_axis_remapping_is_not_scored_by_default(self):
        from signal_rescue.run_benchmark import run
        from signal_rescue.mmobile_data import SOURCE_METADATA
        self.assertFalse(SOURCE_METADATA["scored_mimo_eligible"])
        with self.assertRaises(RuntimeError):
            run()

    def test_zero_difference_ci_is_zero_not_evidence_of_win(self):
        errors = np.array([[1., 1.], [2., 2.], [0., 0.], [1., 1.]])
        result = paired_bootstrap(errors, np.full(4, 8), np.array(["a", "a", "b", "c"]), 0, 1)
        self.assertEqual(result["ber_improvement_ci95"], [0., 0.])
        self.assertEqual(result["groups"], 3)

    def test_freeze_integrity(self):
        from signal_rescue.benchmark import verify_freeze
        import hashlib
        payload = {"params": [1, 2], "splits": []}
        payload["freeze_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        verify_freeze(payload)
        payload["params"][0] = 3
        with self.assertRaises(ValueError):
            verify_freeze(payload)


if __name__ == "__main__":
    unittest.main()
