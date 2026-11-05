"""Verify native PNDM continuation with the real loop and small deterministic nets."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from diffusers import PNDMScheduler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DiFashion-main/DiFashion"))
from models.difashion import DiFashion
from fashion_psp.experiments.run_difashion_fitb_six_seed import BUDGET, SCHEDULE, rank_scores, tree_cpu, resume_config_compatible


class ToyUNet:
    config = SimpleNamespace(sample_size=2)

    def __init__(self):
        self.calls = 0

    def __call__(self, sample, timestep, encoder_hidden_states, **_):
        self.calls += 1
        return (torch.tanh(sample[:, :4] * .07 + timestep.float() * .0001
                           + encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None] * .01),)


def toy_model():
    model = SimpleNamespace(
        args=SimpleNamespace(use_history=True, use_mutual_guidance=True, eta=.1),
        device=torch.device("cpu"), vae_scale_factor=8,
        vae=SimpleNamespace(config=SimpleNamespace(latent_channels=4, scaling_factor=.18215)),
        noise_scheduler=PNDMScheduler(beta_start=.00085, beta_end=.012,
                                     beta_schedule="scaled_linear", skip_prk_steps=True, steps_offset=1),
        text_encoder=lambda ids: (ids.float().unsqueeze(-1),),
        tokenizer=lambda *a, **k: SimpleNamespace(input_ids=torch.zeros(1, 3, dtype=torch.long)),
        fashion_encoder=lambda x: torch.tanh(x * .1),
        unet=ToyUNet(),
        image_processor=SimpleNamespace(postprocess=lambda image, **_: image),
    )
    class Text:
        dtype = torch.float32
        def __call__(self, ids):
            return (ids.float().unsqueeze(-1),)
    model.text_encoder = Text()
    for name in ("prepare_latents", "prepare_extra_step_kwargs", "pred_ori_sample_given_epsilon"):
        setattr(model, name, getattr(DiFashion, name).__get__(model))
    return model


def generate(model, seed, **kwargs):
    return DiFashion.fashion_generation(
        model, uids=torch.tensor([9]), oids=torch.tensor([17]),
        input_ids=torch.arange(12).reshape(1, 4, 3), olists=torch.tensor([[0, 2, 3, 4]]),
        category=torch.tensor([[1, 2, 3, 4]]), history={9: {}},
        outfit_latents=torch.arange(64).reshape(4, 4, 2, 2).float() / 100,
        null_latent=torch.zeros(4, 2, 2),
        category_guidance_scale=12., hist_guidance_scale=4., mutual_guidance_scale=5.,
        generator=torch.Generator().manual_seed(seed), **kwargs,
    )


class SegmentedTests(unittest.TestCase):
    def test_original_and_segmented_are_bitwise_identical_with_serialized_resume(self):
        for seed in (2026, 2027, 8742026):
            direct = toy_model()
            result = generate(direct, seed, output_type="latent", return_dict=False)
            expected = result[0][9][17]["images"][0]
            model = toy_model()
            state = None
            for node, _ in SCHEDULE:
                state = generate(model, seed, output_type="sampling_state", sampling_state=state,
                                 stop_after_step=node)
                self.assertEqual(state["completed_step"], node)
                self.assertEqual(state["scheduler_state"]["counter"], node)
                buffer = io.BytesIO()
                torch.save(tree_cpu(state), buffer)
                buffer.seek(0)
                state = torch.load(buffer, weights_only=False)
                # Another particle uses the shared scheduler between segments.
                generate(model, seed + 99, output_type="sampling_state", stop_after_step=13)
            torch.testing.assert_close(state["latents"][0], expected, rtol=0, atol=0)
            self.assertEqual(direct.unet.calls, 51)
            self.assertEqual(model.unet.calls, 51 + 3 * 13)
            self.assertFalse(torch.equal(state["latents"], state["x_prediction"]))

    def test_online_six_seed_actual_calls_and_terminal_selection(self):
        model = toy_model()
        alive, states = list(range(2026, 2032)), {}
        for node, keep in SCHEDULE:
            scores = {}
            for seed in alive:
                states[seed] = tree_cpu(generate(model, seed, output_type="sampling_state",
                                                 sampling_state=states.get(seed), stop_after_step=node))
                # Only the current node may be scored; use terminal latent at final.
                field = "latents" if node == 51 else "x_prediction"
                scores[seed] = float(states[seed][field].mean())
            alive = rank_scores(scores, keep)
            states = {s: states[s] for s in alive}
        self.assertEqual(model.unet.calls, BUDGET)
        self.assertEqual(BUDGET, 206)
        self.assertEqual(len(alive), 1)

    def test_resume_rejects_changed_schedule_or_config(self):
        model = toy_model()
        state = tree_cpu(generate(model, 2026, output_type="sampling_state", stop_after_step=13))
        with self.assertRaisesRegex(ValueError, "timestep schedule"):
            generate(model, 2026, output_type="sampling_state", sampling_state=state,
                     stop_after_step=39, num_inference_steps=49)
        state["scheduler_config"]["steps_offset"] = 0
        with self.assertRaisesRegex(ValueError, "scheduler config"):
            generate(model, 2026, output_type="sampling_state", sampling_state=state, stop_after_step=39)

    def test_tie_and_nonfinite_score_policy(self):
        self.assertEqual(rank_scores({2029: 1., 2027: 1., 2026: 1.}, 2), [2026, 2027])
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_scores({2026: float("nan")}, 1)

    def test_restart_ignores_private_default_key_order_but_checks_public_parameters(self):
        model = toy_model()
        state = tree_cpu(generate(model, 2026, output_type="sampling_state", stop_after_step=13))
        state["scheduler_config"]["_use_default_values"] = ["timestep_spacing", "prediction_type"]
        model.noise_scheduler.register_to_config(_use_default_values=["prediction_type", "timestep_spacing"])
        resumed = generate(model, 2026, output_type="sampling_state", sampling_state=state, stop_after_step=51)
        direct = generate(toy_model(), 2026, output_type="latent", return_dict=False)[0][9][17]["images"][0]
        torch.testing.assert_close(resumed["latents"][0], direct, rtol=0, atol=0)
        state["scheduler_config"]["prediction_type"] = "v_prediction"
        with self.assertRaisesRegex(ValueError, "scheduler config"):
            generate(model, 2026, output_type="sampling_state", sampling_state=state, stop_after_step=51)

    def test_legacy_resume_accepts_only_known_source_repair_and_identical_settings(self):
        sources = {
            str(Path("fashion_psp/experiments/run_difashion_fitb_six_seed.py")):
                "52483baa31e87ca935084e1102570e3b1df4c0ffff21a2530ff80e78fe4c1d40",
            str(Path("DiFashion-main/DiFashion/models/difashion.py")):
                "49523355241dc24dc2420b0e8fad57d2672249bf85dd7110c0b47bffaff681bd",
        }
        old = {"source_sha256": sources, "schedule": [[13, 4], [39, 2], [51, 1]], "seed_base": 2026}
        current = {**old, "source_sha256": {**sources,
            str(Path("DiFashion-main/DiFashion/models/difashion.py")):
                "ff322d2ad0cc50f0a4d33cf3e1eb96fd6022019f79f00d78fed167ae7eaf77cf"}}
        self.assertTrue(resume_config_compatible(old, current))
        self.assertFalse(resume_config_compatible(old, {**current, "seed_base": 42}))
        self.assertFalse(resume_config_compatible({**old, "source_sha256": {}}, current))


if __name__ == "__main__":
    unittest.main()
