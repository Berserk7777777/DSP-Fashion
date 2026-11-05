"""CPU checks for strict50 storage, seed identity, resume and worker dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fashion_psp.experiments import run_difashion_fitb_16seed_offline_cache as cache
from fashion_psp.experiments.offline_cache_workers import run_workers, task_indices


class OfflineCacheChecks(unittest.TestCase):
    def test_seed_prefixes_and_task_partitions(self):
        for base in (42, 2026):
            for group in range(3):
                for task in (0, 11, 2206):
                    pool = cache.candidate_seeds(task, base, group, 16)
                    self.assertEqual(pool, list(range(base+group*1000000+task*10000,
                                                     base+group*1000000+task*10000+16)))
                    self.assertEqual(cache.candidate_seeds(task, base, group, 8), pool[:8])
        for count in (1, 2, 3, 32):
            actual = [i for worker in range(count) for i in task_indices(2207, worker, count)]
            self.assertEqual(sorted(actual), list(range(2207)))

    def test_exact_image_steps_and_resume_integrity(self):
        self.assertEqual(cache.capture_steps("bon"), [50])
        self.assertEqual(cache.capture_steps("psp"), list(range(1, 51)))
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            images = []
            for step in range(1, 51):
                path = folder / f"step_{step:02d}.png"
                path.write_bytes(f"step{step}".encode())
                images.append(dict(step=step, image=path.name, bytes=path.stat().st_size,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            task = dict(task_index=11, task_id="00011_9248_17906")
            record = dict(config_sha256="frozen", seed=112026, **task,
                          model_forward_calls=50, images=images)
            cache.dump_json(folder / "candidate_metadata.json", record)
            verify = lambda: cache.verify_candidate(folder, task, 112026, "frozen", list(range(1, 51)), True)
            self.assertIsNotNone(verify())
            (folder / "step_50.png").unlink()
            self.assertIsNone(verify())
            with self.assertRaisesRegex(ValueError, "identity/config mismatch"):
                cache.verify_candidate(folder, task, 112027, "frozen", list(range(1, 51)))

    def test_child_arguments_round_trip(self):
        with patch.object(sys, "argv", ["runner", "--method", "psp", "--group-id", "2",
                                        "--output-root", "path with spaces", "--num-particles", "16",
                                        "--concurrent-tasks", "2"]):
            original = cache.parse_args()
        command = cache.child_command(original, 1, 2)
        with patch.object(sys, "argv", ["runner"]+command[3:]):
            worker = cache.parse_args()
        self.assertEqual(worker.worker_index, 1)
        self.assertEqual(worker.worker_count, 2)
        self.assertEqual(worker.concurrent_tasks, 1)
        self.assertEqual(worker.output_root, original.output_root)
        self.assertEqual(worker.num_particles, 16)
        self.assertEqual(worker.group_id, 2)

    def test_worker_failure_terminates_other_children(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            commands = [(0, [sys.executable, "-c", "import time;time.sleep(.2);raise SystemExit(9)"]),
                        (1, [sys.executable, "-c", "import time;time.sleep(60)"])]
            created = []
            original = subprocess.Popen
            def tracked(*args, **kwargs):
                process = original(*args, **kwargs)
                created.append(process)
                return process
            with patch.object(subprocess, "Popen", tracked), self.assertRaisesRegex(RuntimeError, "exit 9"):
                run_workers(commands, root, root)
            self.assertEqual(len(created), 2)
            self.assertTrue(all(p.poll() is not None for p in created))


if __name__ == "__main__":
    unittest.main()
