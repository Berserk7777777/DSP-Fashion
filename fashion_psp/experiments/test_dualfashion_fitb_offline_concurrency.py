"""CPU-only task dispatch integration checks; no generation or model loading."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[2] / "DualFashion-main/run_ifashion_offline_cache.py"
NAMES = {"parse_args", "task_dir", "task_complete", "task_indices", "worker_command", "run_task_workers"}
namespace = dict(argparse=argparse, Path=Path, json=json, os=os, subprocess=subprocess,
                 sys=sys, time=time, __file__=str(RUNNER))
tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
exec(compile(ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in NAMES],
                        type_ignores=[]), str(RUNNER), "exec"), namespace)

WORKER = '''import json, pathlib, sys, time
root, index, count, total, base, fail = sys.argv[1:]
root=pathlib.Path(root)
index,count,total,base=map(int,(index,count,total,base))
if fail == "yes":
    if index==0:
        time.sleep(.15)
        sys.exit(9)
    time.sleep(60)
else:
    for task in range(index,total,count):
        target=root/f"psp_offline_group0_task{task}_seed{base}"
        if (target/'metadata.json').is_file():
            continue
        target.mkdir()
        seeds=[base+task*10000+slot for slot in range(16)]
        image=target/'placeholder.png'
        image.touch()
        payload=dict(task_index=task,candidate_seeds=seeds,offline_candidate_cache=dict(
            enabled=True,steps=[28],candidate_count=16,
            images=[dict(step=28,seed=s,image_path=str(image)) for s in seeds]))
        (target/'metadata.json').write_text(json.dumps(payload))
        time.sleep(.01)
'''


class TaskConcurrencyChecks(unittest.TestCase):
    def test_task_coverage_for_uneven_and_small_pools(self):
        for total in (1, 7, 200, 2207):
            for count in (1, 2, 3, 32):
                assigned = [task for index in range(count)
                            for task in namespace["task_indices"](total, index, count)]
                self.assertEqual(sorted(assigned), list(range(total)))

    def test_worker_arguments_round_trip(self):
        argv = ["runner", "--data-root", "data path", "--processed-info-root", "info path",
                "--image-root", "image path", "--output-root", "output path", "--group", "2",
                "--seed", "2002026", "--num-particles", "16", "--concurrent-tasks", "3",
                "--cpu-offload", "--deterministic-context", "--offline-cache-steps", "28"]
        with patch.object(sys, "argv", argv):
            original = namespace["parse_args"]()
        command = namespace["worker_command"](original, 1, 3)
        with patch.object(sys, "argv", command[1:]):
            worker = namespace["parse_args"]()
        self.assertEqual(worker.concurrent_tasks, 1)
        self.assertEqual(worker.task_worker_index, 1)
        self.assertEqual(worker.task_worker_count, 3)
        for key, value in vars(original).items():
            if key not in {"concurrent_tasks", "task_worker_index", "task_worker_count"}:
                self.assertEqual(getattr(worker, key), value, key)

    def test_real_processes_resume_when_concurrency_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            helper = root / "fake_worker.py"
            helper.write_text(WORKER, encoding="utf-8")
            args = argparse.Namespace(output_root=root, concurrent_tasks=3, group=0, seed=2026,
                                      num_particles=16)
            command = lambda a, i, n: [sys.executable, str(helper), str(root), str(i), str(n),
                                       "17", str(a.seed), "no"]
            with patch.dict(namespace, worker_command=command):
                namespace["run_task_workers"](args, 17, [28], 0)
            saved = {}
            for task in range(17):
                target = namespace["task_dir"](root, 0, task, 2026)
                self.assertTrue(namespace["task_complete"](target, [28], 16))
                metadata = target / "metadata.json"
                saved[task] = metadata.read_bytes()
                self.assertEqual(json.loads(saved[task])["candidate_seeds"],
                                 [2026+task*10000+slot for slot in range(16)])
            # One incomplete task must be recovered under a different assignment;
            # every other task keeps exactly the same frozen metadata bytes.
            unfinished = namespace["task_dir"](root, 0, 11, 2026)
            for path in unfinished.iterdir():
                path.unlink()
            unfinished.rmdir()
            args.concurrent_tasks = 2
            with patch.dict(namespace, worker_command=command):
                namespace["run_task_workers"](args, 17, [28], 0)
            for task in range(17):
                metadata = namespace["task_dir"](root, 0, task, 2026) / "metadata.json"
                self.assertEqual(metadata.read_bytes(), saved[task])
            with patch.dict(namespace, worker_command=command), patch.object(subprocess, "Popen") as launch:
                namespace["run_task_workers"](args, 17, [28], 0)
                launch.assert_not_called()

    def test_worker_failure_stops_other_processes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            helper = root / "fake_worker.py"
            helper.write_text(WORKER, encoding="utf-8")
            args = argparse.Namespace(output_root=root, concurrent_tasks=2, group=0, seed=42,
                                      num_particles=16)
            command = lambda a, i, n: [sys.executable, str(helper), str(root), str(i), str(n),
                                       "2", str(a.seed), "yes"]
            processes = []
            original_popen = subprocess.Popen
            def tracked(*a, **kw):
                process = original_popen(*a, **kw)
                processes.append(process)
                return process
            with patch.dict(namespace, worker_command=command), patch.object(subprocess, "Popen", tracked):
                with self.assertRaisesRegex(RuntimeError, "exit code 9"):
                    namespace["run_task_workers"](args, 2, [28], 0)
            self.assertEqual(len(processes), 2)
            self.assertTrue(all(process.poll() is not None for process in processes))


if __name__ == "__main__":
    unittest.main()
