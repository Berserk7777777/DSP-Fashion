# Baseline Extension Point

Baselines are competing inference policies, not backbone adapters.  Each may
retain its own environment and native implementation, but must export the
same final `CandidateRecord`/`TaskRecord` artifacts as FashionPSP.

Planned entries:

- [`fk_steering/`](fk_steering/README.md): Feynman-Kac sequential Monte Carlo
  resampling with a frozen fashion reward;
- `bfs/`: breadth-first seed/trajectory search;
- `dsearch/`: diffusion tree search; and
- `standard/`: one seed without post-selection.

A baseline manifest must report initial seed pool, model calls, reward/VAE
calls, peak VRAM, wall-clock time, selected seed, and source image path.  It
must not consume GT images or retrieval candidates before final evaluation.
