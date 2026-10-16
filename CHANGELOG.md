# Changelog

All notable changes to this project are recorded here. Each entry says what
was wrong, what changed, and whether behaviour changed.

## Unreleased

### Reproducibility

- **Fixed: seeding had no effect.** Package placement used the process-global
  `np.random` instead of the environment's own generator, so
  `reset(seed=42)` twice gave two different layouts, and constructing the
  environment consumed global RNG state. The environment now draws only from
  `self.np_random`.
- **Added: seeds are passed through training.** `train_agent(seed=...)`,
  `python train_agent.py --seed N` (default `0`) and a **Seed** slider in the
  dashboard pass the seed to `make_vec_env` and `DQN`.
- **Added: reproducibility tests.** Same-seed resets and rollouts are
  identical, the environment never touches global `np.random`, and two seeded
  CPU training runs end with identical weights. README documents what is not
  guaranteed (GPU, different installs).

### Phase A — defects a user hits immediately

- **Fixed: packages reported their pickup cell forever.** A carried package
  now moves with the agent, and a delivered one stays at the delivery zone, in
  the observation and in both renderers. *Behaviour change:* the observation
  differs while carrying and after delivery, so models trained before this
  change were trained on different inputs.
- **Fixed: the last frame of every episode was never shown.** The vectorised
  env resets before the callback runs, so the display showed the next
  episode's start instead of the delivery. Both callbacks now draw done steps
  from the VecEnv's `terminal_observation`.
- **Fixed: STOP overwrote `warehouse_dqn_model.zip`** with the partially
  trained model. A stopped run is no longer saved automatically; SAVE MODEL can
  still save it on request.
- **Fixed: S ("step one episode") advanced one step.** While paused, S now runs
  to the end of the current episode and pauses on its final frame, as the
  legend, banner and README describe.
- **Fixed: the visualizer's reward history was per step.** The sparkline now
  receives one value per finished episode through its own channel, which is
  never dropped.
- **Docs: `BEST_PRACTICES.md` corrected where it contradicted the code**
  (code taken as correct in every case): observation sizes (6 → 9, not
  12 → 15); the orange graph line is a 20-episode average; UI FPS is the
  redraw rate, not training speed; RESET STATS does not affect loading; S
  cannot save mid-run; and a model cannot be loaded into a different grid size
  *or* package count. The last point means no weights carry over between
  stages, so the "curriculum learning" section, which described knowledge
  transferring between stages, is now "Difficulty Progression" and says each
  stage is a fresh network.
- **Added: test suite** (`pytest`) covering the environment contract
  (observation layout and bounds, every reward term, termination and
  truncation, Gymnasium's `check_env`) and the callback fixes above.

### Phase B — contract and correctness gaps

- **Added: `info` is populated.** `step()` reports `delivered`,
  `num_packages`, `delivery_zone`, `action_mask`, `invalid_action`,
  `is_success` (SB3 logs it as `rollout/success_rate`) and
  `termination_reason` (`"all_delivered"`, `"max_steps"` or `None`). It was
  always `{}`.
- **Fixed: the delivery zone was hard-coded in both renderers.** Its position
  is now in `info`; the callbacks forward it and the renderers draw it there.
- **Added: constructor validation.** `grid_size < 3`, `num_packages < 1`,
  `max_steps < 1`, a non-finite `step_cost`, or non-integer sizes raise
  immediately instead of crashing later or ending every episode at once.
- **Fixed: STOP and the training thread raced to set the dashboard state.**
  Only the UI thread writes it now; a stopped run always ends "idle", a
  completed run "done".
- **Fixed: CLI and dashboard trained differently** for the same visible
  settings. Both build the agent through `make_env()`/`make_dqn()`.
  *Behaviour change:* the dashboard's `exploration_fraction` is 0.2 (was
  0.3), matching the CLI and README.
- **Fixed: `test_agent` evaluated on the default environment.** It takes the
  training configuration and refuses a model whose observation space does not
  match. The CLI gains `--grid-size`, `--packages`, `--max-steps`,
  `--step-cost`.
- **Added: Gymnasium conventions.** `render_mode` (`"human"`, `"ansi"`),
  registration as `Warehouse-v0`, `action_masks()`, and environments are
  closed after use. *Behaviour change:* `render()` without a `render_mode` now
  warns instead of printing.
- **Fixed: loose observation bounds.** Coordinates are bounded by
  `grid_size - 1` and flags by 1; internal state is explicitly integer and
  rewards are Python floats. *Behaviour change:* models saved before this
  change cannot be loaded, because SB3 compares the saved observation space.

### Phase C — operational

- **Added: pinned dependencies and a lock file.** `requirements.txt` pins the
  direct dependencies exactly; `requirements.lock` pins every package for all
  platforms (generated with `uv pip compile --universal`); a fresh install from
  it passes the suite. Supported Python is 3.12–3.13 (was documented as 3.8+,
  which current releases do not install on).
- **Fixed: displays showed stale state.** Updates went through bounded queues
  that dropped the *newest* update when full. A keep-latest slot now replaces
  them: training never waits, each frame shows the current step, and the
  number of undrawn steps is shown.
- **Fixed: the dashboard copied 100 episode rewards into every step's
  update.** Episode returns now travel once per episode on their own channel.
  RESET STATS now stays cleared during training.
- **Fixed: the visualizer ran pygame off the main thread**, which macOS does
  not support. It now runs on the main thread with training beside it.
  *API change:* `WarehouseVisualizer.start()` is replaced by `run()`.
- **Added: run folders.** Each run writes `runs/<time>_g<grid>_p<pkgs>_s<seed>/`
  with `config.json` (settings and library versions), the model, monitor CSV,
  SB3 metrics as CSV and TensorBoard, and for CLI runs a log and the reward
  plot. Nothing is overwritten between runs. *Behaviour change:* models are no
  longer written to `./warehouse_dqn_model.zip`.
- **Added: checkpointing and full resume.** Checkpoints (model and replay
  buffer) every 10,000 steps; SAVE MODEL works while training or paused; the
  replay buffer is saved with every model; LOAD MODEL sets the environment
  sliders from the saved run and START resumes it with its optimizer state,
  hyperparameters, step count and replay buffer (previously only weights were
  copied into a fresh model).
