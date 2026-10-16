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
