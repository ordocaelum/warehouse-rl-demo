# Warehouse RL Training Lab - Best Practices Guide

## 🎓 Introduction

This guide will help you get the most out of the Warehouse RL Training Lab. Whether you're learning reinforcement learning fundamentals or experimenting with different configurations, these best practices will help you achieve better results faster.

---

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Difficulty Progression](#difficulty-progression-recommended)
3. [Parameter Tuning](#parameter-tuning)
4. [Model Management](#model-management)
5. [Debugging & Troubleshooting](#debugging--troubleshooting)
6. [Advanced Techniques](#advanced-techniques)
7. [Common Mistakes](#common-mistakes)

---

## 🚀 Getting Started

### First Time Setup

1. **Start with the default settings** – The UI dashboard comes with sensible defaults
2. **Run the simple task first** – Don't jump straight to hard mode
3. **Watch for 5-10 minutes** – Get a feel for how the agent explores and learns
4. **Check the reward graph** – Look for an upward trend in the orange line (20-episode moving average)

### Understanding What "Good" Looks Like

| Grid | Packages | Typical Results | Training Time |
|------|----------|-----------------|----------------|
| 8×8 | 1 | +8 to +10 avg reward | 5-10 min |
| 10×10 | 2 | +2 to +6 avg reward | 10-15 min |
| 12×12 | 3 | -2 to +2 avg reward | 15-20 min |
| 15×15 | 3 | -6 to -4 avg reward | 20-30 min |
| 15×15 | 5 | -10 to -8 avg reward | 30-60 min |

**Key insight:** Negative average reward doesn't mean failure! It just means the task is hard.

---

## 📈 Difficulty Progression (Recommended)

Start with an easy configuration and move to harder ones only once the easy one trains well.

> **Each stage is a separate training run with a new network.** No weights carry over
> between stages: a saved model can only be loaded into the *same* grid size and package
> count (see [Loading Models](#loading-models)). This is a progression for *you* – it lets you
> confirm that your setup and hyperparameters work on a task that learns in minutes before you
> spend longer on harder ones. It is not curriculum learning in the sense of one agent
> carrying its knowledge from stage to stage.

### The Proven Progression

#### ✅ **Stage 1: Master the Basics (8×8, 1 Package)**

```
Grid Size: 8
Packages: 1
Max Steps: 300
Timesteps: 100k
Learning Rate: 1.0e-03
```

**Expected results:**
- Reward converges to **+8 to +10** within 50-100 episodes
- Orange trend line goes UP and stays up
- Agent delivers packages consistently in 10-20 steps

**What agent learns:**
- How to navigate the grid
- How to find packages
- How to reach delivery zone
- Core mechanics of pickup/dropoff

**Checkpoint:** ✅ If average reward > +6, **SAVE MODEL**, then rename `warehouse_dqn_model.zip` to `model_stage1_8x8_1pkg.zip` (see [Saving Models](#saving-models))

---

#### ✅ **Stage 2: Add a Package (10×10, 2 Packages)**

```
Grid Size: 10
Packages: 2
Max Steps: 300
Timesteps: 150k (more training!)
Learning Rate: 1.0e-03
```

**Train fresh.** The Stage 1 model cannot be loaded here: grid size and package count both changed, so the observation space differs.

**Expected results:**
- Reward: **+2 to +6** average (lower because harder)
- Moving average should still trend upward
- Agent completes both deliveries occasionally

**What agent learns:**
- Multi-objective planning (visit 2 locations)
- Route optimization
- Prioritization (which package first?)

**Checkpoint:** ✅ If moving average stabilizes and stops degrading, **SAVE MODEL** and rename it to `model_stage2_10x10_2pkg.zip`

---

#### ✅ **Stage 3: Scale Up (12×12, 3 Packages)**

```
Grid Size: 12
Packages: 3
Max Steps: 300
Timesteps: 200k
Learning Rate: 1.0e-03
```

**Train fresh** (the Stage 2 model cannot be loaded into a different configuration).

**Expected results:**
- Reward: **-2 to +2** average (harder = lower rewards)
- Trend line should eventually go upward
- Occasional episodes with 0-5 packages delivered

**What agent learns:**
- Complex multi-objective planning
- Larger grid navigation
- How to handle tight time constraints

**Checkpoint:** ✅ **SAVE MODEL** and rename it to `model_stage3_12x12_3pkg.zip`

---

#### 🚀 **Stage 4: Challenge (15×15, 3-5 Packages)**

```
Grid Size: 15
Packages: 3 (or 4, or 5)
Max Steps: 221
Timesteps: 300k+
Learning Rate: 1.0e-03
```

**This is the real test!** Training will take longer, but reward signals become clearer.

**Expected results:**
- 3 packages: **-4 to -6** average
- 5 packages: **-8 to -12** average
- Graph will be noisy but should show slight improvement

---

### Why Start Easy

```
Starting at 15×15, 5 packages:
  → A failed run takes a long time to look like a failure
  → You cannot tell a broken setup from a hard task

Starting at 8×8, 1 package:
  → Learns in minutes, so a flat graph means something is wrong
  → Hyperparameters that work here are a reasonable starting point for harder stages
  → You learn what each stage's reward curve should look like
```

---

## 🎚️ Parameter Tuning

### When to Adjust Each Parameter

#### **Grid Size**
| Value | Use Case | Notes |
|-------|----------|-------|
| 6-8 | Learning & debugging | Fast episodes, low state space |
| 10-12 | Intermediate challenge | Balanced difficulty |
| 13-15 | Advanced challenge | Slow episodes, huge state space |

⚠️ **Avoid jumping from 8×8 to 15×15!** Go gradually.

#### **Packages**
| Value | Use Case | Notes |
|-------|----------|-------|
| 1 | Foundational | Best for first-time learning |
| 2-3 | Multi-objective | Good middle stages of the progression |
| 4-5 | Advanced | Requires much more training |

⚠️ **Each additional package multiplies complexity exponentially!**

#### **Max Steps**
| Value | Use Case | Notes |
|-------|----------|-------|
| 100-150 | Aggressive | Forces efficiency |
| 200-300 | Balanced (recommended) | Realistic task duration |
| 400-500 | Lenient | Easier but slower rewards |

**Rule of thumb:** `Max Steps = (Grid Size × 20) + (Packages × 30)`
- 8×8, 1 pkg: 8×20 + 30 = 190 ✓
- 12×12, 3 pkg: 12×20 + 90 = 330 ✓

#### **Learning Rate**
| Value | Use Case | Effects |
|-------|----------|---------|
| 1e-4 | Stable but slow | Gradual learning, less noise |
| 1e-3 | Balanced (recommended) | Good for most tasks |
| 1e-2 | Fast but unstable | Quick learning, high variance |

⚠️ **Don't change during training!** Set it before hitting START.

#### **Timesteps**
| Value | Training Time | When to Use |
|-------|---------------|------------|
| 50k | 5 min | Quick experimentation |
| 100k | 10 min | Standard, good baseline |
| 200k | 20 min | Reaching convergence |
| 300k+ | 30+ min | Solving hard tasks |

**Rule of thumb:** `Timesteps = 100k × (Grid Size / 10) × Packages`

#### **Exploration Parameters**
| Parameter | Default | When to Adjust |
|-----------|---------|-----------------|
| Init Epsilon | 1.0 | Keep at 1.0 (100% exploration) |
| Final Epsilon | 0.1 | Set higher (0.2) for hard tasks |
| Buffer Size | 10k | Increase to 50k for complex tasks |

---

## 💾 Model Management

### Saving Models

**When to save:**
- ✅ When moving average plateaus (agent has converged)
- ✅ When you want to preserve a good checkpoint
- ✅ Before trying a new configuration
- ✅ After significant improvement jumps

**How to save:**
1. Watch training until it stabilizes (5-20 minutes)
2. Check the reward graph for upward/stable trend
3. Click **[SAVE MODEL]** button
4. Model saves to `warehouse_dqn_model.zip`

**Naming convention (manual):**
```bash
mv warehouse_dqn_model.zip models/model_stage1_8x8_1pkg.zip
mv warehouse_dqn_model.zip models/model_stage2_10x10_2pkg.zip
mv warehouse_dqn_model.zip models/model_stage3_12x12_3pkg.zip
```

### Loading Models

**When you can load:**
✅ Same environment configuration (same grid size and package count)
❌ Different grid size or package count (training stops with an error)

**Moving to a new configuration:**
```
1. Save model trained on 8×8, 1 pkg (keep it for later 8×8, 1 pkg runs)
2. Set grid to 10×10, packages to 2
3. Train fresh on the new task (don't load!)
   → This is a new network; nothing is transferred from the 8×8 model
```

**Why not load a 1-pkg model into a 2-pkg environment, or an 8×8 model into 10×10?**
- The observation space is saved with the model, and SB3 refuses to load into a different one
- Package count changes the input size (6 dims → 9 dims), so the saved weights don't fit
- Grid size changes the observation bounds, which also counts as a different space
- You'll get: `"Observation spaces do not match" error`

---

## 🐛 Debugging & Troubleshooting

### Issue: Model crashes on load with "Observation spaces do not match"

**Cause:** The loaded model was trained with a different grid size or package count. The model is loaded when you press START, so the error appears then (state shows ERROR).

**Solution:**
1. Set **Grid Size** and **Packages** to the values the saved model was trained with
2. Click **LOAD MODEL**
3. Click **START TRAINING**

(RESET STATS only clears the displayed graph and metrics; it has no effect on loading.)

**Example:**
- Saved model: 8×8, 1 package
- Can load into: 8×8, 1 package ✓
- Cannot load into: 8×8, 2 packages ✗
- Cannot load into: 10×10, 1 package ✗

### Issue: Agent isn't improving (flat reward graph)

**Checklist:**
1. ✅ Is training actually running? (Check that Total Steps keeps increasing)
2. ✅ Are timesteps high enough? (100k minimum, 200k+ better)
3. ✅ Is the task too hard? (Try easier configuration first)
4. ✅ Is learning rate reasonable? (1e-3 is default, try 1e-4 if unstable)
5. ✅ Did you train long enough? (Wait 10+ minutes)

**Fix:**
- Reduce difficulty (smaller grid, fewer packages)
- Increase timesteps to 200k-300k
- Follow the difficulty progression

### Issue: Training is too slow

**Measure it first:** training speed is **Total Steps ÷ Elapsed** in the PERFORMANCE panel.
The **UI FPS** figure is how often the dashboard redraws (capped at 60); it does not measure training speed.

**Solutions:**
1. Lower **Timesteps** for quick experiments
2. Use an easier configuration, which needs fewer timesteps to learn
3. Close other applications
4. Accept slower training (it still works!)

### Issue: Agent gets stuck in one location

**This is normal!** Early exploration is random. Wait longer.

**Patience guide:**
- Episodes 1-50: Random wandering (expected)
- Episodes 50-200: Learning starts
- Episodes 200+: Convergence or improvement

---

## 🔬 Advanced Techniques

### Experiment: Compare Different Configurations

**Setup:**
```
Track 3 parallel experiments:

Config A: Grid=8, Pkg=1, LR=1e-3  → Check final reward
Config B: Grid=8, Pkg=1, LR=1e-4  → Is slower better?
Config C: Grid=8, Pkg=1, LR=1e-2  → Is faster unstable?
```

**Method:**
1. Run Config A → Save as `exp_a.zip` → Note final average reward
2. Run Config B → Save as `exp_b.zip` → Note final average reward
3. Run Config C → Save as `exp_c.zip` → Note final average reward
4. Compare results

### Transfer Between Configurations: Not Supported

Loading a model trained on one grid size or package count into another is not possible
(see [Loading Models](#loading-models)), so transfer-learning experiments cannot be run with
this tool as it stands.

### Advanced Hyperparameter Sweep

**Grid search** different learning rates:
```python
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

For each lr:
  - Set Learning Rate to lr
  - Train on 8×8, 1 pkg for 100k steps
  - Record final average reward
  - Plot results
```

**Insight:** You'll find the sweet spot LR for your task!

---

## ⚠️ Common Mistakes

### ❌ Mistake #1: Jumping to Hard Tasks Too Early

```
WRONG: "I want a 15×15 grid with 5 packages!"
       → Agent struggles for an hour, learns nothing

RIGHT: Start with 8×8, 1 package
       → Master it in 10 minutes
       → Scale up gradually
       → Total time: 1 hour with MUCH better results
```

### ❌ Mistake #2: Changing Parameters Mid-Training

```
WRONG: Start with LR=1e-3, then change to LR=1e-4 halfway through
       → Training becomes inconsistent

RIGHT: STOP, adjust all parameters, then START
       → START always begins a new run with the current settings
       (sliders moved during a run only apply at the next START)
```

### ❌ Mistake #3: Expecting Immediate Results

```
WRONG: Train for 2 minutes, check reward graph, give up
       
RIGHT: Train for at least 10 minutes
       → Learning ramp-up takes time
       → Patience is key!
```

### ❌ Mistake #4: Skipping the Easy Stages

```
WRONG: Go straight from 8×8, 1 pkg to 15×15, 5 pkg
       → Long runs before you can tell whether anything is wrong

RIGHT: Follow the difficulty progression
       → Confirm each stage trains before moving on
       → Each stage is still a fresh run (no weights carry over)
```

### ❌ Mistake #5: Loading Wrong Model Version

```
WRONG: Train 12×12, 3 pkg model → save
       Then try loading into 15×15, 5 pkg → ERROR: observation spaces do not match

RIGHT: Keep track of model versions
       → Save with descriptive names
       → Only load into matching configs
```

### ❌ Mistake #6: Ignoring the Reward Graph

```
WRONG: Look only at episode reward (noisy)
       
RIGHT: Watch the 20-episode moving average (orange line)
       → This is the true learning signal
       → Should trend upward over time
```

---

## 📊 Interpreting the Reward Graph

### What Good Progress Looks Like

```
Episode Reward (Blue line):  ↗ ↘ ↗ ↘ ↗  (Noisy, but trending up)
20-ep Moving Average (Orange line):  ╱╱╱╱  (Smooth, consistent upward trend)
```

### What Stalled Learning Looks Like

```
Episode Reward (Blue line):  ↘ ↘ ↘ ↘ ↘  (Consistently negative)
Moving Average (Orange line): ___   (Flat, not improving)
```

**Action:** This means either:
- Task is too hard (scale down)
- Need more training time (increase timesteps)
- Hyperparameters need tuning

### What Collapse Looks Like

```
Moving Average starts high (good) but suddenly crashes down
```

**Cause:** Usually learning rate is too high (1e-2 is too aggressive)

**Fix:** Reduce to 1e-3 or 1e-4, retrain

---

## 🎯 Quick Reference Cheat Sheet

### First 5 Minutes
```
1. Launch: python ui_dashboard.py
2. Accept default settings
3. Click START TRAINING
4. Watch and learn!
```

### Recommended First Experiment
```
Grid Size: 8
Packages: 1
Max Steps: 300
Timesteps: 100k
Learning Rate: 1e-3
→ Train until orange line plateaus (~5-10 min)
→ Click SAVE MODEL
```

### Scaling Up (When Ready)
```
Grid Size: 10
Packages: 2
Max Steps: 300
Timesteps: 150k
→ Train fresh (don't load old model)
→ Should see learning within 15 min
```

### Debugging Command
```
If things break:
1. RESET STATS
2. Set difficulty to 8×8, 1 pkg
3. Set timesteps to 50k
4. START TRAINING
→ Verify system works before scaling
```

---

## 🚀 Next Steps

1. **Complete the progression:** Follow Stage 1 → Stage 2 → Stage 3
2. **Save your best models:** Build a model collection
3. **Experiment:** Try different learning rates or grid sizes
4. **Compare:** Which configuration learns fastest?
5. **Optimize:** Find your sweet spot parameters

---

## 📚 Further Reading

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Original Deep Q-Network algorithm
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - Implementation details
- [Gymnasium Docs](https://gymnasium.farama.org/) - Environment design
- [RL Curriculum Learning](https://arxiv.org/abs/2003.12236) - Background on curriculum learning (weight transfer between stages, which this tool does not do)

---

## 💡 Tips & Tricks

### Pro Tip #1: Keyboard Shortcuts
```
SPACE → Pause/Resume training
S     → Save model (once training has finished or been stopped)
R     → Reset stats
Q/ESC → Exit
```

### Pro Tip #2: Watch Total Steps, Not FPS
- **UI FPS** is the dashboard's redraw rate (capped at 60); it does not reflect training speed or task difficulty
- **Total Steps ÷ Elapsed** is the training speed

### Pro Tip #3: Use the Mini Graph
The reward history in the right panel shows last 100 episodes. Watch for:
- Upward trend = Learning ✓
- Flat line = Stalled ✗
- Downward trend = Degrading ✗

### Pro Tip #4: Episode Reward vs Moving Average
- **Episode Reward:** Noisy, ignore short-term spikes
- **Moving Average:** Smooth trend, this is what matters!

---

Happy training! 🚀 Feel free to experiment and discover what works best for your setup!

