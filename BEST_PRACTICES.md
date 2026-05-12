# Warehouse RL Training Lab - Best Practices Guide

## 🎓 Introduction

This guide will help you get the most out of the Warehouse RL Training Lab. Whether you're learning reinforcement learning fundamentals or experimenting with different configurations, these best practices will help you achieve better results faster.

---

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Curriculum Learning](#curriculum-learning-recommended)
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
4. **Check the reward graph** – Look for upward trends in the 50-episode moving average

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

## 📈 Curriculum Learning (Recommended)

The most effective way to train a powerful agent is **curriculum learning**: start easy, gradually increase difficulty.

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

**Checkpoint:** ✅ If average reward > +6, **SAVE THIS MODEL** as `model_stage1_8x8_1pkg.zip`

---

#### ✅ **Stage 2: Add a Package (10×10, 2 Packages)**

```
Grid Size: 10
Packages: 2
Max Steps: 300
Timesteps: 150k (more training!)
Learning Rate: 1.0e-03
```

**Options:**
- **A) Transfer Learning (Faster):** Load saved Stage 1 model, then train
- **B) Fresh Start (More Robust):** Don't load, train from scratch

**Expected results:**
- Reward: **+2 to +6** average (lower because harder)
- Moving average should still trend upward
- Agent completes both deliveries occasionally

**What agent learns:**
- Multi-objective planning (visit 2 locations)
- Route optimization
- Prioritization (which package first?)

**Checkpoint:** ✅ If moving average stabilizes and stops degrading, **SAVE** as `model_stage2_10x10_2pkg.zip`

---

#### ✅ **Stage 3: Scale Up (12×12, 3 Packages)**

```
Grid Size: 12
Packages: 3
Max Steps: 300
Timesteps: 200k
Learning Rate: 1.0e-03
```

**Options:**
- Load Stage 2 model OR train fresh (both work)

**Expected results:**
- Reward: **-2 to +2** average (harder = lower rewards)
- Trend line should eventually go upward
- Occasional episodes with 0-5 packages delivered

**What agent learns:**
- Complex multi-objective planning
- Larger grid navigation
- How to handle tight time constraints

**Checkpoint:** ✅ **SAVE** as `model_stage3_12x12_3pkg.zip`

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

### Why Curriculum Learning Works

```
Without curriculum learning:
Grid 15×15, 5 packages → Agent sees billions of possible states
                       → Takes millions of steps to find a single delivery
                       → Very sparse reward signal
                       → Learning is slow or fails

With curriculum learning:
Stage 1: Learn core mechanics quickly → High reward signal → Agent builds intuition
Stage 2: Add complexity gradually → Still sees successes → Learns new skills
Stage 3: Increase difficulty → Agent has foundation → Learning accelerates
Stage 4: Go hard → Agent has learned representations → Can tackle complex tasks
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
| 2-3 | Multi-objective | Good sweet spot for curriculum |
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
❌ Different environment (will crash)

**Transfer learning workflow:**
```
1. Save model trained on 8×8, 1 pkg
2. Set grid to 10×10, packages to 2
3. Train the model fresh on new task (don't load!)
   → Agent learns to adapt its knowledge
```

**Why not load a 1-pkg model into 2-pkg environment?**
- Neural network input shape changes (12 dims → 15 dims)
- Saved weights won't fit new network
- You'll get: `"Observation spaces do not match" error`

---

## 🐛 Debugging & Troubleshooting

### Issue: Model crashes on load with "Observation spaces do not match"

**Cause:** Loaded model was trained with different package count

**Solution:**
1. Use **RESET STATS**
2. Match the package count to the saved model
3. Then load the model

**Example:**
- Saved model: 8×8, 1 package
- Can load into: 8×8, 1 package ✓
- Cannot load into: 8×8, 2 packages ✗

### Issue: Agent isn't improving (flat reward graph)

**Checklist:**
1. ✅ Is training actually running? (Check FPS in bottom right)
2. ✅ Are timesteps high enough? (100k minimum, 200k+ better)
3. ✅ Is the task too hard? (Try easier configuration first)
4. ✅ Is learning rate reasonable? (1e-3 is default, try 1e-4 if unstable)
5. ✅ Did you train long enough? (Wait 10+ minutes)

**Fix:**
- Reduce difficulty (smaller grid, fewer packages)
- Increase timesteps to 200k-300k
- Try curriculum learning progression

### Issue: Training is too slow (FPS is 20-30)

**Cause:** Task is too complex or machine is busy

**Solutions:**
1. Reduce grid size
2. Reduce package count
3. Lower max_steps
4. Close other applications
5. Accept slower training (it still works!)

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

### Experiment: Transfer Learning Effectiveness

**Question:** Does loading a trained model help or hurt on a new task?

**Test:**
```
Path A (Transfer):
  1. Train on 8×8, 1 pkg → Save
  2. Load model, train on 10×10, 2 pkg fresh
  
Path B (Fresh):
  1. Train fresh on 10×10, 2 pkg
  
Compare: Which learns faster?
```

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

RIGHT: Click RESET STATS, adjust all parameters first, then START
       → Clean training run
```

### ❌ Mistake #3: Expecting Immediate Results

```
WRONG: Train for 2 minutes, check reward graph, give up
       
RIGHT: Train for at least 10 minutes
       → Learning ramp-up takes time
       → Patience is key!
```

### ❌ Mistake #4: Not Using Curriculum Learning

```
WRONG: Train 8×8, 1 pkg → then 8×8, 5 pkg
       → Agent doesn't transfer knowledge from 1 pkg to 5 pkg

RIGHT: Follow curriculum progression
       → Each stage builds on previous
       → Knowledge compounds
```

### ❌ Mistake #5: Loading Wrong Model Version

```
WRONG: Train 12×12, 3 pkg model → save
       Then try loading into 15×15, 5 pkg → crash!

RIGHT: Keep track of model versions
       → Save with descriptive names
       → Only load into matching configs
```

### ❌ Mistake #6: Ignoring the Reward Graph

```
WRONG: Look only at episode reward (noisy)
       
RIGHT: Watch the 50-episode moving average (orange line)
       → This is the true learning signal
       → Should trend upward over time
```

---

## 📊 Interpreting the Reward Graph

### What Good Progress Looks Like

```
Episode Reward (Blue line):  ↗ ↘ ↗ ↘ ↗  (Noisy, but trending up)
Moving Average (Orange line):  ╱╱╱╱  (Smooth, consistent upward trend)
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

1. **Complete the curriculum:** Follow Stage 1 → Stage 2 → Stage 3
2. **Save your best models:** Build a model collection
3. **Experiment:** Try different learning rates or grid sizes
4. **Compare:** Which configuration learns fastest?
5. **Optimize:** Find your sweet spot parameters

---

## 📚 Further Reading

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Original Deep Q-Network algorithm
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - Implementation details
- [Gymnasium Docs](https://gymnasium.farama.org/) - Environment design
- [RL Curriculum Learning](https://arxiv.org/abs/2003.12236) - Why we start easy

---

## 💡 Tips & Tricks

### Pro Tip #1: Keyboard Shortcuts
```
SPACE → Pause/Resume training
S     → Save model (after pausing)
R     → Reset stats
Q/ESC → Exit
```

### Pro Tip #2: Monitor the FPS
- 50-60 FPS = Normal
- 30-50 FPS = Task getting hard
- <30 FPS = Consider reducing difficulty

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

