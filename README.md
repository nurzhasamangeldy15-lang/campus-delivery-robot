# 🤖 Campus Delivery Robot: Autonomous Indoor Navigation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-19%20passed-brightgreen.svg)](./run_tests.py)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](./LICENSE)
[![Course](https://img.shields.io/badge/Course-AI%20Fundamentals-purple.svg)](https://qairu.edu.kz/)

> **Reaching a classroom when neither the robot nor 2GIS knows exactly where it is.**  
> Final Course Project for **AI Fundamentals** (QAIRU, Astana, 2026).  
> **Author:** Nurzhas Amangeldi (Solo Project)

---

## 📌 Project Overview & Problem Statement

Autonomous campus delivery robots must deliver parcels directly to university classrooms. However, in the real world:
1. **GPS & 2GIS are degraded indoors:** Signal attenuation and multipath reflections cause noisy, jumping coordinates.
2. **Classrooms are not mapped indoors:** Commercial mapping APIs (like 2GIS) stop at building perimeters and lack room-level indoor coordinates.
3. **Navigating under uncertainty is risky:** Moving while disoriented can cause wall collisions; pausing forever wastes precious delivery time.

This project solves this compound challenge by integrating **four distinct algorithms across three AI Fundamentals topics** into a unified, robust navigation pipeline.

---

## 🧠 Integrated AI Algorithms

| Topic | Algorithm | Role in Delivery Robot | Theoretical Formulation |
| :--- | :--- | :--- | :--- |
| **Probabilistic Reasoning & Sequential Inference** | **Hidden Markov Model (HMM)** | *"Where am I?"*<br>Filters noisy 2GIS readings into a belief state over grid cells. | Forward Algorithm:<br>$b_t(s') \propto P(e_t \mid s') \sum_s P(s' \mid s) b_{t-1}(s)$<br>Viterbi sequence decoding. |
| **Probabilistic Reasoning** | **Bayesian Inference** | *"Where is the room?"*<br>Resolves destination classroom (A, B, C) from verbal clues. | Posterior Updating:<br>$P(X \mid \text{clues}) \propto P(X) \prod_i P(\text{clue}_i \mid X)$ |
| **Heuristic Search** | **$A^*$ Search** | *"Which route?"*<br>Computes shortest collision-free route avoiding campus buildings. | Evaluation function $f(n) = g(n) + h(n)$ with **admissible Manhattan distance** $h(n)$. |
| **Sequential Decision Making** | **Markov Decision Process (MDP)** | *"Move or Relocalize?"*<br>Decides whether to follow the route or pause for sensor verification. | Bellman Value Iteration:<br>$V^*(s) = \max_a [R(s,a) + \gamma \sum_{s'} T(s,a,s') V^*(s')]$ |

---

## 🏗️ Architecture & Pipeline Flow

```
[ Noisy 2GIS Readings ]           [ Qualitative Verbal Clues ]
           │                                    │
           ▼                                    ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│     HMM Localizer       │          │ Bayesian Room Resolver  │
│  (Forward Algorithm)    │          │   (Posterior Updating)  │
└────────────┬────────────┘          └────────────┬────────────┘
             │                                    │
     Belief Pose & Confidence            Most Likely Goal Cell
             │                                    │
             ├─────────────────┬──────────────────┘
             ▼                 ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│      MDP Controller     │  │   A* Heuristic Router   │
│ (Bellman Value Iter.)   │  │   (Manhattan Distance)  │
└────────────┬────────────┘  └────────────┬────────────┘
             │                            │
      Action Decision             Collision-Free Route
   [ Move / Relocalize ]          [ (r0,c0) -> ... -> (rg,cg) ]
```

---

## 🚀 Quick Start

The core algorithms have **zero external dependencies** and run directly on pure Python.

### 1. Run the Console Demo (Instant)
```bash
python campus_robot.py
# or
python demo.py
```

### 2. Run the Interactive Web UI (Gradio)
```bash
# Optional: install UI dependencies
pip install -r requirements.txt

python app.py
```
Open `http://127.0.0.1:7860` in your web browser to test custom sensor inputs and interactive heatmaps.

### 3. Run the Automated Test Suite (19 Tests)
```bash
python run_tests.py
```
*Zero external packages required. Runs in ~0.02s.*

### 4. Run the Full Monte Carlo Benchmark
```bash
python evaluate.py
```
Executes 3,000 sensor trials and 756 path-planning pair evaluations, generating benchmark plots in `results/`.

---

## 📊 Empirical Evaluation & Key Results

### 1. Heuristic Search Efficiency ($A^*$ vs BFS/UCS)
Evaluated across all 756 start-goal cell pairs on the campus grid:
- **BFS / UCS ($h=0$):** 15.00 mean expanded nodes (100% optimal).
- **$A^*$ (Manhattan Heuristic):** **8.41** mean expanded nodes (100% optimal).
- **Result:** **$A^*$ reduces state exploration by 43.9%** while mathematically guaranteeing optimal path length.

### 2. Sensor Filtering Accuracy (HMM vs Baselines)
Evaluated across 3,000 simulated noisy sensor sequences:
- **Single 2GIS Reading:** 70.2% accuracy, mean Manhattan error = 0.52 cells.
- **Majority Voting:** 61.0% accuracy (fails as robot moves).
- **HMM Forward Filter:** **77.7% accuracy, mean error reduced to 0.23 cells** (54.2% error reduction).

### 3. Risk-Aware Decision Boundary (MDP)
- Test 1 (Consistent readings $\rightarrow$ confidence **$97\%$**): MDP action is **MOVE** ($97\% > 59.1\%$).
- Test 2 (Contradictory readings $\rightarrow$ confidence **$25\%$**): MDP action is **RELOCALIZE** ($25\% \le 59.1\%$).
- Threshold $p^* \approx 59.1\%$ dynamically balances delivery speed against collision penalty.

---

## 📁 Repository Structure

```
├── campus_robot/            # Core algorithmic package
│   ├── __init__.py          # Package initializer
│   ├── world.py             # 6x6 campus grid world & obstacle geometry
│   ├── hmm.py               # Forward filtering & Viterbi trajectory decoding
│   ├── bayes.py             # Bayesian posterior updating over candidate rooms
│   ├── astar.py             # A* heuristic search (and BFS/UCS benchmarks)
│   ├── mdp.py               # Value iteration & belief-weighted action selector
│   ├── pipeline.py          # Unified pipeline glue (HMM -> Bayes -> A* -> MDP)
│   └── visual.py            # Matplotlib campus heatmap and path rendering
├── tests/                   # Automated unit test suite
│   ├── conftest.py          # Pytest configuration & compatibility shims
│   └── test_robot.py        # 19 unit tests covering all 4 algorithms
├── results/                 # Evaluation figures and benchmark data
│   ├── hmm_accuracy.png     # Filtering accuracy vs readings curve
│   ├── search_nodes.png     # A* vs BFS node expansion bar chart
│   └── results.json         # Full empirical benchmark metrics
├── Campus_Delivery_Robot_Presentation.pptx # 7-slide defense presentation deck
├── TECHNICAL_REPORT.md      # Full academic technical report
├── campus_robot.py          # Main entry point (matches slide 6)
├── demo.py                  # Standalone CLI demonstration
├── app.py                   # Interactive Gradio web interface
├── evaluate.py              # Reproducible benchmark experiment script
├── run_tests.py             # Zero-dependency test runner
├── requirements.txt         # Optional UI & plotting dependencies
└── LICENSE                  # MIT License
```

---

## 📄 Academic Deliverables

- 📑 **Technical Report:** [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md)
- 🖥️ **Presentation Slides:** [Campus_Delivery_Robot_Presentation.pptx](./Campus_Delivery_Robot_Presentation.pptx)
- 🧪 **Unit Tests:** 19/19 passing ([tests/test_robot.py](./tests/test_robot.py))

---

## 👤 Author
**Nurzhas Amangeldi**  
Bachelor's Degree in Physical AI / AI and ML  
LLP «QAIRU», Astana, Kazakhstan (2026)
