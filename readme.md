# IronLot — Automated Bidding Model for Used Car Auctions

> **Predicting wholesale clearing prices and deploying an autonomous bidding agent with a $500k bankroll against rival AI models.**

Built as part of a competitive ML module where inference was weighted alongside prediction accuracy — understanding *why* a car was priced a certain way mattered as much as getting the number right.

---

## Overview

IronLot is an end-to-end machine learning pipeline that:

1. **Cleans and wrangles** 447,048 rows of used car auction data with zero data leakage
2. **Engineers features** grounded in statistical analysis and domain reasoning
3. **Trains three LightGBM models** — mean, 25th percentile, and 75th percentile predictions
4. **Deploys an autonomous agent** that bids on cars sequentially using auction theory, quantitative finance, and information design

---

## Repository Structure

```
IronLot/
│
├── analysis_MdAdnanKhalid.ipynb     # Full pipeline: cleaning → EDA → modelling
├── agent_MdAdnanKhalid.py           # Live bidding agent
│
├── model_MdAdnanKhalid.pkl          # Trained mean LightGBM model
├── encoders_MdAdnanKhalid.pkl       # All encoders, lookup tables, quantile models
│
├── report_MdAdnanKhalid.pdf         # Full technical report
└── README.md
```

---

## The Pipeline

### 1. Data Cleaning

**Train-validation split first.** Every statistic, lookup table, and encoder computed on training data only. The agent cannot call `df.mean()` at auction time — all knowledge must travel through `encoders.pkl`.

**Missing values.** Not all missingness is equal:
- Identified MAR vs MCAR by grouping a binary missingness flag across other columns
- Built a 3-tier hierarchical imputation for each column — most specific context first, global fallback last
- Added `pd.notna()` guards after discovering rare groups stored `NaN` medians silently

**Outlier treatment.** Standard IQR gave a lower bound of −$9,950 for prices. The fix:

```
upper = exp(Q3(ln p) + 1.5 × IQR(ln p)) − 1
```

Log-IQR creates asymmetric bounds matching the right-skewed distribution. Every flagged outlier was profiled against its other columns before any removal decision — Ferraris with 12k miles and condition 4.45 are not noise.

---

### 2. Feature Engineering

**Body consolidation.** 45 raw body type values consolidated to 9 clean categories via a saved lookup dictionary. Eliminates near-empty OHE columns.

**Usage intensity — tested and dropped.** `usage_intensity = odometer ÷ car_age` scored MI = 0.066 vs year's 0.453 and odometer's 0.420. Dividing two variables reduces degrees of freedom from two to one — you cannot recover the original signals from their ratio. Information theory formalises this:

```
I(X₁, X₂ ; Y) ≥ I(f(X₁, X₂) ; Y)   for any function f
```

**Binary interaction flags.** Color had global MI = 0.031 but created a $15,000 price spread within luxury vehicles. Orange on a Porsche encodes exotic specification, not color preference. A raw column has no context. An interaction flag does:

```python
luxury_exotic_color = is_luxury AND has_exotic_color
```

Five flags engineered: `is_luxury`, `has_exotic_color`, `has_premium_interior`, `luxury_exotic_color`, `luxury_premium_interior`.

---

### 3. EDA Key Findings

**Spearman over Pearson.** Selling price has skewness 2.1, kurtosis 15. Pearson squares deviations — one Ferrari dominates. Spearman uses ranks — bounded, outlier-robust.

**Mutual Information over correlation.** MI captures any dependency (linear or not). `model` scored 0.639, `transmission` scored 0.003. This guided both encoding strategy and feature selection.

| Feature | MI Score |
|---------|----------|
| model | 0.639 |
| trim | 0.545 |
| make | 0.201 |
| body | 0.140 |
| state | 0.102 |
| transmission | 0.003 |

**CV structural break at condition 3.0.** Coefficient of Variation (σ/μ) per condition band revealed a sharp drop from 0.70 to 0.59 at condition 3.0 — the wholesale-to-retail threshold where buyer pools expand and prices stabilise. This single EDA finding became the foundation of the agent's entire confidence architecture.

| Condition Band | Mean Price | CV |
|----------------|------------|-----|
| (1.5, 2.0] | $5,251 | **0.98** ← peak noise |
| (2.5, 3.0] | $10,249 | 0.70 |
| **(3.0, 3.5]** | **$12,671** | **0.59** ← structural break |
| (4.5, 5.0] | $22,095 | 0.51 |

---

### 4. Modelling

**Why LightGBM:**
- Non-additive interactions (make × year × condition) rule out linear models
- Sequential boosting explicitly corrects prior residuals — outperforms Random Forest's independent bagging
- Leaf-wise growth achieves lower loss for the same number of leaves vs XGBoost's level-wise
- **Native `objective='quantile'`** — three uncertainty models at negligible extra training cost

**Hyperparameter tuning with Optuna TPE.** Grid search over 10 parameters with 5 values each requires 9.7 million trials. Optuna's Tree-structured Parzen Estimator fits density models to past trial results and samples intelligently.

| Model | RMSE |
|-------|------|
| Baseline (default LightGBM) | $2,669.39 |
| **Tuned (Optuna, 100 trials)** | **$1,841.91** |

Mean prediction bias: −$1.17 on a $13,663 average price. Statistically zero systematic error.

**Three-model quantile architecture.** Same hyperparameters, only `objective` and `alpha` change:

```python
# Mean model
objective='regression'

# Pessimistic ceiling (25th percentile)  
objective='quantile', alpha=0.25

# Optimistic check (75th percentile)
objective='quantile', alpha=0.75
```

Uncertainty score per car:
```
relative_width = (p̂₀.₇₅ − p̂₀.₂₅) / p̂₀.₅₀
```

Validation confirmed the quantile models independently reproduced the CV structural break — condition 1.0–1.5 cars averaged width 0.414, condition 4.5–5.0 averaged 0.087. Two methods, same conclusion.

---

### 5. Bidding Agent

The agent processes one car at a time as a dictionary. No dataset access at auction time.

**Decision flow:**

```
Car arrives as dict
        │
        ▼
Preprocess (impute → encode → engineer flags)
        │
        ▼
Three predictions: p̂₀.₂₅, p̂₀.₅₀, p̂₀.₇₅ → compute width
        │
        ├── Negative prediction OR width ≥ 0.414 ──► PASS
        │
        ▼
width < 0.139 (high confidence)?
        │
   YES  │                     NO  │
        ▼                         ▼
Anchor = p̂₀.₅₀              Anchor = p̂₀.₂₅  ← bid shading
Margin = 10%                 Margin = 5%
        │                         │
        └──────────┬──────────────┘
                   ▼
        Fractional Kelly sizing
        f = (edge/w²) × 0.5 × (B/B₀)
                   │
                   ▼
        Hyperbolic bid sequence
        b_{r+1} = b_r + max((c_eff − b_r) · f · (1−0.3w)/(1+0.5r), $50)
```

**Why fractional Kelly:**  
Full Kelly maximises long-run growth but has dangerously high variance over finite auctions. Half-Kelly reduces variance by 75% at the cost of 25% expected growth. The bankroll ratio `B/B₀` provides automatic drawdown protection — no explicit trigger needed.

**Why hyperbolic over exponential decay:**  
By round 15, exponential decay (0.8ʳ) produces a $103 increment on a $20k car — rivals read tiny bids and infer you're at your ceiling. Hyperbolic decay (1/(1+0.5r)) still produces $375 at round 15. The ceiling stays hidden.

**Winner's Curse mitigation:**  
Anchoring to `p̂₀.₂₅` for uncertain cars implements bid shading — the standard rational response to the Winner's Curse in first-price auctions.

Every agent threshold (0.139, 0.414, 10%, 5%) was derived directly from EDA findings. Nothing arbitrary.

---

## Requirements

```bash
pip install pandas numpy lightgbm scikit-learn category_encoders optuna joblib statsmodels
```

---

## Usage

```python
import joblib

# Load
model    = joblib.load('model_MdAdnanKhalid.pkl')
encoders = joblib.load('encoders_MdAdnanKhalid.pkl')

# The agent handles a car dictionary
from agent_MdAdnanKhalid import Agent

agent = Agent(
    model_path    = 'model_MdAdnanKhalid.pkl',
    encoders_path = 'encoders_MdAdnanKhalid.pkl',
    starting_bankroll = 500000
)

bid = agent.analyze_item(car_dict, current_bid=12000)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Validation RMSE | $1,841.91 |
| Baseline RMSE | $2,669.39 |
| Improvement | 31% |
| Mean bias | −$1.17 |
| Training set size | ~357,000 rows |
| Features | 21 |

---

## Author

**Md Adnan Khalid** · Roll No. 250103060  
Pawn Stars Quantitative Analysis Division

---

*Report, slides, and study guide included in the repository.*
