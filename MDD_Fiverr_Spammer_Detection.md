# MODEL DEVELOPMENT DOCUMENT (MDD)
## Project: Fiverr Malicious Spammer & Malware Distributor Detection Engine

| Document Metadata | Details |
| :--- | :--- |
| **Model Name / ID** | `MDL-TS-FIVERR-SPAM-v1.0` |
| **Model Owner / Author** | Jolly Nagar (Machine Learning Engineering) |
| **Business Domain** | Marketplace Trust & Safety / Cyber Fraud & Malware Prevention |
| **Model Type** | Supervised Binary Classification (Histogram-Based Gradient Boosted Decision Trees) |
| **Target Variable** | `label` (`1` = Spammer / Malware Distributor, `0` = Legitimate User) |
| **Dataset Source** | `fiverr_data.csv` (`458,798` records, `53` columns) |
| **Document Status** | Production-Ready Release |

---

## 1. Executive Summary & Business Context

### 1.1 Problem Overview
Cybercriminals actively exploit freelance marketplaces such as Fiverr by registering buyer accounts and sending unsolicited job offers containing weaponized file attachments. Disguised as project briefs or design specifications, these attachments execute Remote Access Trojans (RATs) or keyloggers such as Agent Tesla upon opening. Compromised freelancers suffer drained platform revenues, wasted bidding fees, fraudulent project disputes, and full account takeovers, severely impacting their livelihood and professional network standing.

### 1.2 Modeling Objective
The objective of `MDL-TS-FIVERR-SPAM-v1.0` is to assign a calibrated probability score $P(\text{label} = 1 \mid \mathbf{x}) \in [0, 1]$ to every user account based on 51 anonymized pre-registration and post-registration behavioral attributes (`X1`–`X51`). This score drives an automated, real-time Trust & Safety decision engine that intercepts malicious accounts before weaponized attachments reach freelancers.

---

## 2. Data Architecture, Schema & Quality Audit

### 2.1 Dataset Inventory
The training corpus (`fiverr_data.csv`) comprises **458,798 unique user records** (`user_id` ranging from `1` to `483,098`) and **53 columns**:
* **Target Column (`label`):** Binary integer (`int64`).
  * Legitimate Users (`label = 0`): `446,477` (`97.3145%`)
  * Spammers (`label = 1`): `12,321` (`2.6855%`)
  * Imbalance Ratio: `36.237 : 1`
* **Identifier Column (`user_id`):** Unique integer primary key (`458,798` unique values). Excluded from model features to prevent memorization and data leakage.
* **Anonymized Feature Matrix (`X1`–`X51`):** `50` integer columns (`int64`) and `1` floating-point column (`X13`, `float64` due to missing values) representing registration metadata (e.g., email length, username formatting, HTTP referrer category) and early session actions.

### 2.2 Data Quality & Anomaly Findings
1. **Missing Values (`NaN`):**
   * Exactly **6 missing values** exist across the entire dataset, all confined to feature **`X13`** (`0.0013%` of rows; `user_id` $\in \{22566, 88460, 112376, 279080, 292519, 380630\}$).
   * Remediation: Handled natively by tree-based histogram splitting and imputed via training-set median (`0.0`) inside linear/bagging pipelines to guarantee zero data leakage.
2. **Zero-Variance (Constant) Features:**
   * Seven binary columns have a constant value of `0` across all `458,798` rows: **`X27`, `X29`, `X30`, `X33`, `X46`, `X47`, and `X48`**.
   * Remediation: Deterministically dropped during preprocessing, reducing raw active features from `51` to `44`.
3. **Feature Cardinality & Distribution Skewness:**
   * **Binary Flags (`7` active features):** `X11`, `X12`, `X20`, `X45`, `X49`, `X50`, `X51` $\in \{0, 1\}$.
   * **Low-Cardinality Categorical / Ordinal Encodings (`8` features):** `X4` (3 unique), `X6` (4 unique), `X14` (6 unique), `X37` (6 unique), `X40` (7 unique), `X36` (9 unique), `X13` (10 unique), `X7` & `X35` (11 unique).
   * **High-Cardinality & Right-Skewed Behavioral/Metadata Features (`29` features):**
     * `X1`: Range `[1, 24234]`, Mean `10,850.59`, Std `7,078.66`, Median `11,574.0`
     * `X10`: Range `[1, 49637]`, Mean `20,368.21`, Std `22,837.17`, Median `4,378.5`
     * `X21`: Range `[1, 913]`, Mean `29.26`, Std `33.00`, Median `18.0`
     * `X19`: Range `[0, 213]`, Mean `0.305`, Std `3.55`, Median `0.0` (strongest linear predictor of spam, $r = +0.2558$)
     * `X25`: Range `[0, 688]`, Mean `0.783`, Std `2.85`, Median `0.0`
     * `X32`: Range `[0, 333]`, Mean `0.537`, Std `1.49`, Median `0.0`
4. **Multicollinearity Audit ($|r| > 0.80$):**
   * `X34` and `X35`: $r = 0.9717$
   * `X41` and `X44`: $r = 0.9312$
   * `X10` and `X20`: $r = 0.8617$
   * `X16` and `X18`: $r = 0.8440$

---

## 3. Feature Engineering & Preprocessing Specification

The custom transformer `FiverrFeatureEngineer` transforms the 51 raw inputs into **59 clean, high-signal features** (44 active raw features + 15 engineered features):

| Feature Group | Engineered Feature Name(s) | Mathematical Formulation | Domain Rationale |
| :--- | :--- | :--- | :--- |
| **Identifier & Constant Pruning** | Drop `user_id`, `X27`, `X29`, `X30`, `X33`, `X46`, `X47`, `X48` | $\mathbf{X}_{\text{clean}} = \mathbf{X} \setminus \{\text{constants}, \text{ID}\}$ | Prevents ID overfitting and eliminates zero-information dimensions. |
| **String / Length Structural Ratios** | `X16_X18_diff`, `X16_X18_ratio` | $X_{16} - X_{18}$, $\frac{X_{16} + 1}{X_{18} + 1}$ | Captures structural discrepancies between correlated string length parameters ($r = 0.8440$, e.g., email vs. username length). |
| **Profile / Registration Ratios** | `X21_X22_diff`, `X21_X22_ratio` | $X_{21} - X_{22}$, $\frac{X_{21} + 1}{X_{22} + 1}$ | Contrasts registration breadth (`X21`) against negatively correlated legitimacy indicator (`X22`, $r = -0.1378$). |
| **Collinear Signal Consolidation** | `X34_X35_sum`, `X41_X44_sum` | $X_{34} + X_{35}$, $X_{41} + X_{44}$ | Consolidates near-duplicate behavioral counters ($r = 0.9717$ and $r = 0.9312$) into variance-stabilized composite counts. |
| **Burst Activity Aggregations** | `total_activity_counts`, `nonzero_activity_dims` | $\sum_{j \in S} X_j$, $\sum_{j \in S} \mathbb{I}(X_j > 0)$ | Sums total event volume and active behavioral dimensions across 22 sparse action counters (`X13`–`X15`, `X19`, `X23`–`X26`, `X28`, `X31`–`X32`, `X34`–`X44`). |
| **Binary Flag Aggregation** | `binary_flags_sum` | $\sum_{k \in B} X_k$ | Aggregates active binary risk/verification indicators (`X11`, `X12`, `X20`, `X45`, `X49`, `X50`, `X51`). |
| **Skewness Stabilization** | `X1_log1p`, `X10_log1p`, `X19_log1p`, `X21_log1p`, `X25_log1p`, `X32_log1p` | $\ln(1 + \max(0, X_j))$ | Compresses extreme right tails for linear baselines and stabilizes gradient split binning. |

---

## 4. Model Selection, Methodology & Mathematical Formulation

### 4.1 Partitioning Strategy
To preserve the exact `2.6855%` positive spammer prevalence across all experimental phases, `fiverr_data.csv` is partitioned via **Stratified Random Splitting** (`random_state = 42`):
* **Training Set (70%):** `321,158` rows (`312,533` legitimate, `8,625` spammers)
* **Validation Set (15%):** `68,820` rows (`66,972` legitimate, `1,848` spammers) — used exclusively for hyperparameter selection and Precision-Recall probability threshold optimization.
* **Test Set (15%):** `68,820` rows (`66,972` legitimate, `1,848` spammers) — locked holdout set used once for final reporting.

### 4.2 Candidate Architectures Evaluated
1. **Model 1 — Cost-Sensitive Logistic Regression (Linear Baseline):**
   * Minimizes penalized weighted log-loss with $L_2$ regularization and inverse class frequency weighting ($w_1 \approx 18.62, w_0 \approx 0.514$).
   * Serves as an interpretable linear benchmark.
2. **Model 2 — Balanced Subsample Random Forest (Bagging Ensemble):**
   * Parallel ensemble of deep decision trees trained with bootstrap class balancing (`class_weight = 'balanced_subsample'`, `max_depth = 16`, `min_samples_leaf = 5`).
   * Captures non-linear feature interactions without gradient boosting sensitivity.
3. **Model 3 — Histogram-Based Gradient Boosted Decision Trees (`HistGradientBoostingClassifier` / LightGBM Architecture — Champion Model):**
   * Sequentially fits additive regression trees $F_M(\mathbf{x}) = \sum_{m=1}^{M} \eta h_m(\mathbf{x})$ to the negative binomial log-likelihood gradient using 255-bin feature histograms:
     $$\mathcal{L}(y, F(\mathbf{x})) = -\left[ y \ln \sigma(F(\mathbf{x})) + (1 - y) \ln(1 - \sigma(F(\mathbf{x}))) \right] + \frac{\lambda}{2} \sum_{j=1}^{T} w_j^2$$
   * **Champion Hyperparameters:** `max_iter = 300`, `learning_rate = 0.08`, `max_leaf_nodes = 63`, `min_samples_leaf = 30`, `l2_regularization = 1.0`, `early_stopping = True` (`validation_fraction = 0.1`, `n_iter_no_change = 15`).

### 4.3 Post-Processing & Threshold Calibration
Rather than distorting probability calibration via extreme synthetic oversampling, the champion `HistGradientBoostingClassifier` outputs well-calibrated posterior probabilities $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$. The optimal classification threshold $\tau^*$ is selected on the validation set to maximize the generalized $F_\beta$-score:
$$\tau^* = \arg\max_{\tau \in (0, 1)} \frac{(1 + \beta^2) \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\beta^2 \cdot \text{Precision}(\tau) + \text{Recall}(\tau)}$$
* For balanced Precision-Recall optimization ($\beta = 1.0$), $\tau^* \approx 0.34 - 0.38$.
* For security-prioritized recall ($\beta = 2.0$), $\tau_{\text{review}} \approx 0.15 - 0.20$.

---

## 5. Operational Risk Routing & Decision Matrix

In production, a single hard binary cutoff is suboptimal for marketplace operations. The model's calibrated probability score $\hat{p}$ drives a three-tier Trust & Safety workflow:

| Risk Tier | Probability Range ($\hat{p}$) | Automated Platform Action | Operational Goal |
| :--- | :--- | :--- | :--- |
| **Tier 1: Low Risk (Green)** | $\hat{p} < 0.15$ | **Auto-Allow:** Normal platform access. | Zero friction for `>95%` of legitimate buyers and freelancers. |
| **Tier 2: Medium Risk (Amber)** | $0.15 \le \hat{p} < \tau^*$ | **Step-Up Verification & Attachment Sandbox:** Require phone/2FA challenge; scan/quarantine outgoing job brief attachments in sandbox. | Intercepts borderline/stealth RAT & Agent Tesla campaigns without banning legitimate new buyers. |
| **Tier 3: High Risk (Red)** | $\hat{p} \ge \tau^*$ | **Auto-Block / Account Quarantine:** Suspend outbound messaging and attachment uploads immediately; route to T&S queue. | Immediately neutralizes high-confidence spammer and malware accounts. |

---

## 6. Model Governance, Monitoring & Production Maintenance

### 6.1 Explainability & Reason Codes
* Global feature importance is tracked via **Permutation Importance** on the held-out validation set (measuring PR-AUC drop upon feature shuffling).
* Dominant predictive drivers consistently include `X19` (and `X19_log1p`), `X4`, `X22`, `X11`, `X6`, `X36`, `X10_log1p`, `X16_X18_ratio`, and `total_activity_counts`.

### 6.2 Drift Detection & Retraining Triggers
Adversaries continuously adapt their registration scripts and attachment delivery tactics. Production monitoring computes weekly telemetry metrics:
1. **Population Stability Index (PSI) on Predicted Probabilities and Top 10 Features:**
   $$\text{PSI} = \sum_{b=1}^{B} \left( P_{\text{actual}, b} - P_{\text{expected}, b} \right) \ln \left( \frac{P_{\text{actual}, b}}{P_{\text{expected}, b}} \right)$$
   * **Alert Threshold:** $\text{PSI} > 0.10$ (Warning / Investigate), $\text{PSI} > 0.25$ (Critical Drift — Trigger Automated Retraining).
2. **Performance Decay Trigger:** If weekly validated Precision at $\tau^*$ drops by `> 5%` or false-negative malware reports increase by `> 10%`, initiate incremental retraining on the latest 90-day labeled window.