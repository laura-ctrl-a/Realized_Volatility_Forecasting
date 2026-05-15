# 📈 Realized Volatility Forecasting — BTC/USD

> Comparing HAR-family econometric models and deep learning architectures (LSTM, GRU, CNN-1D, TCN) for next-day realized volatility forecasting on Bitcoin/USD high-frequency data.

---

## 🔍 Overview

This project benchmarks **15 forecasting models** across two paradigms:

- **Econometric (HAR-family):** HAR, log-HAR, LevHAR, HAR-CJ, LevHAR-CJ, SHAR, HAR-L, HAR-Q
- **Deep Learning:** LSTM (base & robust), GRU (base & robust), CNN-1D (base & robust), TCN

Model accuracy is evaluated via **MSE** and statistical significance is assessed through the **Diebold-Mariano (DM) test**, providing a rigorous pairwise comparison framework.

---

## 📁 Repository Structure

```
btcusd-rv-forecasting/
│
├── btcusd_rv_forecasting.ipynb   # Main analysis notebook
├── dm_matrix_btcusd.csv          # DM test statistics (pairwise, all models)
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Models

### HAR-family (OLS, statsmodels)

| Model | Key Features |
|---|---|
| HAR | RV lags: daily, weekly, monthly |
| log-HAR | Log-transformed RV lags |
| LevHAR | HAR + negative return leverage |
| HAR-CJ | Separates continuous (BPV) and jump (J = RV − BPV) components |
| LevHAR-CJ | HAR-CJ + leverage effect |
| SHAR | Signed semi-variance (RSV⁺, RSV⁻) |
| HAR-L | Long-memory extension (2M, 3M, 6M lags) |
| HAR-Q | Realized quarticity correction (HARQ term) |

### Deep Learning (TensorFlow/Keras)

| Model | Architecture |
|---|---|
| LSTM base | 1-layer LSTM(64) |
| LSTM robust | 2-layer LSTM(64→32) + Dropout |
| GRU base | 1-layer GRU(64) |
| GRU robust | 2-layer GRU(64→32) + Dropout |
| CNN-1D base | Conv1D(64) + MaxPool + Dense |
| CNN-1D robust | 2× Conv1D(128→64) + Dropout + Dense |
| TCN | Dilated causal Conv (dilations: 1,2,4,8,16) |

---

## ⚙️ Methodology

### Data Pipeline

- **Asset:** BTC/USD daily realized volatility
- **Features:** RV, BPV, RSV⁺/RSV⁻, RQ, nRQ, returns, absolute returns, last prices
- **Missing values:** Smart fill strategy — linear interpolation for gaps ≤ 5 days, drop otherwise

### Feature Engineering

Rich feature set derived from raw RV:
- Lagged RV (1d, 1w, 1m, 2m, 3m, 6m)
- Log-RV lags
- Negative return components (LevHAR)
- Continuous/jump decomposition (BPV, J)
- Signed semi-variances (RSV⁺, RSV⁻)
- Realized quarticity terms (RQ, HARQ)

### Train/Val/Test Split (chronological)

```
|--- 70% Train ---|-- 10% Val --|---- 20% Test ----|
```

### Deep Learning Specifics

- **Sliding window:** 20 days lookback, multivariate input
- **Scaling:** StandardScaler fitted on train only
- **Callbacks:** EarlyStopping (patience=8) + ReduceLROnPlateau (patience=4)
- **Optimizer:** Adam (lr=1e-3 for robust variants)

### Evaluation

- **Loss function:** MSE
- **Statistical test:** Diebold-Mariano (DM) with Newey-West HAC correction
- **Alignment:** HAR models produce 356 predictions vs. 382 for NNs — test sets are aligned to 356 observations for fair comparison

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/btcusd-rv-forecasting.git
cd btcusd-rv-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your data

Place your `btcusd_df.csv` file (semicolon-separated, columns below) inside a local folder and update the `pathcrypto` variable in the notebook:

```
dates; LastPrices; LastReturns; RV; RQ; nRQ; AbsRet; BPV; RSVp; RSVn; Returns
```

> **Note:** The original data was loaded from Google Drive (Colab). For local execution, replace the `drive.mount` cell with a local path.

### 4. Run the notebook

```bash
jupyter notebook btcusd_rv_forecasting.ipynb
```

---

## 📦 Requirements

```
tensorflow>=2.12
keras-tcn
scikit-learn
statsmodels
scipy
pandas
numpy
matplotlib
jupyter
```

See `requirements.txt` for pinned versions.

---

## 📊 Key Outputs

| Output | Description |
|---|---|
| `dm_matrix_btcusd.csv` | 15×15 matrix of DM statistics (positive = row model better) |
| Training curves | LSTM base learning curve (loss vs val_loss) |
| Metrics table | MSE per model on test set |

---

## 🎓 Academic Context

This notebook is part of a Master's thesis in **Data Analytics for Business and Society** (LM-91) at Ca' Foscari University of Venice, focusing on realized volatility modelling for cryptocurrency markets.

The work extends the classical HAR framework (Corsi, 2009) with modern deep learning architectures, evaluating whether added model complexity translates into statistically significant forecast gains.

---

## 📄 References

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics.
- Barndorff-Nielsen, O.E. & Shephard, N. (2004). *Power and Bipower Variation*.
- Diebold, F.X. & Mariano, R.S. (1995). *Comparing Predictive Accuracy*. JBES.
- Patton, A.J. & Sheppard, K. (2015). *Good Volatility, Bad Volatility*. RFS.

---

## 📝 License

This project is released for academic and educational purposes. Data not included.
