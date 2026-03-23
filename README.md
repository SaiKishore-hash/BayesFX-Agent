# BayesFX Agent

A probabilistic FX decision engine that models uncertainty using Bayesian inference and generates risk-aware trading insights.

---

## Overview

Financial markets are inherently noisy and unpredictable in direction. Instead of predicting exact price movements, this project focuses on modeling uncertainty and risk.

The system estimates distributions for:

* Mean return (μ)
* Volatility (σ)
* Tail risk (ν)

and converts these into actionable decisions.

---

## Key Features

* Bayesian inference using PyMC
* Student-t distribution to model fat tails
* Rolling volatility estimation
* Multi-currency support
* Interactive dashboard using Streamlit
* Agent-based decision system

---

## Why This Project Matters

Traditional models assume normal distributions and fixed parameters, which underestimate extreme events.

This system:

* Captures real-world fat tails
* Models uncertainty instead of point estimates
* Provides probabilistic decision-making

---

## How It Works

1. Collect FX price data
2. Convert prices into log returns
3. Fit a Bayesian Student-t model
4. Estimate posterior distributions
5. Generate decisions based on:

   * Signal strength
   * Volatility
   * Tail risk

---

## Tech Stack

* Python
* PyMC (Bayesian modeling)
* NumPy / Pandas
* Matplotlib
* Streamlit

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements

* Live data streaming
* Portfolio-level risk modeling
* Multi-asset support
* Reinforcement learning-based decision layer

---

## Author

Sai Kishore
