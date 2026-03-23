import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pymc as pm
import arviz as az

# Step 1: get data
ticker = "EURUSD=X"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Step 2: clean prices
prices = data["Close"][ticker]

# Step 3: returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna().tail(100)

# Step 4: estimate parameters
mu = returns.mean()
sigma = returns.std()

print("Mean (mu):", mu)
print("Standard Deviation (sigma):", sigma)

# Step 5: histogram (density!)
plt.figure()
plt.hist(returns, bins=50, density=True, alpha=0.6)

# Step 6: Normal curve
x = np.linspace(returns.min(), returns.max(), 1000)
y = norm.pdf(x, mu, sigma)

plt.plot(x, y)

plt.title("Histogram vs Normal Distribution")
plt.xlabel("Returns")
plt.ylabel("Density")

plt.show()

rolling_vol = returns.rolling(20).std()

st.subheader("Rolling Volatility (20-day)")
st.line_chart(rolling_vol)

# Step 7: Bayesian model
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=0.1)
    sigma = pm.HalfNormal("sigma", sigma=0.1)
    nu = pm.Exponential("nu", 1/10)
    obs = pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=returns)
    trace = pm.sample(1000, return_inferencedata=True, progressbar=False)

# Print summary
print(pm.summary(trace))

az.plot_posterior(trace)
plt.show()

# Step 10: Simple decision logic

st.subheader("Agent Decision")

signal_strength = abs(mu_mean) / mu_std

if signal_strength < 1:
    st.warning("Weak signal → Avoid trading")
elif signal_strength < 2:
    st.info("Moderate signal → Small position")
else:
    if mu_mean > 0:
        st.success("Strong positive signal → LONG")
    else:
        st.error("Strong negative signal → SHORT")

# Volatility + tail risk
if sigma_mean > 0.005:
    st.warning("High volatility → Reduce risk")

if nu_mean < 5:
    st.error("High tail risk → Extreme events likely")
else:
    st.success("Tail risk under control")