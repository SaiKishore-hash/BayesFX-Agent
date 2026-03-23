import streamlit as st
import yfinance as yf
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
st.set_page_config(page_title="BayesFX Agent", layout="wide")

# Sidebar
st.sidebar.header("Controls")
currency = st.sidebar.selectbox(
    "Select Currency Pair",
    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
)
days = st.sidebar.slider("Lookback Window (days)", 50, 500, 100)
st.sidebar.markdown("---")
run_model_btn = st.sidebar.button("Run Analysis")
st.write("Button value:", run_model_btn)
if "model_run" not in st.session_state:
    st.session_state.model_run = False

# Loading data
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2020-01-01", end="2024-01-01")
ticker = currency
data = load_data(ticker)
prices = data["Close"][ticker]
returns = np.log(prices / prices.shift(1)).dropna().tail(days)

tab1, tab2 = st.tabs(["Agent", "How it Works"])

# Separating tabs
with tab1: # Agent Decision
    st.title("BayesFX Agent")
    st.caption("Probabilistic FX Decision Engine using Bayesian Inference")

    st.markdown("## Market Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.line_chart(prices.tail(200))
        st.caption("Price movement")

    with col2:
        rolling_vol = returns.rolling(20).std()
        st.line_chart(rolling_vol)
        st.caption("Rolling volatility (20-day)")

    @st.cache_resource
    def run_model(returns):
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=0.1)
            sigma = pm.HalfNormal("sigma", sigma=0.1)
            nu = pm.Exponential("nu", 1/10)

            obs = pm.StudentT("obs", mu=mu, sigma=sigma, nu=nu, observed=returns)

            trace = pm.sample(
                500,
                chains=1,
                cores=1,
                return_inferencedata=True,
                progressbar=False
            )
        return trace

    # Bayesian model
    if run_model_btn:
        st.session_state.model_run = True
    if st.session_state.model_run:
        with st.spinner("Running Bayesian model..."):
            trace = run_model(returns)
    else:
        st.info("Click 'Run Analysis' to generate results")
        st.stop()

    # Extract values
    mu_mean = trace.posterior["mu"].mean().item()
    mu_std = trace.posterior["mu"].std().item()
    sigma_mean = trace.posterior["sigma"].mean().item()
    nu_mean = trace.posterior["nu"].mean().item()

    # Show stats
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mean (μ)", f"{mu_mean:.6f}")
    col2.metric("Uncertainty (μ std)", f"{mu_std:.6f}")
    col3.metric("Volatility (σ)", f"{sigma_mean:.6f}")
    col4.metric("Tail Risk (ν)", f"{nu_mean:.2f}")

    # Plot posterior
    # st.subheader("Posterior Distributions")
    # fig = az.plot_posterior(trace)
    # st.pyplot(fig)
    az.plot_posterior(trace)
    fig = plt.gcf()
    st.pyplot(fig)

    st.markdown("## Model Insights")
    st.markdown("### Interpretation")
    if abs(mu_mean) < mu_std:
        st.info("Market has no clear directional bias. Returns are dominated by noise.")
    else:
        st.info("Market shows a directional bias, but confidence should be evaluated carefully.")

    if sigma_mean > 0.005:
        st.warning("Market is volatile. Risk management is critical.")
    else:
        st.success("Market volatility is within normal range.")

    if nu_mean < 5:
        st.error("High tail risk detected. Extreme events are more likely.")
    else:
        st.success("Tail risk is moderate. Extreme events less likely.")
    
    # Agent decision
    st.markdown("## Agent Recommendation")
    confidence = abs(mu_mean) / mu_std
    st.metric("Signal Strength", f"{confidence:.2f}")

    if confidence < 1:
        st.warning("Weak signal → Stay out of market")
    elif confidence < 2:
        st.info("Moderate signal → Trade small")
    else:
        if mu_mean > 0:
            st.success("Strong signal → LONG")
        else:
            st.error("Strong signal → SHORT")
    
    # Market Regime
    st.markdown("## Market Regime")
    if sigma_mean > 0.006:
        st.error("High Volatility Regime")
    elif sigma_mean > 0.004:
        st.warning("Moderate Volatility Regime")
    else:
        st.success("Low Volatility Regime")
    
    # Risk Summary Box
    st.markdown("## Risk Summary")
    st.write(f"""
    - Expected Return (μ): {mu_mean:.6f}
    - Volatility (σ): {sigma_mean:.4f}
    - Tail Risk (ν): {nu_mean:.2f}

    Interpretation:
    - Directional edge is {"weak" if abs(mu_mean) < mu_std else "present"}
    - Volatility is {"high" if sigma_mean > 0.005 else "normal"}
    - Tail risk is {"elevated" if nu_mean < 5 else "moderate"}
    """)
    


with tab2:
    st.title("BayesFX Agent")

    st.markdown("## 1. The Problem")
    st.write("""
    Financial markets are noisy and unpredictable.
    Instead of predicting exact prices, we aim to model uncertainty and risk.
    """)

    st.markdown("## 2. From Prices to Returns")

    # Price chart
    st.subheader("Price Series")
    st.line_chart(prices.tail(200))

    # Returns chart
    st.subheader("Log Returns")
    st.line_chart(returns.tail(200))
    st.info("""
    Prices trend over time, but returns fluctuate around zero.
    This is why we model returns instead of prices.
    """)

    st.markdown("""
    ### Why Log Returns?

    Prices grow multiplicatively (e.g., 100 → 110 → 121),
    but statistical models work better with additive quantities.

    Log returns convert multiplicative changes into additive ones,
    making them easier to model and analyze.
    """)

    st.subheader("Distribution of Returns")
    fig, ax = plt.subplots()
    ax.hist(returns, bins=50, density=True)
    ax.set_title("Histogram of Returns")
    st.pyplot(fig)

    st.info("""
    Most returns are small, but extreme events occur more often than expected.
    This is why normal distribution is insufficient.
    """)

    from scipy.stats import norm

    st.subheader("Normal Distribution Fit")

    mu_mle = returns.mean()
    sigma_mle = returns.std()

    x = np.linspace(returns.min(), returns.max(), 1000)
    y = norm.pdf(x, mu_mle, sigma_mle)

    fig, ax = plt.subplots()
    ax.hist(returns, bins=50, density=True, alpha=0.6)
    ax.plot(x, y, color='red')
    ax.set_title("Normal Fit vs Actual Data")

    st.pyplot(fig)

    st.markdown("""
    ### Why Normal Distribution Fails

    The normal distribution underestimates extreme events.
    In real markets, large moves happen more frequently than expected.

    This is called **fat tails**.
    """)

    st.markdown("""
    ### Solution: Student-t Distribution

    We use a Student-t distribution instead of a normal distribution.

    It has an additional parameter (ν) that controls tail thickness.
    Lower ν → more extreme events.
    """)

    st.subheader("Volatility Over Time")
    rolling_vol = returns.rolling(20).std()
    st.line_chart(rolling_vol)

    st.markdown("""
    ### Bayesian Inference

    Instead of estimating a single value for parameters,
    we estimate a distribution.

    This means:
    - μ is not a number, but a range
    - σ is not fixed, but uncertain

    This allows better decision-making under uncertainty.
    """)

    st.subheader("Posterior Distributions")
    az.plot_posterior(trace)
    fig = plt.gcf()
    st.pyplot(fig)

    st.info("""
    The model shows uncertainty in mean returns, but relatively stable volatility.
    This reflects real market behavior.
    """)