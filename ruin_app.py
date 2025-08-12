import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats


def run_app():
        st.header("Original Ruin Theory Model")
# --- Helper Functions for Distributions ---
        def generate_claim_numbers(dist_name, params, time_horizon):
            """Generates the number of claims for each time step."""
            if dist_name == 'Poisson':
                # For a classical process, we simulate the number of events in the interval [0, T]
                return stats.poisson.rvs(mu=params['lambda'] * time_horizon, size=1)
            elif dist_name == 'Binomial':
                # n trials over the time horizon
                return stats.binom.rvs(n=params['n'] * time_horizon, p=params['p'], size=1)
            elif dist_name == 'Negative Binomial':
                # n is the number of successes, p is the probability of success
                return stats.nbinom.rvs(n=params['n'], p=params['p'], size=1)
            return np.array([0]) # Default case

        def generate_claim_sizes(dist_name, params, num_claims):
            """Generates the sizes of the claims."""
            if num_claims == 0:
                return np.array([])
            if dist_name == 'Exponential':
                return stats.expon.rvs(scale=1/params['lambda_exp'], size=num_claims)
            elif dist_name == 'Erlang':
                # In scipy, the shape parameter is 'a' and scale is 'scale'
                return stats.erlang.rvs(a=params['shape'], scale=params['rate'], size=num_claims)
            elif dist_name == 'Gamma':
                return stats.gamma.rvs(a=params['shape_gamma'], scale=1/params['rate_gamma'], size=num_claims)
            elif dist_name == 'Pareto':
                # In scipy, 'b' is the shape parameter
                return stats.pareto.rvs(b=params['shape_pareto'], scale=params['scale_pareto'], size=num_claims)
            return np.array([]) # Default case


        # --- Simulation Core ---

        def run_single_simulation(initial_capital, premium_rate, time_horizon, claim_num_dist, claim_num_params, claim_size_dist, claim_size_params):
            """Runs a single path of the surplus process."""
            num_claims = generate_claim_numbers(claim_num_dist, claim_num_params, time_horizon)[0]
            
            if num_claims == 0:
                times = np.array([0, time_horizon])
                surplus = np.array([initial_capital, initial_capital + premium_rate * time_horizon])
                return times, surplus, False # No ruin if no claims

            claim_times = np.sort(np.random.uniform(0, time_horizon, num_claims))
            claim_sizes = generate_claim_sizes(claim_size_dist, claim_size_params, num_claims)

            times = np.concatenate(([0], claim_times, [time_horizon]))
            
            # Calculate surplus at each time point
            surplus = np.zeros_like(times)
            surplus[0] = initial_capital
            
            cumulative_claims = 0
            ruin_occurred = False

            for i in range(1, len(claim_times) + 1):
                time_diff = times[i] - times[i-1]
                surplus[i] = surplus[i-1] + premium_rate * time_diff - claim_sizes[i-1]
                if surplus[i] < 0:
                    ruin_occurred = True
            
            # Surplus at the end of the horizon
            time_diff_end = times[-1] - times[-2]
            surplus[-1] = surplus[-2] + premium_rate * time_diff_end

            return times, surplus, ruin_occurred


        def estimate_ruin_probability(initial_capital, premium_rate, time_horizon, claim_num_dist, claim_num_params, claim_size_dist, claim_size_params, num_simulations):
            """Estimates the ruin probability by running multiple simulations."""
            ruin_count = 0
            for _ in range(num_simulations):
                _, _, ruin_occurred = run_single_simulation(initial_capital, premium_rate, time_horizon, claim_num_dist, claim_num_params, claim_size_dist, claim_size_params)
                if ruin_occurred:
                    ruin_count += 1
            return ruin_count / num_simulations


        # --- Streamlit App UI ---

        st.set_page_config(layout="wide")

        st.title("Ruin Theory: A Monte Carlo Simulation 🎲")
        st.markdown("""
        This application simulates a classical risk process, allowing you to explore the evolution of an insurer's surplus over time and estimate the probability of ruin.
        Adjust the parameters in the sidebar to see how they affect the outcomes.
        """)

        # --- Sidebar for Inputs ---
        with st.sidebar:
            st.header("Model Parameters")

            initial_capital = st.slider("Initial Capital (u)", 100, 10000, 1000, 100)
            premium_rate = st.number_input("Average Premium Rate (c)", 1.0, 100.0, 20.0, 0.5)
            time_horizon = st.number_input("Time Horizon (T)", 1, 100, 10, 1)
            
            st.markdown("---")
            
            # Claim Number Distribution
            claim_num_dist = st.selectbox("Claim Number Distribution", ['Poisson', 'Binomial', 'Negative Binomial'])
            claim_num_params = {}
            if claim_num_dist == 'Poisson':
                claim_num_params['lambda'] = st.slider("Lambda (λ) - Avg. claims per unit time", 0.1, 10.0, 1.5, 0.1)
            elif claim_num_dist == 'Binomial':
                claim_num_params['n'] = st.slider("n (trials per unit time)", 1, 100, 20)
                claim_num_params['p'] = st.slider("p (probability of claim)", 0.01, 1.0, 0.1)
            elif claim_num_dist == 'Negative Binomial':
                claim_num_params['n'] = st.slider("n (number of successes)", 1, 100, 5)
                claim_num_params['p'] = st.slider("p (probability of success)", 0.01, 1.0, 0.5)

            st.markdown("---")

            # Claim Size Distribution
            claim_size_dist = st.selectbox("Claim Size Distribution", ['Exponential', 'Erlang', 'Gamma', 'Pareto'])
            claim_size_params = {}
            if claim_size_dist == 'Exponential':
                claim_size_params['lambda_exp'] = st.slider("Lambda (λ) - Rate", 0.01, 2.0, 0.1, 0.01)
            elif claim_size_dist == 'Erlang':
                claim_size_params['shape'] = st.slider("Shape (k)", 1, 50, 2)
                claim_size_params['rate'] = st.slider("Rate (λ)", 0.1, 10.0, 1.0, 0.1)
            elif claim_size_dist == 'Gamma':
                claim_size_params['shape_gamma'] = st.slider("Shape (α)", 0.1, 20.0, 2.0, 0.1)
                claim_size_params['rate_gamma'] = st.slider("Rate (β)", 0.1, 20.0, 0.1, 0.1)
            elif claim_size_dist == 'Pareto':
                claim_size_params['shape_pareto'] = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1)
                claim_size_params['scale_pareto'] = st.slider("Scale (x_m)", 1.0, 100.0, 10.0, 1.0)
            
            st.markdown("---")
            num_simulations_ruin = st.select_slider("Number of Simulations (for ruin probability)", options=[100, 500, 1000, 5000, 10000], value=1000)
            
            run_button = st.button("Run Simulation")

        # --- Main Panel for Outputs ---
        if run_button:
            tab1, tab2, tab3 = st.tabs(["Surplus Process Plot", "Ruin Probability vs. Capital", "Ruin Probability Table"])

            # --- Tab 1: Surplus Process Plot ---
            with tab1:
                st.subheader("Single Realization of the Surplus Process")
                
                with st.spinner("Running a single simulation..."):
                    times, surplus, ruin_occurred = run_single_simulation(initial_capital, premium_rate, time_horizon, claim_num_dist, claim_num_params, claim_size_dist, claim_size_params)
                    
                    df_surplus = pd.DataFrame({'Time': times, 'Surplus': surplus})
                    
                    fig = px.line(df_surplus, x='Time', y='Surplus', title='Surplus vs. Time', markers=True)
                    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Ruin Level")
                    fig.update_layout(xaxis_title="Time (t)", yaxis_title="Surplus U(t)")
                    st.plotly_chart(fig, use_container_width=True)

                    if ruin_occurred:
                        st.warning("Ruin occurred in this simulation run.")
                    else:
                        st.success("No ruin occurred in this simulation run.")

            # --- Tab 2: Ruin Probability vs. Initial Capital ---
            with tab2:
                st.subheader("Estimated Ruin Probability vs. Initial Capital")
                
                with st.spinner(f"Estimating ruin probabilities over {num_simulations_ruin} simulations for each capital level..."):
                    capital_levels = np.linspace(0, initial_capital * 2, 10)
                    ruin_probs = []
                    
                    progress_bar = st.progress(0)
                    for i, u in enumerate(capital_levels):
                        prob = estimate_ruin_probability(u, premium_rate, time_horizon, claim_num_dist, claim_num_params, claim_size_dist, claim_size_params, num_simulations_ruin)
                        ruin_probs.append(prob)
                        progress_bar.progress((i + 1) / len(capital_levels))

                    df_ruin = pd.DataFrame({'Initial Capital': capital_levels, 'Ruin Probability': ruin_probs})
                    
                    fig_ruin = px.line(df_ruin, x='Initial Capital', y='Ruin Probability', title='Ruin Probability vs. Initial Capital', markers=True)
                    fig_ruin.update_layout(xaxis_title="Initial Capital (u)", yaxis_title="Estimated Ruin Probability ψ(u)")
                    st.plotly_chart(fig_ruin, use_container_width=True)

            # --- Tab 3: Ruin Probability Table ---
            with tab3:
                st.subheader("Ruin Probability Data")
                st.dataframe(df_ruin, use_container_width=True)

        else:
            st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")

