# streamlit_ruin_extended.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import time

# ---------------- Helper / Distribution functions ----------------
def run_app():
        st.header("Extended Ruin Theory - Finite Time Solvency")
        def generate_claim_numbers(dist_name, params, time_horizon):
            """Generates the number of claims for each time step (single draw)."""
            if dist_name == 'Poisson':
                return stats.poisson.rvs(mu=params['lambda'] * time_horizon, size=1)
            elif dist_name == 'Binomial':
                # n trials over the time horizon (interpreted as n per unit time * time_horizon)
                n_total = int(params['n'] * time_horizon)
                return stats.binom.rvs(n=n_total, p=params['p'], size=1)
            elif dist_name == 'Negative Binomial':
                # n is the number of successes, p is the probability of success
                return stats.nbinom.rvs(n=params['n'], p=params['p'], size=1)
            return np.array([0])

        def generate_claim_sizes(dist_name, params, num_claims):
            """Generates claim sizes for a given number of claims."""
            if num_claims == 0:
                return np.array([])
            if dist_name == 'Exponential':
                return stats.expon.rvs(scale=1/params['lambda_exp'], size=num_claims)
            elif dist_name == 'Erlang':
                # scale must be 1/rate
                return stats.erlang.rvs(a=int(params['shape']), scale=1/params['rate'], size=num_claims)
            elif dist_name == 'Gamma':
                return stats.gamma.rvs(a=params['shape_gamma'], scale=1/params['rate_gamma'], size=num_claims)
            elif dist_name == 'Pareto':
                return stats.pareto.rvs(b=params['shape_pareto'], scale=params['scale_pareto'], size=num_claims)
            return np.array([])

        # ---------------- Simulation core (single path) ----------------

        def run_single_simulation(initial_capital, premium_rate, time_horizon,
                                claim_num_dist, claim_num_params,
                                claim_size_dist, claim_size_params,
                                solvency_level=None, stop_at_ruin=True, rng=None):
            """
            Returns:
            - times: array of event times [0, claim1, claim2, ..., T]
            - surplus: array of surplus values at those times
            - hit_solvent: boolean, True if surplus ever < solvency_level within [0, T]
            - ruin_occurred: boolean, True if surplus ever < 0 (traditional ruin)
            """
            if rng is None:
                rng = np.random

            # Number of claims in [0, T]
            num_claims = int(generate_claim_numbers(claim_num_dist, claim_num_params, time_horizon)[0])

            if num_claims == 0:
                times = np.array([0.0, float(time_horizon)])
                surplus = np.array([initial_capital, initial_capital + premium_rate * time_horizon])
                hit_solvent = (solvency_level is not None) and (surplus.min() < solvency_level)
                return times, surplus, hit_solvent, False

            # claim times uniformly on [0,T]
            claim_times = np.sort(rng.uniform(0, time_horizon, num_claims))
            claim_sizes = generate_claim_sizes(claim_size_dist, claim_size_params, num_claims)

            times = np.concatenate(([0.0], claim_times, [float(time_horizon)]))
            surplus = np.zeros_like(times)
            surplus[0] = initial_capital

            ruin_occurred = False
            hit_solvent = False

            # process claims one by one
            for i in range(1, len(times)-1):  # upto last claim
                dt = times[i] - times[i-1]
                surplus[i] = surplus[i-1] + premium_rate * dt - claim_sizes[i-1]

                # check solvency threshold crossing
                if solvency_level is not None and surplus[i] < solvency_level:
                    hit_solvent = True

                if surplus[i] < 0:
                    ruin_occurred = True
                    if stop_at_ruin:
                        # set remaining times to nan (indicate stopped) or hold at negative value
                        surplus[i+1:] = np.nan
                        break
                    # else continue (allow recovery) - but classical ruin theory usually stops here

            # if we didn't break, compute surplus at final time (after possibly last claim)
            if not np.isnan(surplus[-1]):
                dt_end = times[-1] - times[-2]
                # if previous index was filled, compute final premium accrual
                surplus[-1] = surplus[-2] + premium_rate * dt_end
                if solvency_level is not None and surplus[-1] < solvency_level:
                    hit_solvent = True

            return times, surplus, hit_solvent, ruin_occurred

        # ---------------- Batched simulation helpers ----------------

        def simulate_many_paths(initial_capital, premium_rate, time_horizon,
                                claim_num_dist, claim_num_params,
                                claim_size_dist, claim_size_params,
                                num_paths=1000, solvency_level=None, max_trace_paths=50, rng=None, progress_callback=None):
            """
            Simulate num_paths independent sample paths.
            Returns:
            - list_of_paths: list of tuples (times, surplus) for up to max_trace_paths (used for plotting)
            - fraction_hit_solvency: fraction of paths that fell below solvency_level at any time in [0,T]
            - fraction_ruin: fraction of paths that hit classical ruin (<0)
            """
            if rng is None:
                rng = np.random

            hit_count = 0
            ruin_count = 0
            traces = []
            for i in range(num_paths):
                times, surplus, hit_solvent, ruin_occurred = run_single_simulation(
                    initial_capital, premium_rate, time_horizon,
                    claim_num_dist, claim_num_params,
                    claim_size_dist, claim_size_params,
                    solvency_level=solvency_level, stop_at_ruin=True, rng=rng
                )
                if hit_solvent:
                    hit_count += 1
                if ruin_occurred:
                    ruin_count += 1
                if i < max_trace_paths:
                    traces.append((times, surplus))
                if progress_callback is not None and i % max(1, num_paths // 20) == 0:
                    progress_callback(i / num_paths)

            if progress_callback is not None:
                progress_callback(1.0)

            return traces, hit_count / num_paths, ruin_count / num_paths

        def estimate_ruin_probability(initial_capital, premium_rate, time_horizon,
                                    claim_num_dist, claim_num_params,
                                    claim_size_dist, claim_size_params,
                                    num_simulations=1000, rng=None):
            """Wrapper to estimate classical ruin probability (surplus < 0)."""
            _, _, hit_fraction, ruin_fraction = None, None, None, None
            # reuse simulate_many_paths for speed/consistency (no solvency_level)
            _, _, _, = None, None, None
            _, _, ruin_frac = simulate_many_paths(
                initial_capital, premium_rate, time_horizon,
                claim_num_dist, claim_num_params,
                claim_size_dist, claim_size_params,
                num_paths=num_simulations, solvency_level=None, max_trace_paths=0, rng=rng
            )
            # simulate_many_paths doesn't return exactly that ordering; let's call and unpack properly:
            traces, hit_frac, ruin_frac = simulate_many_paths(
                initial_capital, premium_rate, time_horizon,
                claim_num_dist, claim_num_params,
                claim_size_dist, claim_size_params,
                num_paths=num_simulations, solvency_level=None, max_trace_paths=0, rng=rng
            )
            return ruin_frac

        # ---------------- Utility: find minimal capital u (RBC) for target ruin prob ----------------

        def find_min_capital_for_target_ruin(c, time_horizon, claim_num_dist, claim_num_params,
                                            claim_size_dist, claim_size_params,
                                            target_ruin_prob=0.05, u_min=0.0, u_max=10000.0,
                                            num_simulations=2000, tol=1.0, rng=None, max_iters=12, progress_callback=None):
            """
            Binary search on initial capital u in [u_min, u_max] to find minimal u s.t. ruin_prob(u) <= target_ruin_prob.
            tol: absolute tolerance in units of capital (stop when u_max - u_min < tol)
            max_iters: safety for binary search iterations
            Returns (u_est, ruin_prob_at_u)
            """
            if rng is None:
                rng = np.random

            low, high = u_min, u_max
            best_u = high
            best_prob = 1.0

            for it in range(max_iters):
                mid = 0.5 * (low + high)
                # estimate ruin prob at mid
                _, _, hit_frac, ruin_frac = None, None, None, None
                _, hit_frac, ruin_frac = simulate_many_paths(
                    mid, c, time_horizon,
                    claim_num_dist, claim_num_params,
                    claim_size_dist, claim_size_params,
                    num_paths=num_simulations, solvency_level=None, max_trace_paths=0, rng=rng
                )
                if progress_callback is not None:
                    progress_callback((it+1) / max_iters)
                if ruin_frac <= target_ruin_prob:
                    best_u = mid
                    best_prob = ruin_frac
                    high = mid
                else:
                    low = mid
                if (high - low) <= tol:
                    break

            return best_u, best_prob

        # ---------------- Streamlit UI ----------------

        st.set_page_config(layout="wide")
        st.title("Ruin Theory (Extended): Finite-time ruin, RBC, sample paths")

        st.markdown("""
        This extended app:
        - simulates many sample paths and counts how many fell below a *solvency level* (risk-based capital) within finite horizon T,  
        - draws sample paths (a subset for visualization),  
        - computes an RBC estimate (minimal initial capital u so that ruin prob ≤ target),  
        - performs a quick grid search over (u, c) to find feasible combinations meeting a target ruin probability.
        """)

        # --- Sidebar inputs ---
        with st.sidebar:
            st.header("Model Parameters")

            initial_capital = st.slider("Initial Capital (u)", 0, 20000, 1000, 100)
            premium_rate = st.number_input("Average Premium Rate (c)", 0.1, 200.0, 20.0, 0.1)
            time_horizon = st.number_input("Time Horizon (T)", 1, 200, 10, 1)

            st.markdown("---")
            claim_num_dist = st.selectbox("Claim Number Distribution", ['Poisson', 'Binomial', 'Negative Binomial'])
            claim_num_params = {}
            if claim_num_dist == 'Poisson':
                claim_num_params['lambda'] = st.slider("Lambda (λ) - Avg. claims per unit time", 0.1, 20.0, 1.5, 0.1)
            elif claim_num_dist == 'Binomial':
                claim_num_params['n'] = st.slider("n (trials per unit time)", 1, 200, 20)
                claim_num_params['p'] = st.slider("p (probability of claim)", 0.01, 1.0, 0.1)
            elif claim_num_dist == 'Negative Binomial':
                claim_num_params['n'] = st.slider("n (number of successes)", 1, 200, 5)
                claim_num_params['p'] = st.slider("p (probability of success)", 0.01, 1.0, 0.5)

            st.markdown("---")
            claim_size_dist = st.selectbox("Claim Size Distribution", ['Exponential', 'Erlang', 'Gamma', 'Pareto'])
            claim_size_params = {}
            if claim_size_dist == 'Exponential':
                claim_size_params['lambda_exp'] = st.slider("Lambda (λ) - Rate", 0.01, 5.0, 0.1, 0.01)
            elif claim_size_dist == 'Erlang':
                claim_size_params['shape'] = st.slider("Shape (k)", 1, 50, 2)
                claim_size_params['rate'] = st.slider("Rate (λ)", 0.1, 10.0, 1.0, 0.1)
            elif claim_size_dist == 'Gamma':
                claim_size_params['shape_gamma'] = st.slider("Shape (α)", 0.1, 20.0, 2.0, 0.1)
                claim_size_params['rate_gamma'] = st.slider("Rate (β) - scale param in rvs", 0.01, 20.0, 1.0, 0.01)
            elif claim_size_dist == 'Pareto':
                claim_size_params['shape_pareto'] = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1)
                claim_size_params['scale_pareto'] = st.slider("Scale (x_m)", 1.0, 500.0, 10.0, 1.0)

            st.markdown("---")
            num_paths = st.select_slider("Number of sample paths (for counting & plotting)", options=[100, 200, 500, 1000, 2000], value=500)
            max_plot_paths = st.slider("Max paths to draw (for visualization)", 1, 200, 50, 1)
            num_sim_for_rbc = st.select_slider("Simulations used for RBC estimate", options=[200, 500, 1000, 2000, 5000], value=1000)
            solvency_level = st.number_input("Solvency/Risk-based capital (RBC) level (show crossing if < this)", 0.0, 200000.0, float(initial_capital), 100.0)

            st.markdown("---")
            st.subheader("Optimization / Search")
            target_ruin_prob = st.slider("Target finite-time ruin probability", 0.001, 0.5, 0.05, 0.001)
            search_do = st.checkbox("Run grid search to find minimal capital for various c", value=True)
            u_search_max = st.number_input("Max capital to consider in search", 1000, 100000, 20000, 1000)
            c_search_min = st.number_input("c search min", 0.1, 200.0, max(0.5, premium_rate*0.5), 0.1)
            c_search_max = st.number_input("c search max", 0.1, 500.0, premium_rate*1.5, 0.1)
            c_grid_points = st.slider("c grid points (for search)", 3, 20, 7, 1)

            st.markdown("---")
            run_button = st.button("Run Extended Simulation")

        # ---------------- Main panel ----------------

        if run_button:
            st.header("Simulating sample paths and counting solvency breaches...")

            progress = st.progress(0.0)
            status = st.empty()

            def progress_cb(p):
                try:
                    progress.progress(min(1.0, float(p)))
                except:
                    pass

            start_time = time.time()
            traces, frac_below_solv, frac_ruin = simulate_many_paths(
                initial_capital, premium_rate, time_horizon,
                claim_num_dist, claim_num_params,
                claim_size_dist, claim_size_params,
                num_paths=num_paths, solvency_level=solvency_level, max_trace_paths=max_plot_paths, rng=np.random, progress_callback=progress_cb
            )
            elapsed = time.time() - start_time
            status.markdown(f"Simulated {num_paths} paths in {elapsed:.2f}s — fraction below solvency level: **{frac_below_solv:.3f}**, classical ruin fraction: **{frac_ruin:.3f}**")
            progress.progress(1.0)

            # --- Plot sample paths (subset) ---
            st.subheader("Sample paths (subset)")
            fig = px.line()
            for idx, (times, surplus) in enumerate(traces):
                # build data frame for this path
                dfp = pd.DataFrame({'Time': times, 'Surplus': surplus})
                fig.add_scatter(x=dfp['Time'], y=dfp['Surplus'], mode='lines', name=f'path_{idx+1}', showlegend=False, opacity=0.6)
            # horizontal markers
            fig.add_hline(y=0, line_dash="dash", annotation_text="Ruin (0)")
            fig.add_hline(y=solvency_level, line_dash="dot", annotation_text=f"Solvency Level ({solvency_level})")
            fig.update_layout(title="Sample surplus paths", xaxis_title="Time", yaxis_title="Surplus")
            st.plotly_chart(fig, use_container_width=True)

            # --- Summary metrics ---
            st.subheader("Finite-time Solvency Summary")
            st.write(f"- Number of simulated paths: **{num_paths}**")
            st.write(f"- Fraction of paths that went below solvency level (within T): **{frac_below_solv:.4f}**")
            st.write(f"- Fraction of paths that hit classical ruin (U(t) < 0) within T: **{frac_ruin:.4f}**")

            # --- Compute RBC: minimal u for given c so ruin_prob <= target_ruin_prob ---
            st.subheader("Estimate Risk-Based Capital (RBC) — minimal initial capital u s.t. ruin_prob(u) ≤ target")

            with st.spinner("Searching for minimal capital (binary search)..."):
                r_progress = st.progress(0.0)
                def rcb_progress_cb(p): r_progress.progress(min(1.0, float(p)))
                u_est, u_prob = find_min_capital_for_target_ruin(
                    premium_rate, time_horizon,
                    claim_num_dist, claim_num_params,
                    claim_size_dist, claim_size_params,
                    target_ruin_prob=target_ruin_prob,
                    u_min=0.0, u_max=u_search_max,
                    num_simulations=num_sim_for_rbc, tol=10.0, rng=np.random, max_iters=12, progress_callback=rcb_progress_cb
                )
            st.write(f"- Estimated minimal capital u (RBC) ≈ **{u_est:.1f}**, with estimated ruin prob **{u_prob:.4f}** (target: {target_ruin_prob})")

            # --- Grid search over c to find minimal u for each c (optional) ---
            if search_do:
                st.subheader("Grid search over premium rate (c) to find required u for target ruin probability")
                c_grid = np.linspace(c_search_min, c_search_max, int(c_grid_points))
                results = []
                outer_progress = st.progress(0.0)
                for idx, c_val in enumerate(c_grid):
                    u_found, prob_at_u = find_min_capital_for_target_ruin(
                        c_val, time_horizon,
                        claim_num_dist, claim_num_params,
                        claim_size_dist, claim_size_params,
                        target_ruin_prob=target_ruin_prob,
                        u_min=0.0, u_max=u_search_max,
                        num_simulations=int(max(200, num_sim_for_rbc//2)),
                        tol=50.0, rng=np.random, max_iters=8, progress_callback=None
                    )
                    results.append((c_val, u_found, prob_at_u))
                    outer_progress.progress((idx+1)/len(c_grid))
                df_results = pd.DataFrame(results, columns=['c', 'min_u_for_target', 'ruin_prob_at_u'])
                st.write("Grid search results (coarse):")
                st.dataframe(df_results)

                # plot min_u vs c
                fig2 = px.line(df_results, x='c', y='min_u_for_target', title="Required Initial Capital (u) vs Premium Rate (c)")
                fig2.update_layout(xaxis_title="Premium Rate c", yaxis_title="Minimal Capital u to meet target")
                st.plotly_chart(fig2, use_container_width=True)

                # choose best (lowest u) combination
                best_row = df_results.loc[df_results['min_u_for_target'].idxmin()]
                st.write(f"Best (minimal u) found at c = **{best_row['c']:.3f}**, required u ≈ **{best_row['min_u_for_target']:.1f}**, ruin prob ≈ **{best_row['ruin_prob_at_u']:.4f}**")

            st.success("Extended simulation completed.")

        else:
            st.info("Adjust parameters and click 'Run Extended Simulation' to run the extended analysis.")
