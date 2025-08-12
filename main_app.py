import streamlit as st
import ruin_app       # this is your original ruin theory code file
import app_finite_ruin  # this is your finite ruin probability extended code file

# Sidebar selector
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Original Ruin Theory", "Extended Finite Ruin Probability"])

if page == "Original Ruin Theory":
    ruin_app.run_app()

elif page == "Extended Finite Ruin Probability":
    app_finite_ruin.run_app()
