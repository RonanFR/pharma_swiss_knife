import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

import plotly.express as px


if __name__ == '__main__':
    # Data
    # -----------------------------------------------
    df = pd.DataFrame({
        "time": np.array([4, 34, 71, 89, 109, 130, 151]),
        "concentration": np.array([44651, 19891, 12373, 10474, 8795, 7010, 6058]),
        "used_for_fitting": [True, True, False, True, True, True, True]
    })
    edited_df = st.data_editor(
        data=df,
        num_rows="dynamic",
    )

    all_times = df["time"].values
    training_times = df[df["used_for_fitting"]]["time"].values
    validation_times = df[~df["used_for_fitting"]]["time"].values
    max_time = all_times.max() + 10

    all_concentrations = df["concentration"].values
    training_concentrations = df[df["used_for_fitting"]]["concentration"].values
    validation_concentrations = df[~df["used_for_fitting"]]["concentration"].values


    # Optimization
    # -----------------------------------------------

    st.text("In the following we use the above data to fit a model of the form:")
    st.latex(r"C = A \cdot \exp(-\alpha \cdot t) + B \cdot \exp(-\beta \cdot t)")
    st.latex(r"t = \text{``Time''}")
    st.latex(r"C = \text{``Concentration''}")

    # Parameters
    a_init = st.number_input("A (initial guess)", min_value=0., max_value=None, value=30000.)
    alpha_init = st.number_input("alpha (initial guess)", min_value=0., max_value=None, value=0.1)
    B_init = st.number_input("B (initial guess)", min_value=0., max_value=None, value=25000.)
    beta_init = st.number_input("beta (initial guess)", min_value=0., max_value=None, value=0.009)
    x0 = np.asarray([a_init, alpha_init, B_init, beta_init])

    # Loss function
    def double_exp_fun(params, times):
        return params[0] * np.exp(- params[1] * times) + params[2] * np.exp(- params[3] * times)

    def loss(params, times, concentrations):
        return double_exp_fun(params, times) - concentrations

    # Solver
    result = least_squares(
        lambda x: loss(x, training_times, training_concentrations),
        x0 = x0
    )

    # Results
    # -----------------------------------------------

    # Charts
    fig = px.scatter(df, x="time", y="concentration", color="used_for_fitting")
    fig.add_scatter(
        x=np.arange(0, max_time, 0.1), 
        y=double_exp_fun(params=result.x, times=np.arange(0, 160, 0.1)),
        mode='lines',
        name = "Optimized solution"
    )
    fig.add_scatter(
        x=np.arange(0, max_time, 0.1), 
        y=double_exp_fun(params=x0, times=np.arange(0, 160, 0.1)),
        mode='lines',
        name = "Initial guess"
    )
    st.plotly_chart(fig)

    # Values
    st.metric("A (optimal)", value=round(result.x[0], 1), delta=round(result.x[0] - x0[0], 1))
    st.metric("alpha (optimal)", value=round(result.x[1], 5), delta=round(result.x[1] - x0[1], 5))
    st.metric("B (optimal)", value=round(result.x[2], 1), delta=round(result.x[2] - x0[2], 1))
    st.metric("beta (optimal)", value=round(result.x[3], 7), delta=round(result.x[3] - x0[3], 7))
    st.metric(
        "A/alpha + B/beta (optimal)", 
        value=round(result.x[0] / result.x[1] + result.x[2] / result.x[3], 1), 
        delta=round(result.x[0] / result.x[1] + result.x[2] / result.x[3] - x0[0] / x0[1] + x0[2] / x0[3], 1)
    )
    st.metric(
        "Average gap with data points not used for fitting", 
        value=np.round(np.abs([np.abs(double_exp_fun(params=result.x, times=np.array(t)) - c) for t, c in zip(validation_times, validation_concentrations)]), 3)
    )
