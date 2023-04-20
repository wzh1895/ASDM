# Firstly import the packages

import pysd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
model = pysd.read_xmile('Revised_Goal_Gap.stmx')

# Name the app

st.title('Over 78 Week Wait Goal Gap Model')

# Add Variables to the Smoking Cessation SD Class object

st.subheader('Enter the Size of the Backlog')
start_point = st.number_input(label='Backlog Size', min_value=None, max_value=None, value=3253)
st.subheader('Slide the Slider to Vary The Adjustment Time')
adjustment_time = st.slider("Adjustment Time in Months", 1, 12, 1)

# Run the Model

values = model.run(initial_condition=(0,{'> 78 weeks': start_point}), params={'Adjustment': adjustment_time})

# Export the Simulation Results

df_waiters = values[['> 78 weeks','Closed Long- Wait Pathways']]
st.subheader('Effects of Increasing Effort to Close Long Waiter Pathways - Predictions over 1 Year')
st.line_chart(df_waiters)