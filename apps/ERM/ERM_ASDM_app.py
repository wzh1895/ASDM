from ASDM.Engine import Structure
from ASDM.Parser import Parser
from ASDM.Solver import Solver
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import time

model = Structure(from_xmile='Elective Recovery Model.stmx')

# Name the app

st.title('Elective Recovery Model')

# Variables
st.subheader('Slide the Sliders to the left to Vary The starting proportion of patients waiting over 24mths for treatment and the time to simulate over in weeks')

# Add a slider to the sidebar:
sim_time = st.sidebar.slider(
    'Number of Weeks to Run the model for',
    1, 500, 250
)

# Add a slider to the sidebar:
long_wait = st.sidebar.slider(
    'Waiting over 24mths for treatment at model start',
    1, 1000, 150
)

# Add a slider to the sidebar:
shorter_wait = st.sidebar.slider(
    'Waiting 6 to 12mths for treatment at model start',
    1, 1000, 400
)

# Edit Stock
model.add_stock(name='Waiting over 24mths for treatment', equation={'nosubscript': str(long_wait)}, x=0, y=0, non_negative=True)
model.add_stock(name='Waiting 6 to 12mths for treatment', equation={'nosubscript': str(shorter_wait)}, x=0, y=0, non_negative=True)


# Run Model
model.clear_last_run()
model.simulate(simulation_time=sim_time, dt=1)

# Create the Model

df_outcome = model.export_simulation_result()

df_demand = df_outcome[['Routine treatment', 'Urgent treatment']]
df_backlog = df_outcome[['Total waiting for diagnostics or treatment','Waiting 6 to 12mths for treatment','Waiting 12 to 24mths for treatment','Waiting over 24mths for treatment']]

# Then plot

st.subheader('Estimated Backlog in Weeks')
st.line_chart(df_backlog)

if st.checkbox('Show Backlog Dataframe'):
    chart_data = df_backlog
    chart_data

'Calculating Demand...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Re-Running the Engine {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...done!'

st.subheader('Demand for Urgent and Routine Treatment')
st.line_chart(df_demand)

if st.checkbox('Show Treatment Demand DataFrame'):
    chart_data2 = df_demand
    chart_data2