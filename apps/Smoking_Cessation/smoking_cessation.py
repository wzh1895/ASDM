import math
import pandas as pd
import streamlit as st
from ASDM.Engine import Structure
from ASDM.Utilities import plot_time_series
from IPython.display import Image

# Name the apps

st.title('Smoking Cessation Machine')

# Add Variables to the Smoking Cessation SD Class object

st.subheader('Slide the Slider to Vary Re-Investment Levels')

re_investment = st.slider("Proportion of Savings Spent on Cessation", 1, 100, 1)

class SmokingCessation(Structure):
    def __init__(self):
        super(SmokingCessation, self).__init__()
        self.add_stock("currentSmokers", 900, non_negative=True)
        self.add_stock("exSmokers", 100, non_negative=True)
        self.add_stock("lapsedExSmokers", 0, non_negative=True)
        self.add_flow("smokersQuitting", "smokingCessationServiceFunding/spendPerQuitter", flow_from='currentSmokers', flow_to="exSmokers")        
        self.add_flow("exSmokersStartingAgain", "exSmokers*averageQuitterFailureRate", flow_from='exSmokers', flow_to="lapsedExSmokers")
        self.add_aux("effectOnSpendPerQuitter", "currentSmokers/init(currentSmokers)")
        self.add_aux("spendPerQuitter", "200/effectOnSpendPerQuitter")
        self.add_aux("percentageOfSavingsSpentOnCessation", re_investment)                
        self.add_aux("averageQuitterFailureRate", 0.05)
        self.add_aux("healthcareSavings", "exSmokers*monthlyCostSavingsPerExSmoker")
        self.add_aux("smokingCessationServiceFunding", "(healthcareSavings*percentageOfSavingsSpentOnCessation)/100")
        self.add_aux("monthlyCostSavingsPerExSmoker", 50)

# Create an instance of the model

smoking_model = SmokingCessation()
smoking_model.clear_last_run()
smoking_model.simulate(simulation_time=36, dt=0.25)

# Export the Simulation Results

df_smoking_outcome = smoking_model.export_simulation_result()
df_smoking = df_smoking_outcome[['currentSmokers', 'exSmokers', 'exSmokersStartingAgain']]

st.subheader('Effects of Re-Investment on Smoking Levels')
st.line_chart(df_smoking)