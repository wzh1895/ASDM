"""
Python model 'Revised_Goal_Gap.py'
Translated using PySD
"""

from pathlib import Path

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.9.1"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 1,
    "final_time": lambda: 12,
    "time_step": lambda: 1 / 4,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="INITIAL TIME", units="Months", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="FINAL TIME", units="Months", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="TIME STEP", units="Months", comp_type="Constant", comp_subtype="Normal"
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


@component.add(
    name="SAVEPER",
    units="Months",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The save time step for the simulation.
    """
    return __data["time"].saveper()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(name="Goal 1", comp_type="Constant", comp_subtype="Normal")
def goal_1():
    return 0


@component.add(
    name="Gap 1",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"nvs_78_weeks": 1, "goal_1": 1},
)
def gap_1():
    return nvs_78_weeks() - goal_1()


@component.add(
    name="Intervention Outflow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"gap_1": 1, "adjustment": 1},
)
def intervention_outflow():
    return gap_1() / adjustment()


@component.add(
    name="Adjustment", units="Months", comp_type="Constant", comp_subtype="Normal"
)
def adjustment():
    return 3


@component.add(
    name="> 78 weeks",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_nvs_78_weeks": 1},
    other_deps={
        "_integ_nvs_78_weeks": {"initial": {}, "step": {"intervention_outflow": 1}}
    },
)
def nvs_78_weeks():
    return _integ_nvs_78_weeks()


_integ_nvs_78_weeks = Integ(
    lambda: -intervention_outflow(), lambda: 3253, "_integ_nvs_78_weeks"
)


@component.add(
    name="Closed Long- Wait Pathways",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_closed_long_wait_pathways": 1},
    other_deps={
        "_integ_closed_long_wait_pathways": {
            "initial": {},
            "step": {"intervention_outflow": 1},
        }
    },
)
def closed_long_wait_pathways():
    return _integ_closed_long_wait_pathways()


_integ_closed_long_wait_pathways = Integ(
    lambda: intervention_outflow(), lambda: 10, "_integ_closed_long_wait_pathways"
)
