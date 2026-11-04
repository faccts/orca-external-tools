"""
Classes for setting up global definitions.
"""

from enum import StrEnum

# Length conversion factors (Bohr -> unit)
LENGTH_CONVERSION = {"Ang": 0.529177210903}
# Energy conversion factors (Hartree -> unit)
ENERGY_CONVERSION = {
    "eV": 27.21138625,
    "kcal/mol": 627.509,
    "J": 4.3597447222071e-18,
}


class Property(StrEnum):
    """
    Properties that are supported by ORCA via external tools.
    """

    ENERGY = "energy"
    GRADIENT = "gradient"
    HESSIAN = "hessian"
