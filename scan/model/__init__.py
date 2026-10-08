"""
Module containing regression models for modeling the relationship between
physio signals and functional MRI signals.
"""

from scan.model import glm
from scan.model.fc import FCMapModel
from scan.model.glm import (
    GLMSpline,
    GLMSplineResults,
    HRFBasis,
)

DistributedLagModel = GLMSpline

__all__ = [
    "DistributedLagModel",
    "FCMapModel",
    "GLMSpline",
    "GLMSplineResults",
    "HRFBasis",
    "glm",
]
