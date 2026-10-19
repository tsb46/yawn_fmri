"""
Module containing regression models for modeling the relationship between
physio signals and functional MRI signals.
"""

from yawn.model import glm
from yawn.model.fc import FCMapModel
from yawn.model.glm import (
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
