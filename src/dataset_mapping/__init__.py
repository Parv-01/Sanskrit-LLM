"""Dataset mapping and schema definitions."""

from .schema import DatasetSchema, SymptomSchema, TreatmentSchema
from .converter import JSONConverter

__all__ = ["DatasetSchema", "SymptomSchema", "TreatmentSchema", "JSONConverter"]
