"""Automatic reconstruction of vertically shredded document images."""

from .baseline import reconstruct_baseline
from .document import DOCUMENT_VARIANTS, generate_document
from .evaluation import order_metrics
from .learned import CompatibilityModel, train_compatibility_model, train_v03_compatibility_model
from .shred import ShreddedDocument, compose_strips, shred_document

__all__ = [
    "CompatibilityModel",
    "DOCUMENT_VARIANTS",
    "ShreddedDocument",
    "compose_strips",
    "generate_document",
    "order_metrics",
    "reconstruct_baseline",
    "shred_document",
    "train_compatibility_model",
    "train_v03_compatibility_model",
]

__version__ = "0.2.0"
