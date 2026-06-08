"""Backward-compatibility shim — import from ``inf_bp_texts.schemas`` instead.

This module re-exports all symbols from :mod:`inf_bp_texts.schemas` so that
any cached notebooks or scripts that still reference ``inf_bp_texts.types``
continue to work.  New code should import directly from ``.schemas``.
"""

from .schemas import BasicMetadata, ParsedContentOverview, ValidBookSet, ValidSectionSet

__all__ = [
    "BasicMetadata",
    "ParsedContentOverview",
    "ValidBookSet",
    "ValidSectionSet",
]
