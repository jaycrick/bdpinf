"""Shared type aliases for birth-death process epoch definitions."""

from __future__ import annotations

EpochDefinition = tuple[int, int, str]
"""An epoch as a ``(start_year, end_year, label)`` triple."""

__all__ = ["EpochDefinition"]
