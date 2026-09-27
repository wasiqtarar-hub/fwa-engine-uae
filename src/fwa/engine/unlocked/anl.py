"""Unlocked implementations for the ANL family.

Each function here runs only when a dataset unlock (`rules/unlocks/ANL.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.
"""

from __future__ import annotations

from typing import Callable

IMPLEMENTATIONS: dict[str, Callable] = {}
