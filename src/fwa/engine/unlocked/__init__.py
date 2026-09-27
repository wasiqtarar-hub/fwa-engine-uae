"""Implementations that run only when a dataset unlock is satisfied.

The catalogue's own implementations live in :mod:`fwa.engine.controls` and
run on any dataset their static classification allows. The functions in this
package need tables a claim-header extract does not carry (claim lines, a
licence register, a bundling edit table …). They are reachable **only**
through a declaration in ``rules/unlocks/``, and only on a run whose dataset
populates every table that declaration names.

Each family module exports ``IMPLEMENTATIONS``; they are merged here. A name
declared twice is an error rather than a silent override.
"""

from __future__ import annotations

from typing import Callable

from . import anl, cln, doc, ent, net, pay_a, pay_b, phr, pol

__all__ = ["UNLOCKED_IMPLEMENTATIONS"]

UNLOCKED_IMPLEMENTATIONS: dict[str, Callable] = {}
for _module in (ent, pol, pay_a, pay_b, cln, phr, doc, net, anl):
    for _name, _fn in _module.IMPLEMENTATIONS.items():
        if _name in UNLOCKED_IMPLEMENTATIONS:
            raise RuntimeError(f"Unlocked implementation {_name!r} is declared twice.")
        UNLOCKED_IMPLEMENTATIONS[_name] = _fn
