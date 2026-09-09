"""Mapping the proposal's Table 4.1 portfolio onto ASlib scenario algorithm names.

The proposal fixes a five-solver portfolio: MiniSat, Glucose, CaDiCaL, CryptoMiniSat and
Kissat. ASlib scenarios name competition entries, not solver families, so the mapping is
by pattern with an explicit preference order — and it is deliberately conservative:
heavily modified derivatives (`Maple*`, `gluHack`, `expGlucose`, `upGlucose`) are *not*
treated as their ancestors, because a hacked Glucose is a different solver for the
purpose of studying complementarity.

Coverage of the two candidate scenarios (verified, see docs/PLAN.md section 10):

    SAT18-EXP   MiniSat, Glucose, CaDiCaL, CryptoMiniSat      (no Kissat: it debuted 2020)
    SAT20-MAIN  Glucose, CaDiCaL, CryptoMiniSat, Kissat       (no MiniSat)

Together they cover all five families, which is why the plan reports both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .scenario import Scenario

# family -> ordered regexes, most-preferred first. Every pattern is fully anchored: a
# family resolves only to a recognisably stock entry, never to a derivative. A loose
# `^glucose` fallback would happily claim `glucose_kiel` (a Glucose *hack* submitted to
# SAT 2016) as Glucose, and SAT03-16_INDU contains nothing else — reporting that family
# as absent is the honest answer.
PROPOSAL_PORTFOLIO: dict[str, tuple[str, ...]] = {
    "MiniSat": (r"^minisat[-_.]?v?2\.2[\w.-]*$", r"^minisat[\d._-]*$"),
    # Prefer a bare versioned name (`glucose4.2.1`, `glucose3.0+proofs`) over a variant
    # carrying extra words (`glucose-3.0-inprocess`, `glucose-3.0_PADC_10`).
    "Glucose": (
        r"^glucose[-_]?4[\d.]*(\+[\w.-]+)?$",
        r"^glucose[-_]?3[\d.]*(\+[\w.-]+)?$",
    ),
    "CaDiCaL": (r"^cadical([-_]sc\d+)?(\+default)?$",),
    "CryptoMiniSat": (
        r"^cryptominisat[-_]?ccnr\+default$",
        r"^cryptominisat[\d._-]*(\+[\w.-]+)?$",
        r"^cms\d+[-_]main[\w.-]*$",
    ),
    "Kissat": (
        r"^kissat[-_]sc\d+[-_]default\+default$",
        r"^kissat[\d._-]*(\+[\w.-]+)?$",
    ),
}


@dataclass(frozen=True)
class PortfolioMatch:
    wanted: str
    algorithm: str | None
    pattern: str | None


def match_portfolio(
    scenario: Scenario, portfolio: dict[str, tuple[str, ...]] | None = None
) -> list[PortfolioMatch]:
    """Resolve each proposal solver family to at most one algorithm in the scenario."""
    portfolio = portfolio or PROPOSAL_PORTFOLIO
    taken: set[str] = set()
    matches: list[PortfolioMatch] = []

    for family, patterns in portfolio.items():
        chosen: tuple[str, str] | None = None
        for pattern in patterns:
            candidates = sorted(
                a for a in scenario.algorithms
                if a not in taken and re.match(pattern, a, flags=re.IGNORECASE)
            )
            if candidates:
                chosen = (candidates[0], pattern)
                break
        if chosen:
            taken.add(chosen[0])
            matches.append(PortfolioMatch(family, chosen[0], chosen[1]))
        else:
            matches.append(PortfolioMatch(family, None, None))

    return matches
