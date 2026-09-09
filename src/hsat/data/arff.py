"""Minimal ARFF reader for ASlib scenario files.

ASlib ships plain, unquoted-mostly ARFF. `scipy.io.arff` mangles STRING attributes and
chokes on the `?` missing-value convention used throughout ASlib, so this module reads
the subset of the format the scenarios actually use.

Supported: @relation, @attribute (NUMERIC / STRING / nominal {a,b,c}), @data with
comma-separated rows, `?` for missing, quoted fields containing commas, % comments.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ATTR_RE = re.compile(r"^@attribute\s+(?P<name>'[^']*'|\"[^\"]*\"|\S+)\s+(?P<type>.+?)\s*$", re.I)


@dataclass(frozen=True)
class Attribute:
    name: str
    kind: str  # "numeric" | "string" | "nominal"
    levels: tuple[str, ...] | None = None


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_arff(path: str | Path) -> pd.DataFrame:
    """Read an ARFF file into a DataFrame.

    Numeric attributes become float columns with NaN for `?`; string and nominal
    attributes become object columns with None for `?`.
    """
    path = Path(path)
    attributes: list[Attribute] = []
    rows: list[list[str]] = []
    in_data = False

    with _open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                lowered = line.lower()
                if lowered.startswith("@data"):
                    in_data = True
                    continue
                if lowered.startswith("@attribute"):
                    match = _ATTR_RE.match(line)
                    if match is None:
                        raise ValueError(f"{path}: unparseable attribute line: {line!r}")
                    name = _unquote(match.group("name"))
                    decl = match.group("type").strip()
                    if decl.startswith("{"):
                        levels = tuple(
                            _unquote(v) for v in decl.strip("{}").split(",") if v.strip()
                        )
                        attributes.append(Attribute(name, "nominal", levels))
                    elif decl.lower().startswith(("numeric", "real", "integer")):
                        attributes.append(Attribute(name, "numeric"))
                    else:
                        attributes.append(Attribute(name, "string"))
                continue
            # data section
            fields = next(csv.reader([line], skipinitialspace=True))
            if len(fields) != len(attributes):
                raise ValueError(
                    f"{path}: row has {len(fields)} fields, expected {len(attributes)}: {line!r}"
                )
            rows.append(fields)

    if not attributes:
        raise ValueError(f"{path}: no @attribute declarations found")

    names = [a.name for a in attributes]
    frame = pd.DataFrame(rows, columns=names, dtype=object)

    for attribute in attributes:
        column = frame[attribute.name]
        cleaned = column.map(lambda v: None if v is None or str(v).strip() == "?" else str(v).strip())
        if attribute.kind == "numeric":
            frame[attribute.name] = pd.to_numeric(cleaned, errors="coerce").astype(np.float64)
        else:
            frame[attribute.name] = cleaned

    return frame


def read_arff_attributes(path: str | Path) -> list[Attribute]:
    """Read only the header of an ARFF file."""
    path = Path(path)
    attributes: list[Attribute] = []
    with _open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@data"):
                break
            if line.lower().startswith("@attribute"):
                match = _ATTR_RE.match(line)
                if match is None:
                    continue
                name = _unquote(match.group("name"))
                decl = match.group("type").strip()
                if decl.startswith("{"):
                    levels = tuple(_unquote(v) for v in decl.strip("{}").split(",") if v.strip())
                    attributes.append(Attribute(name, "nominal", levels))
                elif decl.lower().startswith(("numeric", "real", "integer")):
                    attributes.append(Attribute(name, "numeric"))
                else:
                    attributes.append(Attribute(name, "string"))
    return attributes
