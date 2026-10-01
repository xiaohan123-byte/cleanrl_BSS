"""Algebraic model fingerprints, independent of Python hash randomization."""
from __future__ import annotations

import hashlib
import json
import math


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _number(value):
    value = float(value)
    return (0. if value == 0 else value).hex()


def model_fingerprints(model):
    """Hash every bound/type/objective coefficient and every linear row.

    The semantic hash ignores row/column order and row sign, retaining duplicate
    rows. No coefficients are rounded. The ordered hash retains column/row order.
    """
    from coptpy import COPT

    model.update()
    variables = list(model.getVars())
    names = [v.name for v in variables]
    if len(set(names)) != len(names):
        raise ValueError('model fingerprint requires unique variable names')
    columns = [[v.name, v.vtype, _number(v.lb), _number(v.ub), _number(v.obj)] for v in variables]
    header = [int(model.getAttr(COPT.Attr.ObjSense)), _number(model.getAttr(COPT.Attr.ObjConst))]
    ordered = hashlib.sha256(_encoded([header, columns]))
    row_hashes = []
    for constraint in model.getConstrs():
        row = model.getRow(constraint)
        entries = [(row.getVar(k).index, row.getCoeff(k)) for k in range(row.getSize())]
        constant = row.getConstant()
        lo = constraint.getInfo(COPT.Info.LB) - constant
        hi = constraint.getInfo(COPT.Info.UB) - constant
        ordered.update(_encoded([_number(lo), _number(hi),
                                 [(index, _number(v)) for index, v in sorted(entries)]]))
        grouped = {}
        for index, value in entries:
            grouped.setdefault(names[index], []).append(value)
        terms = sorted((name, math.fsum(values)) for name, values in grouped.items())
        terms = [(name, value) for name, value in terms if value != 0]
        if terms and terms[0][1] < 0:
            terms = [(name, -value) for name, value in terms]
            lo, hi = -hi, -lo
        row_hashes.append(hashlib.sha256(_encoded(
            [_number(lo), _number(hi), [(name, _number(v)) for name, v in terms]])).digest())
    semantic = hashlib.sha256(_encoded([header, sorted(columns)]))
    for value in sorted(row_hashes):
        semantic.update(value)
    return dict(ordered_sha256=ordered.hexdigest(), semantic_sha256=semantic.hexdigest(),
                columns=len(columns), rows=len(row_hashes),
                binary_variables=int(model.getAttr(COPT.Attr.Bins)),
                coefficient_comparison='exact float64, no rounding')
