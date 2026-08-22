#!/usr/bin/env python3
"""Independent standard-Sage R248S p=2-new census at level 38400."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

from sage.all import GF, Integer, ModularSymbols, gcd, kronecker_character, kronecker_symbol, set_verbose


def log(x: str) -> None:
    print(x, flush=True)


def rss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def checkpoint(outdir: Path, started: float, stage: str, **kw) -> None:
    d = {"stage": stage, "elapsed_seconds": time.monotonic() - started, "max_rss_kib": rss(), **kw}
    (outdir / "checkpoint.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    log(f"CHECKPOINT stage={stage} elapsed={d['elapsed_seconds']:.6f} max_rss_kib={d['max_rss_kib']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    level = 38400
    q = args.prime
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not Integer(q).is_prime() or level % q == 0 or q in (2, 3, 5, 7):
        raise ValueError("bad auxiliary prime")
    field = GF(Integer(q))
    set_verbose(1)

    log(f"BESSEL5_R248S_LEVEL38400_STANDARD_P2NEW_START q={q}")
    log(f"PYTHON_VERSION={platform.python_version()}")
    log(f"SAGE_VERSION={os.environ.get('SAGE_VERSION', 'runtime')}")
    checkpoint(outdir, started, "START")

    chi = kronecker_character(Integer(-3)).extend(Integer(level)).change_ring(field)
    assert int(chi.conductor()) == 3 and int(chi.order()) == 2 and chi(-1) == field(-1)
    for n in [7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 53, 61, 71, 73]:
        if gcd(n, level) == 1:
            assert chi(Integer(n)) == field(kronecker_symbol(Integer(-3), Integer(n)))
    log(f"CHARACTER_OK modulus={chi.modulus()} conductor={chi.conductor()} order={chi.order()}")

    log("AMBIENT_BEGIN")
    ambient = ModularSymbols(chi, Integer(3), sign=Integer(1), base_ring=field, use_cache=False)
    ambient_dim = int(ambient.dimension())
    log(f"AMBIENT_END dimension={ambient_dim}")
    checkpoint(outdir, started, "AMBIENT", ambient_dimension=ambient_dim)

    log("CUSPIDAL_BEGIN")
    cusp = ambient.cuspidal_subspace()
    cusp_dim = int(cusp.dimension())
    log(f"CUSPIDAL_END dimension={cusp_dim}")
    checkpoint(outdir, started, "CUSPIDAL", cuspidal_dimension=cusp_dim)

    log("P2NEW_STANDARD_BEGIN")
    p2new = cusp.new_subspace(Integer(2))
    p2new_dim = int(p2new.dimension())
    log(f"P2NEW_STANDARD_END dimension={p2new_dim}")
    if p2new_dim < 2432:
        raise AssertionError("p=2-new superset smaller than exact newspace")
    checkpoint(outdir, started, "P2NEW", p2new_dimension=p2new_dim)

    log("T7_STANDARD_BEGIN")
    t7 = p2new.hecke_matrix(Integer(7))
    log(f"T7_STANDARD_END rows={t7.nrows()} cols={t7.ncols()}")
    checkpoint(outdir, started, "T7", p2new_dimension=p2new_dim)

    identity = t7.parent().one()
    plus_rank = int((t7 - field(8) * identity).rank())
    minus_rank = int((t7 + field(8) * identity).rank())
    plus_nullity = p2new_dim - plus_rank
    minus_nullity = p2new_dim - minus_rank
    verdict = "NO_GO_AT_P7_ON_P2NEW_SUPERSET" if plus_nullity == 0 and minus_nullity == 0 else "P2NEW_RANK_DROP"

    result = {
        "programme": "BESSEL5",
        "round": "R248S",
        "algorithm": "sage_standard_p2new_standard_hecke_matrix",
        "level": level,
        "weight": 3,
        "character": "chi_-3",
        "auxiliary_prime": q,
        "ambient_dimension": ambient_dim,
        "cuspidal_dimension": cusp_dim,
        "p2new_dimension": p2new_dim,
        "exact_newspace_dimension_reference": 2432,
        "t7_plus8_rank": plus_rank,
        "t7_plus8_nullity": plus_nullity,
        "t7_minus8_rank": minus_rank,
        "t7_minus8_nullity": minus_nullity,
        "verdict": verdict,
        "max_rss_kib": rss(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (outdir / f"level38400_q{q}_standard_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (outdir / "summary.txt").write_text("\n".join(f"{k}={v}" for k, v in result.items()) + "\n")
    (outdir / "SUCCESS").write_text("PASS\n")
    log(f"RESULT q={q} p2new_dimension={p2new_dim} plus_rank={plus_rank} plus_nullity={plus_nullity} minus_rank={minus_rank} minus_nullity={minus_nullity} verdict={verdict}")
    log("SELF_CHECK=PASS")
    log(f"BESSEL5_R248S_LEVEL38400_STANDARD_P2NEW_END q={q}")
    return 0 if verdict == "NO_GO_AT_P7_ON_P2NEW_SUPERSET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
