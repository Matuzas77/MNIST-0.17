#!/usr/bin/env python3
"""BESSEL5 R249S sparse restricted-Hecke exact newspace census.

Construct the cached all-prime newspace exactly over GF(q), then apply the raw
ambient Heilbronn operator separately to each embedded newspace basis vector.
This avoids constructing the full ambient T7 matrix.  At the smoke level the
restricted matrix is required to equal Sage's standard Hecke matrix exactly.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

from sage.all import GF, Integer, Matrix, ModularSymbols, gcd, kronecker_character, kronecker_symbol, set_verbose  # type: ignore


def log(message: str) -> None:
    print(message, flush=True)


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def checkpoint(outdir: Path, started: float, stage: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": rss_kib(),
    }
    payload.update(extra)
    (outdir / "checkpoint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        f"CHECKPOINT stage={stage} elapsed={payload['elapsed_seconds']:.6f} "
        f"max_rss_kib={payload['max_rss_kib']}"
    )


def fixed_character(level: int, field):
    chi = kronecker_character(Integer(-3)).extend(Integer(level)).change_ring(field)
    assert int(chi.modulus()) == level
    assert int(chi.conductor()) == 3
    assert int(chi.order()) == 2
    assert chi(Integer(-1)) == field(-1)
    for n in [1, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 53, 61, 71, 73]:
        if gcd(Integer(n), Integer(level)) == 1:
            assert chi(Integer(n)) == field(kronecker_symbol(Integer(-3), Integer(n)))
    return chi


def sparse_restricted_hecke(space, prime: int, outdir: Path, started: float):
    ambient = space.ambient_hecke_module()
    operator = ambient.hecke_operator(Integer(prime))
    dimension = int(space.dimension())
    field = space.base_ring()
    restricted = Matrix(field, dimension, dimension, sparse=False)
    basis = space.basis()

    log(f"SPARSE_T{prime}_BEGIN dimension={dimension}")
    for i, basis_vector in enumerate(basis):
        embedded = ambient(basis_vector.element())
        image = operator.apply_sparse(embedded)
        coordinates = space.coordinate_vector(image)
        restricted[i] = coordinates
        if i == 0 or (i + 1) % 32 == 0 or i + 1 == dimension:
            log(
                f"SPARSE_T{prime}_ROW rows_done={i + 1} total_rows={dimension} "
                f"max_rss_kib={rss_kib()}"
            )
            checkpoint(
                outdir,
                started,
                f"T{prime}_ROWS",
                rows_done=i + 1,
                total_rows=dimension,
            )
    log(f"SPARSE_T{prime}_END rows={restricted.nrows()} cols={restricted.ncols()}")
    return restricted


def run(level: int, q: int, expected_new_dimension: int, outdir: Path, verify_standard: bool) -> dict[str, Any]:
    if not Integer(q).is_prime():
        raise ValueError(f"q={q} is not prime")
    if gcd(Integer(level), Integer(q)) != 1 or q in (2, 3, 5, 7):
        raise ValueError("auxiliary prime must be away from 2,3,5,7 and the level")

    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    field = GF(Integer(q))
    set_verbose(1)

    log(f"BESSEL5_R249S_SPARSE_NEWSPACE_START level={level} q={q}")
    log(f"PYTHON_VERSION={platform.python_version()}")
    log(f"SAGE_VERSION={os.environ.get('SAGE_VERSION', 'runtime')}")
    checkpoint(outdir, started, "START", level=level, auxiliary_prime=q)

    chi = fixed_character(level, field)
    ambient = ModularSymbols(
        chi,
        Integer(3),
        sign=Integer(1),
        base_ring=field,
        use_cache=True,
    )
    log("AMBIENT_BEGIN")
    ambient_dimension = int(ambient.dimension())
    presentation_before = ambient.manin_gens_to_basis()
    log(f"AMBIENT_END dimension={ambient_dimension}")

    log("CUSPIDAL_BEGIN")
    cuspidal = ambient.cuspidal_submodule()
    cuspidal_dimension = int(cuspidal.dimension())
    log(f"CUSPIDAL_END dimension={cuspidal_dimension}")

    log("NEWSPACE_BEGIN")
    newspace = cuspidal.new_submodule()
    newspace_dimension = int(newspace.dimension())
    log(f"NEWSPACE_END dimension={newspace_dimension}")
    if newspace_dimension != expected_new_dimension:
        raise AssertionError(
            f"newspace dimension mismatch: got {newspace_dimension}, expected {expected_new_dimension}"
        )
    cache_reused = presentation_before is ambient.manin_gens_to_basis()
    if not cache_reused:
        raise AssertionError("target ambient Manin presentation was not reused")
    log("CACHE_REUSE=true")
    checkpoint(
        outdir,
        started,
        "NEWSPACE",
        ambient_dimension=ambient_dimension,
        cuspidal_dimension=cuspidal_dimension,
        newspace_dimension=newspace_dimension,
        cache_reused=cache_reused,
    )

    gc.collect()
    restricted = sparse_restricted_hecke(newspace, 7, outdir, started)
    sparse_equals_standard = None
    if verify_standard:
        log("STANDARD_T7_BEGIN")
        standard = newspace.hecke_matrix(Integer(7))
        sparse_equals_standard = restricted == standard
        if not sparse_equals_standard:
            raise AssertionError("sparse restricted T7 differs from Sage standard T7")
        log("STANDARD_T7_END equal=true")

    identity = restricted.parent().one()
    plus_rank = int((restricted - field(8) * identity).rank())
    plus_nullity = newspace_dimension - plus_rank
    minus_rank = int((restricted + field(8) * identity).rank())
    minus_nullity = newspace_dimension - minus_rank
    verdict = (
        "NO_GO_AT_P7_ON_EXACT_NEWSPACE"
        if plus_nullity == 0 and minus_nullity == 0
        else "EXACT_NEWSPACE_RANK_DROP"
    )

    result: dict[str, Any] = {
        "programme": "BESSEL5",
        "round": "R249S",
        "algorithm": "sage_cached_all_prime_newspace_sparse_restricted_hecke",
        "level": level,
        "weight": 3,
        "character": "chi_-3",
        "character_conductor": 3,
        "sign": 1,
        "auxiliary_prime": q,
        "ambient_dimension": ambient_dimension,
        "cuspidal_dimension": cuspidal_dimension,
        "newspace_dimension": newspace_dimension,
        "expected_characteristic_zero_newspace_dimension": expected_new_dimension,
        "cache_reused": cache_reused,
        "sparse_equals_standard": sparse_equals_standard,
        "t7_plus8_rank": plus_rank,
        "t7_plus8_nullity": plus_nullity,
        "t7_minus8_rank": minus_rank,
        "t7_minus8_nullity": minus_nullity,
        "t7_square64_nullity": plus_nullity + minus_nullity,
        "verdict": verdict,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": rss_kib(),
    }
    (outdir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outdir / "summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    (outdir / "SUCCESS").write_text("PASS\n", encoding="utf-8")
    log(
        "RESULT "
        f"level={level} q={q} newspace_dimension={newspace_dimension} "
        f"plus_rank={plus_rank} plus_nullity={plus_nullity} "
        f"minus_rank={minus_rank} minus_nullity={minus_nullity} "
        f"verdict={verdict}"
    )
    log(f"SPARSE_EQUALS_STANDARD={sparse_equals_standard}")
    log(f"MAX_RSS_KIB={result['max_rss_kib']}")
    log(f"ELAPSED_SECONDS={result['elapsed_seconds']:.6f}")
    log("SELF_CHECK=PASS")
    log(f"BESSEL5_R249S_SPARSE_NEWSPACE_END level={level} q={q}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--expected-new-dimension", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--verify-standard", action="store_true")
    args = parser.parse_args()
    run(
        args.level,
        args.prime,
        args.expected_new_dimension,
        args.outdir,
        args.verify_standard,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
