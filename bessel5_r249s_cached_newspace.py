#!/usr/bin/env python3
"""Exact cached modular-symbol T7 census for BESSEL5 R249S."""

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

from sage.all import (  # type: ignore
    GF,
    Integer,
    ModularSymbols,
    gcd,
    kronecker_character,
    kronecker_symbol,
    set_verbose,
)
from sage.modular.modsym import relation_matrix as relation_matrix_module  # type: ignore


def log(message: str) -> None:
    print(message, flush=True)


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def checkpoint(outdir: Path, started: float, stage: str, **extra: Any) -> None:
    record: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": rss_kib(),
    }
    record.update(extra)
    (outdir / "checkpoint.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        f"CHECKPOINT stage={stage} elapsed={record['elapsed_seconds']:.6f} "
        f"max_rss_kib={record['max_rss_kib']}"
    )


def fixed_character(level: int, field):
    chi = kronecker_character(Integer(-3)).extend(Integer(level)).change_ring(field)
    assert int(chi.modulus()) == level
    assert int(chi.conductor()) == 3
    assert int(chi.order()) == 2
    assert chi(Integer(-1)) == field(-1)
    for n in [1, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 53, 61, 71, 73]:
        if gcd(Integer(n), Integer(level)) == 1:
            expected = field(kronecker_symbol(Integer(-3), Integer(n)))
            assert chi(Integer(n)) == expected
    return chi


def install_presentation_counter() -> tuple[dict[int, int], Any]:
    calls: dict[int, int] = {}
    original = relation_matrix_module.compute_presentation

    def counted(syms, sign, field, sparse=None):
        level = int(syms.level())
        calls[level] = calls.get(level, 0) + 1
        log(
            f"PRESENTATION_CALL_BEGIN level={level} call={calls[level]} "
            f"sign={sign} field={field}"
        )
        started = time.monotonic()
        result = original(syms, sign, field, sparse=sparse)
        log(
            f"PRESENTATION_CALL_END level={level} call={calls[level]} "
            f"elapsed={time.monotonic()-started:.6f} rss_kib={rss_kib()}"
        )
        return result

    relation_matrix_module.compute_presentation = counted
    return calls, original


def run(level: int, q: int, expected_new_dimension: int, outdir: Path) -> dict[str, Any]:
    if not Integer(q).is_prime():
        raise ValueError(f"q={q} is not prime")
    if q in (2, 3, 5, 7) or gcd(Integer(level), Integer(q)) != 1:
        raise ValueError("auxiliary prime must be away from 2,3,5,7 and the level")

    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    field = GF(Integer(q))
    set_verbose(1)
    presentation_calls, original_presentation = install_presentation_counter()

    log(f"BESSEL5_R249S_CACHED_NEWSPACE_START level={level} q={q}")
    log(f"PYTHON_VERSION={platform.python_version()}")
    log(f"SAGE_VERSION={os.environ.get('SAGE_VERSION', 'runtime')}")
    checkpoint(outdir, started, "START", level=level, auxiliary_prime=q)

    try:
        chi = fixed_character(level, field)
        log(
            "CHARACTER_OK "
            f"modulus={chi.modulus()} conductor={chi.conductor()} "
            f"order={chi.order()} value_minus_one={chi(-1)}"
        )

        log("AMBIENT_BEGIN")
        ambient = ModularSymbols(
            chi,
            Integer(3),
            sign=Integer(1),
            base_ring=field,
            use_cache=True,
        )
        ambient_dimension = int(ambient.dimension())
        log(f"AMBIENT_END dimension={ambient_dimension}")
        checkpoint(
            outdir,
            started,
            "AMBIENT",
            ambient_dimension=ambient_dimension,
            target_presentation_calls=presentation_calls.get(level, 0),
        )

        log("CUSPIDAL_BEGIN")
        cuspidal = ambient.cuspidal_submodule()
        cuspidal_dimension = int(cuspidal.dimension())
        log(f"CUSPIDAL_END dimension={cuspidal_dimension}")
        checkpoint(
            outdir,
            started,
            "CUSPIDAL",
            ambient_dimension=ambient_dimension,
            cuspidal_dimension=cuspidal_dimension,
            target_presentation_calls=presentation_calls.get(level, 0),
        )

        log("NEWSPACE_BEGIN")
        newspace = cuspidal.new_submodule()
        newspace_dimension = int(newspace.dimension())
        log(f"NEWSPACE_END dimension={newspace_dimension}")
        if newspace_dimension != expected_new_dimension:
            raise AssertionError(
                f"newspace dimension mismatch: {newspace_dimension} != {expected_new_dimension}"
            )
        target_calls = presentation_calls.get(level, 0)
        if target_calls != 1:
            raise AssertionError(
                f"target presentation count is {target_calls}, expected exactly 1"
            )
        checkpoint(
            outdir,
            started,
            "NEWSPACE",
            ambient_dimension=ambient_dimension,
            cuspidal_dimension=cuspidal_dimension,
            newspace_dimension=newspace_dimension,
            target_presentation_calls=target_calls,
            presentation_calls_by_level=presentation_calls,
        )

        gc.collect()
        log("T7_BEGIN")
        T7 = newspace.hecke_matrix(Integer(7))
        if int(T7.nrows()) != newspace_dimension or int(T7.ncols()) != newspace_dimension:
            raise AssertionError("T7 matrix has wrong dimensions")
        log(f"T7_END rows={T7.nrows()} cols={T7.ncols()}")
        checkpoint(outdir, started, "T7", newspace_dimension=newspace_dimension)

        identity = T7.parent().one()
        log("PLUS8_RANK_BEGIN")
        plus_rank = int((T7 - field(8) * identity).rank())
        plus_nullity = newspace_dimension - plus_rank
        log(f"PLUS8_RANK_END rank={plus_rank} nullity={plus_nullity}")
        checkpoint(
            outdir,
            started,
            "PLUS8",
            plus_rank=plus_rank,
            plus_nullity=plus_nullity,
        )

        log("MINUS8_RANK_BEGIN")
        minus_rank = int((T7 + field(8) * identity).rank())
        minus_nullity = newspace_dimension - minus_rank
        log(f"MINUS8_RANK_END rank={minus_rank} nullity={minus_nullity}")

        verdict = (
            "NO_GO_AT_P7_ON_EXACT_NEWSPACE"
            if plus_nullity == 0 and minus_nullity == 0
            else "EXACT_NEWSPACE_RANK_DROP"
        )
        result: dict[str, Any] = {
            "programme": "BESSEL5",
            "round": "R249S",
            "algorithm": "sage_cached_exact_all_prime_newspace",
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
            "target_presentation_calls": target_calls,
            "presentation_calls_by_level": presentation_calls,
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
        log(f"PRESENTATION_CALLS_BY_LEVEL={json.dumps(presentation_calls, sort_keys=True)}")
        log(f"MAX_RSS_KIB={result['max_rss_kib']}")
        log(f"ELAPSED_SECONDS={result['elapsed_seconds']:.6f}")
        log("SELF_CHECK=PASS")
        log(f"BESSEL5_R249S_CACHED_NEWSPACE_END level={level} q={q}")
        return result
    finally:
        relation_matrix_module.compute_presentation = original_presentation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--expected-new-dimension", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    run(args.level, args.prime, args.expected_new_dimension, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
