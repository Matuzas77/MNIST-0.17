#!/usr/bin/env python3
"""Exact BESSEL5 R248S level-38400 p=2-new Hecke census.

The exact all-prime newspace is contained in the cuspidal subspace new at 2.
This program constructs that larger p=2-new space over GF(q), applies T_7
sparsely to its basis, and tests T_7=+8 and T_7=-8 exactly.
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

from sage.all import GF, Integer, Matrix, ModularSymbols, gcd, kronecker_character, kronecker_symbol, set_verbose


def log(message: str) -> None:
    print(message, flush=True)


def rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def checkpoint(outdir: Path, started: float, stage: str, **extra) -> None:
    data = {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_kib": rss_kib(),
        **extra,
    }
    (outdir / "checkpoint.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log(f"CHECKPOINT stage={stage} elapsed={data['elapsed_seconds']:.6f} max_rss_kib={data['max_rss_kib']}")


def fixed_character(level: int, field):
    chi = kronecker_character(Integer(-3)).extend(Integer(level)).change_ring(field)
    assert int(chi.modulus()) == level
    assert int(chi.conductor()) == 3
    assert int(chi.order()) == 2
    assert chi(Integer(-1)) == field(-1)
    for n in [1, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 53, 61, 71, 73]:
        if gcd(n, level) == 1:
            assert chi(Integer(n)) == field(kronecker_symbol(Integer(-3), Integer(n)))
    return chi


def p2new_subspace(cusp, outdir: Path, started: float):
    lower = Integer(cusp.level() // 2)
    log(f"LOWERING_MAP_1_BEGIN lower_level={lower}")
    d1 = cusp.degeneracy_map(lower, Integer(1)).matrix()
    log(f"LOWERING_MAP_1_END rows={d1.nrows()} cols={d1.ncols()}")
    checkpoint(outdir, started, "LOWERING_1", rows=int(d1.nrows()), cols=int(d1.ncols()))

    log(f"LOWERING_MAP_2_BEGIN lower_level={lower}")
    d2 = cusp.degeneracy_map(lower, Integer(2)).matrix()
    log(f"LOWERING_MAP_2_END rows={d2.nrows()} cols={d2.ncols()}")
    assert d1.nrows() == d2.nrows()

    log("LOWERING_AUGMENT_BEGIN")
    lowering = d1.augment(d2)
    log(f"LOWERING_AUGMENT_END rows={lowering.nrows()} cols={lowering.ncols()}")
    del d1, d2
    gc.collect()

    log("P2NEW_KERNEL_BEGIN")
    kernel = lowering.kernel()
    log(f"P2NEW_KERNEL_END dimension={kernel.dimension()} degree={kernel.degree()}")
    del lowering
    gc.collect()

    # The kernel is in coordinates relative to the cuspidal basis.  This is the
    # exact embedding operation used for a nonembedded coordinate submodule.
    p2new = cusp.submodule_from_nonembedded_module(kernel, check=False)
    assert int(p2new.dimension()) == int(kernel.dimension())
    checkpoint(outdir, started, "P2NEW", p2new_dimension=int(p2new.dimension()), cusp_dimension=int(cusp.dimension()))
    return p2new


def sparse_restricted_t7(space, outdir: Path, started: float):
    ambient = space.ambient_hecke_module()
    operator = ambient.hecke_operator(Integer(7))
    rank = int(space.dimension())
    field = space.base_ring()
    restricted = Matrix(field, rank, rank, sparse=False)
    coordinate_module = space.free_module()

    log(f"SPARSE_T7_BEGIN dimension={rank}")
    for i, x in enumerate(space.basis()):
        image = operator.apply_sparse(ambient(x))
        coordinates = coordinate_module.coordinate_vector(image.element())
        restricted[i] = coordinates
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == rank:
            log(f"SPARSE_T7_ROW rows_done={i + 1} total_rows={rank} max_rss_kib={rss_kib()}")
            checkpoint(outdir, started, "T7_ROWS", rows_done=i + 1, total_rows=rank, p2new_dimension=rank)
    log(f"SPARSE_T7_END rows={restricted.nrows()} cols={restricted.ncols()}")
    return restricted


def run(level: int, q: int, outdir: Path, verify_standard: bool) -> dict:
    if not Integer(q).is_prime() or level % q == 0 or q in (2, 3, 5, 7):
        raise ValueError("auxiliary prime must be prime and away from 2,3,5,7 and the level")

    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    field = GF(Integer(q))
    set_verbose(1)

    log(f"BESSEL5_R248S_LEVEL38400_SPARSE_P2NEW_START level={level} q={q}")
    log(f"PYTHON_VERSION={platform.python_version()}")
    log(f"SAGE_VERSION={os.environ.get('SAGE_VERSION', 'runtime')}")
    checkpoint(outdir, started, "START")

    chi = fixed_character(level, field)
    log(f"CHARACTER_OK modulus={chi.modulus()} conductor={chi.conductor()} order={chi.order()} value_minus_one={chi(-1)}")

    log("AMBIENT_BEGIN")
    ambient = ModularSymbols(chi, Integer(3), sign=Integer(1), base_ring=field, use_cache=False)
    ambient_dimension = int(ambient.dimension())
    log(f"AMBIENT_END dimension={ambient_dimension}")
    checkpoint(outdir, started, "AMBIENT", ambient_dimension=ambient_dimension)

    log("CUSPIDAL_BEGIN")
    cusp = ambient.cuspidal_subspace()
    cuspidal_dimension = int(cusp.dimension())
    log(f"CUSPIDAL_END dimension={cuspidal_dimension}")
    checkpoint(outdir, started, "CUSPIDAL", ambient_dimension=ambient_dimension, cuspidal_dimension=cuspidal_dimension)

    p2new = p2new_subspace(cusp, outdir, started)
    p2new_dimension = int(p2new.dimension())
    if level == 38400 and p2new_dimension < 2432:
        raise AssertionError(f"p=2-new dimension {p2new_dimension} is below exact newspace dimension 2432")
    log(f"P2NEW_DIMENSION={p2new_dimension}")

    t7 = sparse_restricted_t7(p2new, outdir, started)
    sparse_equals_standard = None
    if verify_standard:
        log("STANDARD_T7_VERIFICATION_BEGIN")
        standard = p2new.hecke_matrix(Integer(7))
        sparse_equals_standard = bool(t7 == standard)
        if not sparse_equals_standard:
            raise AssertionError("sparse restricted T7 differs from Sage standard T7")
        log("STANDARD_T7_VERIFICATION_END equal=true")

    identity = t7.parent().one()
    log("PLUS8_RANK_BEGIN")
    plus_rank = int((t7 - field(8) * identity).rank())
    plus_nullity = p2new_dimension - plus_rank
    log(f"PLUS8_RANK_END rank={plus_rank} nullity={plus_nullity}")

    log("MINUS8_RANK_BEGIN")
    minus_rank = int((t7 + field(8) * identity).rank())
    minus_nullity = p2new_dimension - minus_rank
    log(f"MINUS8_RANK_END rank={minus_rank} nullity={minus_nullity}")

    verdict = "NO_GO_AT_P7_ON_P2NEW_SUPERSET" if plus_nullity == 0 and minus_nullity == 0 else "P2NEW_RANK_DROP"
    result = {
        "programme": "BESSEL5",
        "round": "R248S",
        "algorithm": "sage_gf_sparse_restricted_hecke_on_cuspidal_p2new",
        "theorem_logic": "exact_newspace_subset_of_kernel_of_two_p2_lowering_maps",
        "level": level,
        "weight": 3,
        "character": "chi_-3",
        "character_conductor": 3,
        "sign": 1,
        "auxiliary_prime": q,
        "ambient_dimension": ambient_dimension,
        "cuspidal_dimension": cuspidal_dimension,
        "p2new_dimension": p2new_dimension,
        "exact_newspace_dimension_reference": 2432 if level == 38400 else None,
        "t7_plus8_rank": plus_rank,
        "t7_plus8_nullity": plus_nullity,
        "t7_minus8_rank": minus_rank,
        "t7_minus8_nullity": minus_nullity,
        "sparse_equals_standard": sparse_equals_standard,
        "verdict": verdict,
        "max_rss_kib": rss_kib(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (outdir / f"level{level}_q{q}_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (outdir / "summary.txt").write_text("\n".join(f"{k}={v}" for k, v in result.items()) + "\n")
    (outdir / "SUCCESS").write_text("PASS\n")

    log(f"RESULT level={level} q={q} p2new_dimension={p2new_dimension} plus_rank={plus_rank} plus_nullity={plus_nullity} minus_rank={minus_rank} minus_nullity={minus_nullity} verdict={verdict}")
    log(f"MAX_RSS_KIB={result['max_rss_kib']}")
    log(f"ELAPSED_SECONDS={result['elapsed_seconds']:.6f}")
    log("SELF_CHECK=PASS")
    log(f"BESSEL5_R248S_LEVEL38400_SPARSE_P2NEW_END level={level} q={q}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=38400)
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--verify-standard", action="store_true")
    args = parser.parse_args()
    result = run(args.level, args.prime, args.outdir, args.verify_standard)
    return 0 if result["verdict"] == "NO_GO_AT_P7_ON_P2NEW_SUPERSET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
