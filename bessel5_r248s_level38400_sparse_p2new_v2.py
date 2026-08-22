#!/usr/bin/env python3
"""Corrected wrapper for the R248S sparse p=2-new computation.

The two lowering-map kernels are computed in coordinates relative to the
cuspidal basis.  They must therefore be embedded with
``submodule_from_nonembedded_module`` rather than ``submodule``.
"""

from __future__ import annotations

import gc

import bessel5_r248s_level38400_sparse_p2new as base


def corrected_p2new_subspace(cusp, outdir, started):
    lower = base.Integer(cusp.level() // 2)
    base.log(f"LOWERING_MAP_1_BEGIN lower_level={lower}")
    d1 = cusp.degeneracy_map(lower, base.Integer(1)).matrix()
    base.log(f"LOWERING_MAP_1_END rows={d1.nrows()} cols={d1.ncols()}")
    base.checkpoint(outdir, started, "LOWERING_1", rows=int(d1.nrows()), cols=int(d1.ncols()))

    base.log(f"LOWERING_MAP_2_BEGIN lower_level={lower}")
    d2 = cusp.degeneracy_map(lower, base.Integer(2)).matrix()
    base.log(f"LOWERING_MAP_2_END rows={d2.nrows()} cols={d2.ncols()}")
    if d1.nrows() != d2.nrows():
        raise AssertionError("lowering maps have incompatible domains")

    base.log("LOWERING_AUGMENT_BEGIN")
    d = d1.augment(d2)
    base.log(f"LOWERING_AUGMENT_END rows={d.nrows()} cols={d.ncols()}")
    del d1, d2
    gc.collect()

    base.log("P2NEW_KERNEL_BEGIN")
    kernel = d.kernel()
    base.log(f"P2NEW_KERNEL_END dimension={kernel.dimension()} degree={kernel.degree()}")
    del d
    gc.collect()

    p2new = cusp.submodule_from_nonembedded_module(kernel, check=False)
    if int(p2new.dimension()) != int(kernel.dimension()):
        raise AssertionError("embedded p=2-new dimension mismatch")
    base.checkpoint(
        outdir,
        started,
        "P2NEW",
        p2new_dimension=int(p2new.dimension()),
        cusp_dimension=int(cusp.dimension()),
    )
    return p2new


base.p2new_subspace = corrected_p2new_subspace

if __name__ == "__main__":
    raise SystemExit(base.main())
