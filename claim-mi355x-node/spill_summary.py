#!/usr/bin/env python3
"""Parse llvm-readelf --notes output dumped per tile-config and summarize
SGPR / VGPR / spill / scratch counts for the TensorQuant Persistent
grouped GEMM kernels of each tile config, across 4 a/b layouts.

Run inside kyle_rocm713 after /tmp/sweep_spill.sh produced /tmp/notes_<TAG>.txt.
"""
import re, sys

CONFIGS = ["128x128", "256x256", "256x128", "128x256"]

# Filtering rules for kernel blocks:
#  * Must be the QuantGroupedGemmKernel template
#  * Persistent = true in pipeline params: "...PipelineSchedulerE1ELb1ELNS_10TailNumberE10ELNS_10CastPolicyE1"
#    (E1=Intrawave scheduler, Lb1=Persistent=true, TailNumberE10=Full, CastPolicy=1)
#  * TensorQuant = 3 — appears as the LSx_3 enum near the end of GemmConfig
#  * We pick the const-pointer variant `JPU3AS4KviE` (the `KviE` mangled tail)
#    to avoid double-counting the non-const variant which has identical regs.
#
# Layout signatures (a_layout, b_layout) — derived from tensor_layout::gemm
# template fragment in the demangled name. ColumnMajor uses 11 chars, RowMajor
# uses 8 chars; the second occurrence may be a back-reference (SG_/SH_) so we
# match flexibly.
LAYOUT_PATTERNS = {
    # RC (NT): A=RowMajor, B=ColumnMajor
    "RC(NT)":  r"8RowMajorENSF_11ColumnMajor",
    # RR (NN): A=RowMajor, B=RowMajor (B as back-ref SG_)
    "RR(NN)":  r"8RowMajorESG_SG_LNS_9QuantType",
    # CR (TN): A=ColumnMajor, B=RowMajor
    "CR(TN)":  r"11ColumnMajorENSF_8RowMajor",
    # CC: A=ColumnMajor, B=ColumnMajor (B as back-ref SG_)
    "CC":      r"11ColumnMajorESG_NSF_8RowMajor",  # the 3rd layout (c) is RowMajor anyway
}
# CC actually shares prefix with CR for some manglings; we'll prefer the most
# specific match — narrow it:
LAYOUT_PATTERNS["CC"] = r"11ColumnMajorESG_NSF_8RowMajorELNS_9QuantType"

PERSISTENT_TAG = "LNS_21GemmPipelineSchedulerE1ELb1ELNS_10TailNumberE10ELNS_10CastPolicyE1"

def per_block(text):
    return re.split(r"(?=\s+\.name:\s+_Z)", text)

def find_kernel(blocks, layout_pat):
    for b in blocks:
        if "QuantGroupedGemmKernel" not in b: continue
        if PERSISTENT_TAG not in b: continue
        if not re.search(layout_pat, b): continue
        if "JPU3AS4KviE" not in b: continue  # const-ptr variant only
        return b
    return None

def extract(block):
    g = lambda key: re.search(rf"\.{key}:\s+(\d+)", block).group(1)
    return dict(
        sgpr=g("sgpr_count"),
        vgpr=g("vgpr_count"),
        sgpr_spill=g("sgpr_spill_count"),
        vgpr_spill=g("vgpr_spill_count"),
        scratch=g("private_segment_fixed_size"),
    )

def main():
    print(f"{'Config':<10}{'Layout':<10}{'sgpr':>6}{'vgpr':>6}{'sgpr_sp':>10}{'vgpr_sp':>10}{'scratch':>10}")
    print("-" * 62)
    for cfg in CONFIGS:
        path = f"/tmp/notes_{cfg}.txt"
        try:
            blocks = per_block(open(path).read())
        except FileNotFoundError:
            print(f"{cfg:<10}MISSING ({path})")
            continue
        for layout, pat in LAYOUT_PATTERNS.items():
            blk = find_kernel(blocks, pat)
            if blk is None:
                print(f"{cfg:<10}{layout:<10}  NOT FOUND")
                continue
            r = extract(blk)
            print(f"{cfg:<10}{layout:<10}{r['sgpr']:>6}{r['vgpr']:>6}"
                  f"{r['sgpr_spill']:>10}{r['vgpr_spill']:>10}{r['scratch']:>10}")
        print()

if __name__ == "__main__":
    main()
