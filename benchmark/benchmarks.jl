using SparseColumnPivotedQR, BenchmarkTools
using StableRNGs, SparseArrays, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Overdetermined sparse least-squares problem
A = sprand(rng, 2000, 500, 0.02) + 2I * 0
A = sprand(rng, 2000, 500, 0.02)
b = rand(rng, 2000)

# =============================================================================
# QR factorization with column pivoting
# =============================================================================

SUITE["factorize"] = BenchmarkGroup()

SUITE["factorize"]["scpqr"] = @benchmarkable scpqr($A)
SUITE["factorize"]["scpqr_natural"] = @benchmarkable scpqr($A; ordering = :natural)
SUITE["factorize"]["scpqr_analyze"] = @benchmarkable scpqr_analyze($A)

F = scpqr(A)

# =============================================================================
# Solve least-squares
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["ldiv"] = @benchmarkable $F \ $b
SUITE["solve"]["refactor"] = @benchmarkable scpqr_refactor!($F, $A)
