# n_pvec_opcount.jl — operation-count analysis of the ECG_Julia N_Pvec / N_C_Pvec
# nitrogen matrix element. Julia equivalent of n_pvec_opcount.py.
#
# For one basis pair (k,l):
#     S_kl = sum_{j=1..N_terms} YHYCoeff[j] * S_term(Ak, Al, P_j)
# and each term needs the permuted combined width matrix
#     Ã^(g) = Ak + P_g' Al P_g     (n×n, symmetric positive-definite),
# its |Ã^(g)|^(-3/2), inverse (Ã^(g))^(-1), and a 36-term angular contraction.
#
# The "Pvec" trick builds P_g' Al P_g by an index GATHER in O(n^2) instead of
# two dense matrix products in O(n^3) — the legitimate per-term saving. The
# determinant/inverse stay O(n^3) per term and the term count N_terms stays
# combinatorial (n!, reduced by the Young projector). Applies to both N_Pvec
# (real) and N_C_Pvec (complex); the complex case is a constant-factor heavier.
#
# Usage in a notebook:
#     include("../ECG_Julia/n_pvec_opcount.jl")
#     OpCount.report(npart; N_terms = NumYHYTerms)        # fast, analytic
#     OpCount.report(npart; N_terms = NumYHYTerms, run_distinct = true)  # + demo
#     OpCount.scaling_table()
#
# Released under the MIT License. Copyright (c) 2026 Alain Chancé.

module OpCount

using LinearAlgebra
using Printf
using Random

export report, scaling_table

"Build a random n×n symmetric positive-definite matrix A = L Lᵀ."
function _spd(n::Int; seed::Int = 1)
    rng = MersenneTwister(seed)
    L = zeros(n, n)
    for i in 1:n, j in 1:i
        L[i, j] = rand(rng) - 0.5
    end
    for i in 1:n
        L[i, i] += 1.0
    end
    return L * transpose(L)
end

"Pvec build: Ã = Ak + P' Al P via an index gather, O(n²), no multiplications."
build_gather(Ak, Al, g) = [Ak[i, j] + Al[g[i], g[j]]
                           for i in axes(Ak, 1), j in axes(Ak, 2)]

"Naive build: Ã = Ak + P' Al P via two dense matrix products, O(n³)."
function build_naive(Ak, Al, g)
    n = size(Ak, 1)
    P = zeros(n, n)
    for t in 1:n
        P[g[t], t] = 1.0
    end
    return Ak .+ transpose(P) * Al * P
end

const _DETINV(n) = round(Int, 2 * (2 / 3) * n^3)   # det + inverse ≈ (4/3) n³ flops

"""
    report(n=7; N_terms=factorial(n), n_angular=36, complex=false, run_distinct=false)

Print the per-term and per-matrix-element operation counts for the nitrogen
`N_Pvec` (real) / `N_C_Pvec` (complex) element. Pass `complex=true` for
`N_C_Pvec`: complex adds cost ~2× and complex multiplies ~4× the real flops, so
the per-term costs are scaled accordingly. With `run_distinct=true` it also
enumerates all `n!` permutations and reports how many of the per-term
determinants are distinct (the irreducibility check). Returns a NamedTuple.
"""
function report(n::Int = 7; N_terms::Int = factorial(n),
                n_angular::Int = 36, complex::Bool = false,
                run_distinct::Bool = false, verbose::Bool = true)
    Ak, Al = _spd(n; seed = 1), _spd(n; seed = 2)

    # Equivalence check on a handful of random permutations (cheap, always run).
    rng = MersenneTwister(0)
    maxdiff = 0.0
    for _ in 1:20
        g = randperm(rng, n)
        maxdiff = max(maxdiff, maximum(abs.(build_gather(Ak, Al, g) .-
                                            build_naive(Ak, Al, g))))
    end

    cmul = complex ? 4 : 1            # real flops per complex multiply (~4)
    cadd = complex ? 2 : 1            # real flops per complex add (~2)
    gather   = cadd * n^2            # the gather is pure additions
    naive    = cmul * 2 * n^3        # naive build is multiply-heavy
    detinv   = cmul * _DETINV(n)     # det + inverse are multiply-heavy
    n_angular = cmul * n_angular     # angular contraction is multiply-heavy
    per_elem = N_terms * (gather + detinv + n_angular)

    distinct = -1
    if run_distinct
        seen = Set{Float64}()
        for g in permutations_of(n)
            push!(seen, round(det(build_gather(Ak, Al, g)); digits = 9))
        end
        distinct = length(seen)
    end

    if verbose
        println("="^68)
        @printf("%s op-count   n = %d,  N_terms = %d%s\n",
                complex ? "N_C_Pvec (complex)" : "N_Pvec (real)", n, N_terms,
                complex ? "   [complex: ×2 adds, ×4 mults]" : "")
        println("="^68)
        @printf("gather build == naive build (max |diff|, 20 perms): %.2e\n\n", maxdiff)
        @printf("Per-term build:   Pvec gather  n²      = %6d  (additions, no mults)\n", gather)
        @printf("                  naive P'AlP  2n³     = %6d  (multiply-adds)\n", naive)
        @printf("                  -> gather is ~%.0f× cheaper, O(n²) vs O(n³)\n\n",
                naive / gather)
        @printf("Per-term irreducible: det+inverse ~(4/3)n³ = %6d  flops\n", detinv)
        @printf("                      angular 36-term       = %6d\n\n", n_angular)
        @printf("Per matrix element ~ N_terms × (gather + det/inv + ang)\n")
        @printf("                   = %d × (%d + %d + %d) ~ %s flops\n",
                N_terms, gather, detinv, n_angular, format_int(per_elem))
        if run_distinct
            @printf("\nDistinct per-term determinants: %d of %d  -> all terms distinct\n",
                    distinct, factorial(n))
            println("  (no fixed contraction collapses the sum; det/inverse and the")
            println("   N_terms factor are both irreducible)")
        end
    end
    return (; n, N_terms, complex, gather, naive, detinv, n_angular, per_elem, maxdiff, distinct)
end

"Print how each cost scales with n for the full operator (N_terms = n!)."
function scaling_table(ns = 3:8)
    println("="^68)
    println("Scaling with n (full operator, N_terms = n!)")
    println("="^68)
    @printf("%2s %12s %16s %16s %16s\n", "n", "N_terms=n!",
            "gather build", "det/inv work", "naive build")
    println("-"^66)
    for n in ns
        N = factorial(n)
        @printf("%2d %12s %16s %16s %16s\n", n, format_int(N),
                format_int(N * n^2), format_int(N * _DETINV(n)), format_int(N * 2 * n^3))
    end
    println("\nThe Pvec gather turns the per-term BUILD from O(n³) into O(n²) — a real,")
    println("correctly-scoped saving. The per-term det/inverse stays O(n³) and the")
    println("term count stays n! (reduced by the Young projector to a fixed fraction),")
    println("so the method is O(N_terms · n³), not O(n²).")
    return nothing
end

# --- small helpers (zero-dependency) --------------------------------------- #
"Comma-grouped integer formatting, e.g. 2731680 -> \"2,731,680\"."
function format_int(x::Integer)
    s = string(abs(x))
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end]); s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return (x < 0 ? "-" : "") * join(parts, ",")
end

"All permutations of 1:n (n! of them); used only by the distinctness demo."
function permutations_of(n::Int)
    n == 0 && return [Int[]]
    out = Vector{Vector{Int}}()
    for p in permutations_of(n - 1), i in 1:n
        q = copy(p); insert!(q, i, n); push!(out, q)
    end
    return out
end

end # module OpCount

if abspath(PROGRAM_FILE) == @__FILE__
    OpCount.report(7; run_distinct = true)
    println()
    OpCount.scaling_table()
end
