"""
symmetry_operators.jl
=====================
Builds the 4D Transposit array (all pair transposition matrices P_{ij})
and computes the Y and Y†Y operator expansions from a product-of-factors
input string.

Verified against the Nitrogen atom reference output (960 Y terms, 5040 YHY
terms) and the HD+ case (1 Y term, 1 YHY term).

Particle / index conventions (matching book Appendix A):
  • Particles are 1-indexed. Particle 1 is the nucleus; particles 2..N+1
    are the N electrons.
  • After removing the centre-of-mass, N Jacobi vectors remain (1-indexed).
    Their matrix rows are 1-indexed in Julia (1..N).
  • Transposition P_{a,b} in Jacobi-vector space (heliocentric coordinates
    x_i = r_{i+1} - r_1):
      a == b              → identity
      a == 1, b ≥ 2       → pseudo-particle transposition with the reference
                            particle:  x_{b-1} → -x_{b-1},
                                       x_i → x_i - x_{b-1}  (i ≠ b-1)
                            i.e. identity with column (b-1) replaced by -1's
      else                → swap rows (a-1) and (b-1)  (1-indexed)

Input string format:
  A product of parenthesised factors, each being a signed linear combination
  of transpositions.  Examples:
      (P11)
      (1+P56)(1+P78)(1-P68)(1-P57)(1-P27-P25)(1-P23-P37-P35)(1-P34-P24-P47-P45)
      (1-P13P24)(1+P12)(1+P34)

  Supported term forms inside each factor:
      "1"  or "+1"        → identity, coefficient +1
      "-1"                → identity, coefficient -1
      "P<a><b>"           → P_{a,b}, coefficient +1  (a,b each a single digit)
      "+P<a><b>"          → P_{a,b}, coefficient +1
      "-P<a><b>"          → P_{a,b}, coefficient -1
      "P<a><b>P<c><d>…"   → product of transpositions (single term), with the
                            optional leading sign as its coefficient.  The
                            matrices are multiplied left-to-right; transposition
                            products appearing in practice (e.g. P13P24 in the
                            Ps2 operator) act on disjoint index pairs and
                            therefore commute.

Enumeration / product convention:
  • Factors F_1 F_2 … F_k labelled left-to-right as written.
  • The rightmost factor (F_k) varies *fastest* in the enumeration.
  • The combined matrix for a choice (t_1, …, t_k) is
        F_k[t_k] * F_{k-1}[t_{k-1}] * … * F_1[t_1]
    (rightmost matrix applied first — standard operator convention).
  • The combined coefficient is the scalar product of the individual
    coefficients.

YHY convention:
  YHY is defined as  Y * Y'  (transpose, not conjugate transpose):
      YHY = Σ_{i,j} c_i c_j  M_i * M_j'
  Terms sharing the same permutation matrix are merged (coefficients summed).
  Terms whose merged coefficient is (numerically) zero are discarded.
"""

module SymmetryOperators

using LinearAlgebra
using Printf

export build_transposition_matrix, build_all_transpositions,
       parse_operator_string, expand_Y_operator, compute_YHY_operator,
       compute_operators, print_results, verify_against_reference


# ─────────────────────────────────────────────────────────────────────────────
# Core matrix builders
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_transposition_matrix(n, a, b) → Matrix{Float64}

Build the n×n transposition matrix P_{a,b}.

`n`   — Jacobi-vector dimension (= number of electrons).
`a,b` — 1-indexed particle labels.
"""
function build_transposition_matrix(n::Int, a::Int, b::Int)::Matrix{Float64}
    P = Matrix{Float64}(I, n, n)   # start from identity
    if a == b
        return P                   # same particle (e.g. P11) → identity
    end
    a, b = minmax(a, b)            # P_{a,b} = P_{b,a}; ensure a < b
    if b - 1 > n
        error("P_{$a,$b} out of range: particle $b needs Jacobi row $(b-1) > n = $n")
    end
    if a == 1
        # Pseudo-particle transposition with the reference particle 1
        # (heliocentric coordinates x_i = r_{i+1} - r_1):
        #   x_{b-1} → -x_{b-1},   x_i → x_i - x_{b-1}  (i ≠ b-1)
        P[:, b - 1] .= -1.0
        return P
    end
    r1 = a - 1                     # 1-indexed Jacobi row
    r2 = b - 1
    P[r1, r1] = 0.0
    P[r2, r2] = 0.0
    P[r1, r2] = 1.0
    P[r2, r1] = 1.0
    return P
end


"""
    build_all_transpositions(n) → Array{Float64,4}

Build the 4D array Transposit of all pair transposition matrices.

Layout (1-based Julia indexing):
    Transposit[:, :, a, b]  corresponds to P_{a,b}
    Transposit[i, j, a, b]  =  (P_{a,b})[i,j]

Shape:  (n, n, n+2, n+2)
Index slot 1 along the particle-label axes is unused (zero-initialised);
1-based access is direct:  Transposit[:, :, 1, 2]  →  P_{12}.
"""
function build_all_transpositions(n::Int)::Array{Float64,4}
    T = zeros(Float64, n, n, n + 2, n + 2)
    for a in 1:(n + 2)
        for b in 1:(n + 2)
            # Guard: only valid particle indices 1..(n+1)
            if a <= n + 1 && b <= n + 1
                T[:, :, a, b] = build_transposition_matrix(n, a, b)
            end
        end
    end
    return T
end


# ─────────────────────────────────────────────────────────────────────────────
# Input parsing
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_operator_string(expr, n) → Vector{Vector{Tuple{Float64, Matrix{Float64}}}}

Parse a product-of-factors expression and return a list of factors.
Each factor is a list of (coefficient, matrix) tuples.
"""
function parse_operator_string(
    expr::AbstractString,
    n::Int,
)::Vector{Vector{Tuple{Float64,Matrix{Float64}}}}

    # Extract the content of every (…) group
    raw_factors = [m.match[2:end-1]
                   for m in eachmatch(r"\([^)]+\)", expr)]
    if isempty(raw_factors)
        error("No parenthesised factors found in: $expr")
    end

    factors = Vector{Vector{Tuple{Float64,Matrix{Float64}}}}()

    for raw in raw_factors
        terms = Tuple{Float64,Matrix{Float64}}[]

        # Split on every '+' or '-' that starts a new token, keeping the sign.
        # We insert a separator before each leading +/-.
        marked = replace(raw, r"(?=[+-])" => "\x00")
        parts  = filter(!isempty, split(marked, "\x00"))

        for part in parts
            part = strip(part)
            isempty(part) && continue

            # Pure ±1 → identity
            if occursin(r"^[+-]?1$", part)
                coeff = startswith(part, "-") ? -1.0 : 1.0
                push!(terms, (coeff, Matrix{Float64}(I, n, n)))
                continue
            end

            # ±P<a><b>[P<c><d>…] → transposition or product of transpositions
            m = match(r"^([+-]?)((?:P\d\d)+)$", part)
            if m !== nothing
                coeff = (m.captures[1] == "-") ? -1.0 : 1.0
                M = Matrix{Float64}(I, n, n)
                for t in eachmatch(r"P(\d)(\d)", m.captures[2])
                    a_idx = parse(Int, t.captures[1])
                    b_idx = parse(Int, t.captures[2])
                    M = M * build_transposition_matrix(n, a_idx, b_idx)
                end
                push!(terms, (coeff, M))
                continue
            end

            error("Unrecognised term in factor: $part")
        end

        isempty(terms) && error("Empty factor: $raw")
        push!(factors, terms)
    end

    return factors
end


# ─────────────────────────────────────────────────────────────────────────────
# Y operator expansion
# ─────────────────────────────────────────────────────────────────────────────

"""
    expand_Y_operator(factors) → (Vector{Float64}, Vector{Matrix{Float64}})

Expand the product of factors into individual (coefficient, matrix) terms.

Enumeration order   : the rightmost factor varies fastest.
Matrix product order: F_k * F_{k-1} * … * F_1  (rightmost applied first).
"""
function expand_Y_operator(
    factors::Vector{Vector{Tuple{Float64,Matrix{Float64}}}},
)::Tuple{Vector{Float64},Vector{Matrix{Float64}}}

    n          = size(factors[1][1][2], 1)
    k          = length(factors)
    term_sizes = [length(f) for f in factors]
    total      = prod(term_sizes)

    coeffs   = Vector{Float64}(undef, total)
    matrices = Vector{Matrix{Float64}}(undef, total)

    # Iterate over the Cartesian product with rightmost-fastest ordering.
    # We compute a mixed-radix counter manually.
    indices = ones(Int, k)   # 1-based index into each factor's term list
    for t in 1:total
        c = 1.0
        M = Matrix{Float64}(I, n, n)

        # Reverse traversal → rightmost factor applied first
        for fi in k:-1:1
            ci, Mi = factors[fi][indices[fi]]
            c *= ci
            M  = M * Mi
        end

        coeffs[t]   = c
        matrices[t] = M

        # Increment mixed-radix counter (rightmost digit varies fastest)
        for fi in k:-1:1
            indices[fi] += 1
            if indices[fi] <= term_sizes[fi]
                break
            else
                indices[fi] = 1  # carry
            end
        end
    end

    return coeffs, matrices
end


# ─────────────────────────────────────────────────────────────────────────────
# Y†Y operator (reduced)
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_YHY_operator(y_coeffs, y_matrices)
        → (Vector{Float64}, Vector{Matrix{Float64}})

Compute YHY = Y† * Y and reduce to unique-matrix terms.

YHY = Σ_{i,j} c_i c_j  M_i * M_j⁻¹

The group-algebra adjoint maps g → g⁻¹, i.e. each representation matrix to
its inverse.  For pure permutation matrices (atoms) the inverse equals the
transpose, but for pseudo-particle transpositions involving the reference
particle (molecules, e.g. P_{1,b}) the matrices are not orthogonal and the
inverse must be used.

Terms sharing the same matrix are merged (coefficients summed); the merge
key is the full integer matrix content, since an argmax-per-row signature
is ambiguous for matrices with -1 entries.
Terms whose merged coefficient is numerically zero are discarded.
"""
function compute_YHY_operator(
    y_coeffs::Vector{Float64},
    y_matrices::Vector{Matrix{Float64}},
)::Tuple{Vector{Float64},Vector{Matrix{Float64}}}

    coeff_sum  = Dict{NTuple,Float64}()
    matrix_for = Dict{NTuple,Matrix{Float64}}()

    for (ci, Mi) in zip(y_coeffs, y_matrices)
        for (cj, Mj) in zip(y_coeffs, y_matrices)
            # Rep matrices are unimodular integer matrices → inverse is
            # integer-valued; rounding removes floating-point noise.
            M   = Mi * round.(inv(Mj))
            key = Tuple(round.(Int, vec(M)))
            coeff_sum[key]  = get(coeff_sum, key, 0.0) + ci * cj
            if !haskey(matrix_for, key)
                matrix_for[key] = M
            end
        end
    end

    yhy_coeffs   = Float64[]
    yhy_matrices = Matrix{Float64}[]

    for sig in sort(collect(keys(coeff_sum)))
        c = coeff_sum[sig]
        if !isapprox(c, 0.0; atol=1e-12)
            push!(yhy_coeffs,   c)
            push!(yhy_matrices, matrix_for[sig])
        end
    end

    return yhy_coeffs, yhy_matrices
end


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_operators(expr, n) → NamedTuple

Full pipeline: build Transposit → parse → expand Y → compute YHY.

Returns a NamedTuple with fields:
    Transposit   :: Array{Float64,4}   shape (n, n, n+2, n+2)
    YCoeff       :: Vector{Float64}
    YMatr        :: Vector{Matrix{Float64}}
    NumYTerms    :: Int
    YHYCoeff     :: Vector{Float64}
    YHYMatr      :: Vector{Matrix{Float64}}
    NumYHYTerms  :: Int
"""
function compute_operators(expr::AbstractString, n::Int)
    Transposit        = build_all_transpositions(n)
    factors           = parse_operator_string(expr, n)
    y_c, y_m          = expand_Y_operator(factors)
    yhy_c, yhy_m      = compute_YHY_operator(y_c, y_m)

    return (
        Transposit   = Transposit,
        YCoeff       = y_c,
        YMatr        = y_m,
        NumYTerms    = length(y_c),
        YHYCoeff     = yhy_c,
        YHYMatr      = yhy_m,
        NumYHYTerms  = length(yhy_c),
    )
end


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    print_results(result; max_terms=5)

Print a short summary of the computed operators.
"""
function print_results(result; max_terms::Int = 5)
    @printf("NumYTerms   = %d\n", result.NumYTerms)
    @printf("NumYHYTerms = %d\n", result.NumYHYTerms)
    println()

    # Permutation matrices get a compact perm label; pseudo-particle
    # matrices (with -1 entries) are shown row-wise.
    function term_label(M)
        if all(x -> x == 0.0 || x == 1.0, M)
            perm = [argmax(M[r, :]) for r in 1:size(M, 1)]
            return "perm=" * string(perm)
        else
            rows = [Int.(M[r, :]) for r in 1:size(M, 1)]
            return "rows=" * string(rows)
        end
    end

    k_show = min(max_terms, result.NumYTerms)
    println("First $k_show Y term(s):")
    for k in 1:k_show
        c = result.YCoeff[k]
        M = result.YMatr[k]
        @printf("  [%3d]  coeff=%+4d   %s\n", k, Int(c), term_label(M))
    end

    println()
    k_show = min(max_terms, result.NumYHYTerms)
    println("First $k_show YHY term(s):")
    for k in 1:k_show
        c = result.YHYCoeff[k]
        M = result.YHYMatr[k]
        @printf("  [%3d]  coeff=%+4d   %s\n", k, Int(c), term_label(M))
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Verification against saved reference text files
# ─────────────────────────────────────────────────────────────────────────────

"""
    verify_against_reference(result, ycoeff_file, ymatr_file,
                              yhycoeff_file, yhymatr_file, n) → Bool

Load four reference text files and check the computed result.

File formats:
  *Coeff.txt — one integer per line
  *Matr.txt  — rows of each n×n matrix, space-separated floats,
               NumTerms × n lines total (no blank separator)

Returns `true` if all checks pass.
"""
function verify_against_reference(
    result,
    ycoeff_file::AbstractString,
    ymatr_file::AbstractString,
    yhycoeff_file::AbstractString,
    yhymatr_file::AbstractString,
    n::Int,
)::Bool

    function load_coeffs(path)
        lines = readlines(path)
        return [parse(Int, strip(l)) for l in lines if !isempty(strip(l))]
    end

    function load_matrices(path, num_terms, dim)
        content = read(path, String)
        vals    = [parse(Float64, v) for v in split(content) if !isempty(v)]
        mats    = Vector{Matrix{Float64}}(undef, num_terms)
        stride  = dim * dim
        for k in 1:num_terms
            chunk     = vals[(k-1)*stride+1 : k*stride]
            mats[k]   = reshape(chunk, dim, dim)'   # row-major → column-major
        end
        return mats
    end

    y_c_ref   = load_coeffs(ycoeff_file)
    yhy_c_ref = load_coeffs(yhycoeff_file)
    y_m_ref   = load_matrices(ymatr_file,   length(y_c_ref),   n)
    yhy_m_ref = load_matrices(yhymatr_file, length(yhy_c_ref), n)

    @printf("Reference: NumYTerms=%d, NumYHYTerms=%d\n",
            length(y_c_ref), length(yhy_c_ref))
    @printf("Computed:  NumYTerms=%d, NumYHYTerms=%d\n",
            result.NumYTerms, result.NumYHYTerms)

    # Y terms — order-sensitive
    y_errors = 0
    for k in 1:length(y_c_ref)
        if !isapprox(result.YCoeff[k], Float64(y_c_ref[k]); atol=1e-12)
            y_errors += 1
        elseif !isapprox(result.YMatr[k], y_m_ref[k]; atol=1e-10)
            y_errors += 1
        end
    end
    @printf("Y  term mismatches : %d\n", y_errors)

    # YHY terms — matched by full matrix content (order-insensitive)
    ref_dict = Dict{NTuple,Float64}()
    for (c_ref, M_ref) in zip(yhy_c_ref, yhy_m_ref)
        key = Tuple(round.(Int, vec(M_ref)))
        ref_dict[key] = Float64(c_ref)
    end

    yhy_errors = 0
    for (c_got, M_got) in zip(result.YHYCoeff, result.YHYMatr)
        key = Tuple(round.(Int, vec(M_got)))
        if !haskey(ref_dict, key)
            yhy_errors += 1
        elseif !isapprox(c_got, ref_dict[key]; atol=1e-12)
            yhy_errors += 1
        end
    end

    missing_terms = length(ref_dict) - result.NumYHYTerms
    @printf("YHY term mismatches: %d  (missing vs ref: %d)\n",
            yhy_errors, missing_terms)

    ok = (y_errors == 0 && yhy_errors == 0 && missing_terms == 0)
    println(ok ? "✓  All terms match reference." : "✗  Some terms do NOT match.")
    return ok
end

end  # module SymmetryOperators


# ─────────────────────────────────────────────────────────────────────────────
# Demo / main script  (run with:  julia symmetry_operators.jl)
# Skipped when this file is include()d as a library.
# ─────────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__

using .SymmetryOperators

# ── Case 1: HD+ ──────────────────────────────────────────────────────────────
println("=" ^ 60)
println("Case: HD+")
println("=" ^ 60)

result_hdp = compute_operators("(P11)", 1)
print_results(result_hdp)
println("Transposit[:, :, 1, 1]  (P11 = identity):")
display(result_hdp.Transposit[:, :, 1, 1])
println()

# ── Case 2: Nitrogen atom ─────────────────────────────────────────────────────
println("=" ^ 60)
println("Case: Nitrogen atom  (N = 7 electrons)")
println("=" ^ 60)

expr_N = "(1+P56)(1+P78)(1-P68)(1-P57)(1-P27-P25)(1-P23-P37-P35)(1-P34-P24-P47-P45)"
result_N = compute_operators(expr_N, 7)
print_results(result_N)

println("Transposit[:, :, 5, 6]  (P56 = swap Jacobi rows 4,5 in 1-indexed):")
display(Int.(result_N.Transposit[:, :, 5, 6]))
println()

# ── Verification ──────────────────────────────────────────────────────────────
ref_files = ("YCoeff.txt", "YMatr.txt", "YHYCoeff.txt", "YHYMatr.txt")
if all(isfile, ref_files)
    println("-" ^ 60)
    println("Verifying Nitrogen result against reference files …")
    verify_against_reference(result_N, ref_files..., 7)
else
    println("(Reference files not found in current directory — skipping verification.)")
end

end  # if abspath(PROGRAM_FILE) == @__FILE__
