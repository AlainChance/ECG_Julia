#-------------------------------------------------------------------------------------------------------------------------------
# Module ECG_Matelem_N_matr  --  Nitrogen "matrix-operations" matrix elements with AUTOMATIC-DIFFERENTIATION gradients
#
# Selected by MatElem_method = "N_Matrix_operations". It computes the real L=0, M=0 nitrogen matrix elements Hkl, Skl and
# obtains
#   Dk = (dHkl/dvechLk, dSkl/dvechLk)   and   Dl = (dHkl/dvechLl, dSkl/dvechLl)     (each length 2*npt)
# by forward-mode AD (ForwardDiff) of a single type-generic kernel _HS(vechLk, vechLl) — exactly the scheme of
# ECG_Matelem_N_Pvec. The ONLY difference from N_Pvec is how the permuted ket width matrix is formed: here by the explicit
# permutation-matrix product  tAl = P' * Al * P  with  P = YHYMatr[:,:,j0]  (the "matrix-operations" method), rather than the
# Pvec index gather. N_matr therefore reproduces the N_Pvec matrix elements by an independent route and serves as a
# cross-check (no Fortran nitrogen reference exists).
#
# Why AD works: _HS allocates every intermediate with the PROMOTED element type T of its inputs, so ForwardDiff Duals flow
# through. The integer/physical data (n, P = YHYMatr[:,:,j0], parvec from set_parvec, MassMatrix M, pseudo-charges Q0/Q,
# covec) are constant w.r.t. the nonlinear parameters.
#
# Rewritten June 2026 with the type-generic AD kernel.
#
# References
# [Sharkey2014] K. L. Sharkey, L. Adamowicz, An algorithm for nonrelativistic quantum-mechanical finite-nuclear-mass
# variational calculations of nitrogen atom in L=0, M=0 states…, J. Chem. Phys. 140, 174112 (2014). [doi:10.1063/1.4873916]
# [ATOMMOLnonBO] S. Bubin, L. Adamowicz, Computer program `atom-mol-nonBO`…, J. Chem. Phys. 152, 204102 (2020).
# [Muolo] A. Muolo, Explicitly Correlated Gaussians and the Quantum Few-Body Problem, DISS. ETH NO. 25680 (2018).
#-------------------------------------------------------------------------------------------------------------------------------
module ECG_Matelem_N_matr

import ...ECG_Init: trace_f, verbose1, verbose2, Param, set_parvec
import ...ECG_Init: ONEHALF, ONETHIRD, ONEFOURTH, ONEFIFTH, ONESEVENTH, THREEHALF, PI, Mini

using Parameters, LinearAlgebra, ForwardDiff

#-------------------------------------------------------------------------------------------------------------------------------
# Type-generic kernel: (vechLk, vechLl) -> (Hkl, Skl). Identical physics to ECG_Matelem_N_Pvec, except the permuted ket width
# matrix is tAl = P'*Al*P with the explicit permutation matrix P (= YHYMatr[:,:,j0]). All scratch uses the promoted element
# type T so ForwardDiff Duals flow through.
#-------------------------------------------------------------------------------------------------------------------------------
function _HS(vechLk, vechLl, n, M, Q0, Q, covec, P, parvec)
    T  = promote_type(eltype(vechLk), eltype(vechLl))
    Lk = zeros(T, n, n); Ll = zeros(T, n, n)
    idx = 1
    for i in 1:n, j in i:n
        Lk[j,i] = vechLk[idx]; Ll[j,i] = vechLl[idx]; idx += 1
    end
    Ak = Lk*transpose(Lk); Al = Ll*transpose(Ll)
    tAl = transpose(P)*Al*P                                  # P'AlP via the explicit permutation matrix (matrix-operations method)
    tAkl = Ak .+ tAl
    iA   = inv(tAkl)
    dA   = det(tAkl)
    AAk = iA*Ak; AAl = iA*tAl
    MAk = M*Ak;  MAl = M*tAl
    AAkMAl = AAk*MAl; AAlMAk = AAl*MAk
    AAkM = AAk*M;  AAlM = AAl*M
    AMA = AAkMAl*iA; ATA = AAlMAk*iA
    MAkA = MAk*iA;   MAlA = MAl*iA
    tau = tr(AAkMAl)
    FAC = ONEFOURTH*(PI^(n*THREEHALF))*(dA^(-THREEHALF))
    sumS = zero(T); sumT = zero(T); sumV = zero(T)
    for ii in 1:36
        xk=parvec[ii,1]; xl=parvec[ii,2]; yk=parvec[ii,3]; yl=parvec[ii,4]; zk=parvec[ii,5]; zl=parvec[ii,6]
        tS = iA[xk,xl]*iA[yk,yl]*iA[zk,zl]
        sumS += covec[ii]*tS
        tT = iA[yk,yl]*iA[zk,zl]*(M[xk,xl]-MAlA[xk,xl]-AAkM[xk,xl]+ONEHALF*(AMA[xk,xl]+AMA[xl,xk]+ATA[xk,xl]+ATA[xl,xk])) +
             iA[xk,xl]*iA[zk,zl]*(M[yk,yl]-MAlA[yk,yl]-AAkM[yk,yl]+ONEHALF*(AMA[yk,yl]+AMA[yl,yk]+ATA[yk,yl]+ATA[yl,yk])) +
             iA[xk,xl]*iA[yk,yl]*(M[zk,zl]-MAlA[zk,zl]-AAkM[zk,zl]+ONEHALF*(AMA[zk,zl]+AMA[zl,zk]+ATA[zk,zl]+ATA[zl,zk]))
        sumT += covec[ii]*tT
        tV = zero(T)
        for i in 1:n
            Xi = iA[i,i]
            ex = iA[i,xk]*iA[xl,i]; ey = iA[i,yk]*iA[yl,i]; ez = iA[i,zk]*iA[zl,i]
            R = tS - ONETHIRD*(iA[xk,xl]*iA[yk,yl]*ez + iA[xk,xl]*ey*iA[zk,zl] + ex*iA[yk,yl]*iA[zk,zl])/Xi +
                     ONEFIFTH*(iA[xk,xl]*ey*ez + ex*iA[yk,yl]*ez + ex*ey*iA[zk,zl])/Xi^2 -
                     ONESEVENTH*ex*ey*ez/Xi^3
            tV += Q0*Q[i]*(R/sqrt(Xi))
        end
        for i in 1:n, j in i+1:n
            Xi = iA[i,i]+iA[j,j]-2*iA[i,j]
            ex = iA[i,xk]*iA[xl,i]+iA[j,xk]*iA[xl,j]-iA[i,xk]*iA[xl,j]-iA[i,xl]*iA[xk,j]
            ey = iA[i,yk]*iA[yl,i]+iA[j,yk]*iA[yl,j]-iA[i,yk]*iA[yl,j]-iA[i,yl]*iA[yk,j]
            ez = iA[i,zk]*iA[zl,i]+iA[j,zk]*iA[zl,j]-iA[i,zk]*iA[zl,j]-iA[i,zl]*iA[zk,j]
            R = tS - ONETHIRD*(iA[xk,xl]*iA[yk,yl]*ez + iA[xk,xl]*ey*iA[zk,zl] + ex*iA[yk,yl]*iA[zk,zl])/Xi +
                     ONEFIFTH*(iA[xk,xl]*ey*ez + ex*iA[yk,yl]*ez + ex*ey*iA[zk,zl])/Xi^2 -
                     ONESEVENTH*ex*ey*ez/Xi^3
            tV += Q[i]*Q[j]*(R/sqrt(Xi))
        end
        sumV += covec[ii]*tV
    end
    Skl = sumS*FAC*ONEHALF
    Tkl = FAC*(sumS*3*tau + sumT)
    Vkl = sumV*FAC*(PI^(-ONEHALF))
    return Tkl+Vkl, Skl
end

function MatrixElements(k0,l0,j0,grad_k,grad_l; param::Param=param, verbose1=verbose1, verbose2=verbose2)
    @unpack n, npt, PseudoCharge0, PseudoCharge, MassMatrix, NonlinParam, YHYMatr, covec = param

    P      = YHYMatr[:, :, j0]                       # explicit permutation matrix for symmetry term j0
    parvec = set_parvec(param, k0, l0, j0)
    vechLk = collect(NonlinParam[k0, 1:npt])
    vechLl = collect(NonlinParam[l0, 1:npt])

    kernel(vk, vl) = _HS(vk, vl, n, MassMatrix, PseudoCharge0, PseudoCharge, covec, P, parvec)

    # Guard a singular/near-singular tAkl: a singular trial returns zeros so it is rejected rather than crashing the run.
    local Hkl, Skl
    try
        Hkl, Skl = kernel(vechLk, vechLl)
    catch
        return 0.0, 0.0, 0.0, 0.0
    end
    if !isfinite(Hkl) || !isfinite(Skl)
        return 0.0, 0.0, 0.0, 0.0
    end

    # AD gradients: jacobian rows are dHkl/dvech and dSkl/dvech; Dk = (dHkl/dvechLk, dSkl/dvechLk)
    Dk = 0.0
    if grad_k
        Jk = ForwardDiff.jacobian(v -> collect(kernel(v, vechLl)), vechLk)
        Dk = vcat(Jk[1, :], Jk[2, :])
    end
    Dl = 0.0
    if grad_l
        Jl = ForwardDiff.jacobian(v -> collect(kernel(vechLk, v)), vechLl)
        Dl = vcat(Jl[1, :], Jl[2, :])
    end

    return Hkl, Skl, Dk, Dl
end

end # module ECG_Matelem_N_matr
