#---------------------------------------------------------------------------------------------------------------------
# Module ECG_Matelem_N_C_Pvec  --  N_C_Pvec (complex L=0 Nitrogen) matrix elements with 
# AUTOMATIC-DIFFERENTIATION gradients (June 2026)
#
# Complex analogue of ECG_Matelem_N_Pvec_AD.jl. Computes the complex matrix elements Hkl, Skl and obtains
#   Dk = (dHkl/dvechLk, dSkl/dvechLk)   and   Dl = (dHkl/dvechLl, dSkl/dvechLl)   (each length 2*npt, COMPLEX)
# by forward-mode AD (ForwardDiff) of a type-generic kernel. Because Hkl, Skl are complex functions of the real
# parameter vector, each derivative is obtained from a 4-row Jacobian of [real(H), imag(H), real(S), imag(S)] and
# recombined into a complex column.
#
# vechLk = NonlinParam[k0, 1:npt] with npt = 2*np1: the first np1 entries are the lower triangle of L_k, the next np1
# are the (symmetric) B_k; f_k = exp[-r'(Lk Lk' + i Bk) r]. The bra matrix is Ck = Ak - i Bk (Ak = Lk Lk').
#
# Reference
# [Sharkey2014] K. L. Sharkey, L. Adamowicz, An algorithm for nonrelativistic quantum-mechanical finite-nuclear-mass 
# variational calculations of nitrogen atom in L=0, M=0 states…, J. Chem. Phys. 140, 174112 (2014). 
# [doi:10.1063/1.4873916](https://doi.org/10.1063/1.4873916)
#---------------------------------------------------------------------------------------------------------------------
module ECG_Matelem_N_C_Pvec

import ...ECG_Init: trace_f, verbose1, verbose2, Param, set_parvec
import ...ECG_Init: ONEHALF, ONETHIRD, ONEFOURTH, ONEFIFTH, ONESEVENTH, THREEHALF, TWO, THREE, PI, Mini
import ...ECG_Param: all_real

using Parameters, LinearAlgebra, ForwardDiff

#-------------------------------------------------------------------------------------------------------------------------------
# Type-generic complex kernel: (vechLk, vechLl) -> (Hkl, Skl). All scratch arrays use the promoted element type so that
# ForwardDiff Duals (real) propagate through the complex arithmetic, the matrix inverse/determinant and the 36-term sum.
#-------------------------------------------------------------------------------------------------------------------------------
function _HS(vechLk, vechLl, n, np1, M, Q0, Q, covec, pp, parvec)
    T  = promote_type(eltype(vechLk), eltype(vechLl))
    Lk = zeros(T, n, n); Ll = zeros(T, n, n)
    Bk = zeros(T, n, n); Bl = zeros(T, n, n)
    idx = 1
    for i in 1:n, j in i:n
        Lk[j,i] = vechLk[idx]; Ll[j,i] = vechLl[idx]; idx += 1
    end
    # all_real == "T" enforces purely real correlated Gaussians: leave the imaginary B blocks at zero so
    # Ck = Ak, tCl = tAl and the energy is independent of the B parameters (their AD gradient is then exactly 0).
    if all_real != "T"
        idx = np1 + 1
        for i in 1:n, j in i:n
            Bk[i,j] = vechLk[idx]; Bk[j,i] = vechLk[idx]
            Bl[i,j] = vechLl[idx]; Bl[j,i] = vechLl[idx]
            idx += 1
        end
    end
    Ak = Lk*transpose(Lk); Al = Ll*transpose(Ll)
    tAl = zeros(T, n, n); tBl = zeros(T, n, n)
    for i in 1:n, j in 1:n
        tAl[i,j] = Al[pp[i+n], pp[j+n]]            # P'AlP via the Pvec gather
        tBl[i,j] = Bl[pp[i+n], pp[j+n]]
    end
    Ck   = Complex.(Ak,  -Bk)                       # contains Ck* (Ak - i Bk)
    tCl  = Complex.(tAl,  tBl)
    tCkl = Ck .+ tCl
    iC   = inv(tCkl)
    dC   = det(tCkl)
    AAk = iC*Ck; AAl = iC*tCl
    MAk = M*Ck;  MAl = M*tCl
    AAkMAl = AAk*MAl; AAlMAk = AAl*MAk
    AAkM = AAk*M
    AMA = AAkMAl*iC; ATA = AAlMAk*iC
    MAlA = MAl*iC
    tau = tr(AAkMAl)
    FAC = ONEFOURTH*(PI^(n*THREEHALF))*(dC^(-THREEHALF))
    CT  = Complex{T}
    sumS = zero(CT); sumT = zero(CT); sumV = zero(CT)
    for ii in 1:36
        xk=parvec[ii,1]; xl=parvec[ii,2]; yk=parvec[ii,3]; yl=parvec[ii,4]; zk=parvec[ii,5]; zl=parvec[ii,6]
        tS = iC[xk,xl]*iC[yk,yl]*iC[zk,zl]
        sumS += covec[ii]*tS
        tT = iC[yk,yl]*iC[zk,zl]*(M[xk,xl]-MAlA[xk,xl]-AAkM[xk,xl]+ONEHALF*(AMA[xk,xl]+AMA[xl,xk]+ATA[xk,xl]+ATA[xl,xk])) +
             iC[xk,xl]*iC[zk,zl]*(M[yk,yl]-MAlA[yk,yl]-AAkM[yk,yl]+ONEHALF*(AMA[yk,yl]+AMA[yl,yk]+ATA[yk,yl]+ATA[yl,yk])) +
             iC[xk,xl]*iC[yk,yl]*(M[zk,zl]-MAlA[zk,zl]-AAkM[zk,zl]+ONEHALF*(AMA[zk,zl]+AMA[zl,zk]+ATA[zk,zl]+ATA[zl,zk]))
        sumT += covec[ii]*tT
        tV = zero(CT)
        for i in 1:n
            Xi = iC[i,i]
            ex = iC[i,xk]*iC[xl,i]; ey = iC[i,yk]*iC[yl,i]; ez = iC[i,zk]*iC[zl,i]
            R = tS - ONETHIRD*(iC[xk,xl]*iC[yk,yl]*ez + iC[xk,xl]*ey*iC[zk,zl] + ex*iC[yk,yl]*iC[zk,zl])/Xi +
                     ONEFIFTH*(iC[xk,xl]*ey*ez + ex*iC[yk,yl]*ez + ex*ey*iC[zk,zl])/(Xi^TWO) -
                     ONESEVENTH*ex*ey*ez/(Xi^THREE)
            tV += Q0*Q[i]*(R/sqrt(Xi))
        end
        for i in 1:n, j in i+1:n
            Xi = iC[i,i]+iC[j,j]-TWO*iC[i,j]
            ex = iC[i,xk]*iC[xl,i]+iC[j,xk]*iC[xl,j]-iC[i,xk]*iC[xl,j]-iC[i,xl]*iC[xk,j]
            ey = iC[i,yk]*iC[yl,i]+iC[j,yk]*iC[yl,j]-iC[i,yk]*iC[yl,j]-iC[i,yl]*iC[yk,j]
            ez = iC[i,zk]*iC[zl,i]+iC[j,zk]*iC[zl,j]-iC[i,zk]*iC[zl,j]-iC[i,zl]*iC[zk,j]
            R = tS - ONETHIRD*(iC[xk,xl]*iC[yk,yl]*ez + iC[xk,xl]*ey*iC[zk,zl] + ex*iC[yk,yl]*iC[zk,zl])/Xi +
                     ONEFIFTH*(iC[xk,xl]*ey*ez + ex*iC[yk,yl]*ez + ex*ey*iC[zk,zl])/(Xi^TWO) -
                     ONESEVENTH*ex*ey*ez/(Xi^THREE)
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
    @unpack n, npt, PseudoCharge0, PseudoCharge, MassMatrix, NonlinParam, PP, covec = param
    np1 = Int(n*(n+1)/2)

    pp     = PP[:, j0]
    parvec = set_parvec(param, k0, l0, j0)
    vechLk = collect(NonlinParam[k0, 1:npt])
    vechLl = collect(NonlinParam[l0, 1:npt])

    kernel(vk, vl) = _HS(vk, vl, n, np1, MassMatrix, PseudoCharge0, PseudoCharge, covec, pp, parvec)

    # Guard against a singular/near-singular tAkl: a singular trial returns zeros so it is rejected 
    # rather than crashing the run.
    local Hkl, Skl
    try
        Hkl, Skl = kernel(vechLk, vechLl)
    catch
        return 0.0+0.0im, 0.0+0.0im, 0, 0
    end
    if !isfinite(Hkl) || !isfinite(Skl)
        return 0.0+0.0im, 0.0+0.0im, 0, 0
    end

    # AD gradients of the COMPLEX (Hkl, Skl) w.r.t. the real parameter vector, via a [re,im] Jacobian.
    Dk = 0
    if grad_k
        Jk = ForwardDiff.jacobian(vechLk) do v
            h, s = kernel(v, vechLl)
            [real(h), imag(h), real(s), imag(s)]
        end
        Dk = vcat(Jk[1,:] .+ im.*Jk[2,:], Jk[3,:] .+ im.*Jk[4,:])
    end
    Dl = 0
    if grad_l
        Jl = ForwardDiff.jacobian(vechLl) do v
            h, s = kernel(vechLk, v)
            [real(h), imag(h), real(s), imag(s)]
        end
        Dl = vcat(Jl[1,:] .+ im.*Jl[2,:], Jl[3,:] .+ im.*Jl[4,:])
    end

    return Hkl, Skl, Dk, Dl
end

end # module ECG_Matelem_N_C_Pvec
