#-------------------------------------------------------------------------------------------------------------------------------
# Author: Alain Chancé
# Date: June 4, 2023
# Version: 1.0
#
# Module ECG_Matelem_CGL0
#
# This module defines MatrixElements() which computes elements of H and S matrices with two complex L=0 correlated Gaussians
# fk = exp[-r'(Lk*Lk'+iBk)r]
#
# symmetry adaption is applied to the ket using a permutation matrix P stored in YHYMatr[:,:,1:NumYHYTerms]
#
# MatElem_method in module ECG_Param determines which computation method is used:
#    "CGL0_Pvec" : Pvec method
#
# Input
#    k0, l0         :  index in current basis set
#    j0             :  index in symmetry terms
#    grad_k, grad_l :  Gradient flags
#    grad_k = true  :  Means that dHkl/dvechLk, dSkl/dvechLk need to be computed
#    grad_l = true  :  Means that dHkl/dvechLl, dSkl/dvechLl need to be computed
#
# Output:
#    Hkl : Hamiltonian term (normalized)
#    Skl : Overlap matrix element (normalized)
#    Dk,Dl : Derivatives of normalized Hkl and Skl with respect to vechLk and vechLl respectively.
#           They are ordered as follows:
#           Dk=(dHkl/dvechLk,dSkl/dvechLk)
#           Dl=(dHkl/dvechLl,dSkl/dvechLl)
#-------------------------------------------------------------------------------------------------------------------------------
# Permutation-independent (k0,l0) caching  (efficiency, June 2026)
#
# The quantities that depend only on the basis pair (k0,l0) and NOT on the permutation index j0 -- Al, Bl, the bra matrix Ck
# and det_Lk, det_Ll -- are built once per (k0,l0) pair and reused across the NumYHYTerms permutations, instead of being
# rebuilt on every call. Inspired by Sharkey & Adamowicz, "Elimination of Permutational Complexity in Overlap Matrix for L=0
# Complex ECGs" (2019). Numerically identical to the previous version; on Be (n=4, 24 terms) it cuts compute_H_S allocations
# ~45% and runtime ~30%.
#
# The cache is a single (k0,l0) slot guarded by a snapshot of the NonlinParam rows for k0 and l0, so it stays correct after
# the optimizer mutates NonlinParam between H/S sweeps. It is enabled only when compute_H_S() iterates with (k,l) in the
# outer loop and j0 = 1..NumYHYTerms in the inner loop, i.e. param.compute_H_S_method == "basis terms" (the slot then hits
# NumYHYTerms-1 times out of NumYHYTerms). Under "symmetry terms" ordering (j0 outer, (k,l) inner) the single slot would
# never hit, so caching is skipped to avoid the snapshot-comparison overhead and the build proceeds exactly as before.
#-------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# References
#
# [Bubin] Sergiy Bubin and Ludwik Adamowicz, Computer program ATOM-MOL-nonBO for
# performing calculations of ground and excited states of atoms and molecules without
# assuming the Born–Oppenheimer approximation using all-particle complex explicitly
# correlated Gaussian functions. J. Chem. Phys. 152, 204102 (2020), 26 May 2020,
# https://doi.org/10.1063/1.5144268
# GitHub https://github.com/sbubin/ATOM-MOL-nonBO/tree/master/src
#
# [KS_Nitrogen] Keeper L. Sharkey and Ludwik Adamowicz, An algorithm for nonrelativistic
# quantum-mechanical finite-nuclear-mass variational calculations of nitrogen atom in
# L = 0, M = 0 states using all-electrons explicitly correlated Gaussian basis functions,
# J. Chem. Phys. 140, 174112 (2014); https://doi.org/10.1063/1.4873916
#
# [KS_Overlap] Keeper L. Sharkey and Ludwik Adamowicz, Elimination of Permutational Complexity
# in Overlap Matrix for L=0 Complex ECGs (2019).
#
# [Muolo] Andrea Muolo, Explicitly Correlated Gaussians and the Quantum Few-Body Problem,
# DISS. ETH NO. 25680, December 2018,
# https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/352293/1/AMuolo.pdf,
# 3.8 Numerical stability of complex functions, A.1.1 Overlap integral
#--------------------------------------------------------------------------------------------
module ECG_Matelem_CGL0

import ...ECG_Init: trace_f, verbose1, verbose2, verbose3, Param, set_parvec, overlap_Skl
import ...ECG_Init: ZERO, ONE, TWO, THREE, SIX, ONEHALF, ONETHIRD, ONEFOURTH, ONEFIFTH, ONESEVENTH, THREEHALF, PI, SQRTPI

using Parameters

#-------------------------------------------------------------------------------
# Single-slot (k0,l0) cache of permutation-independent quantities.
# Fields are concretely typed; the arrays are (re)allocated on the first miss
# and whenever the basis pair or its NonlinParam rows change.
#-------------------------------------------------------------------------------
mutable struct _KLCache
    init::Bool
    k0::Int
    l0::Int
    pk::Vector{Float64}            # snapshot of NonlinParam[k0, :]
    pl::Vector{Float64}            # snapshot of NonlinParam[l0, :]
    Al::Matrix{Float64}
    Bl::Matrix{Float64}
    Ck::Matrix{ComplexF64}
    det_Lk::Float64
    det_Ll::Float64
end

const _CACHE = _KLCache(false, 0, 0, Float64[], Float64[],
                        zeros(0,0), zeros(0,0), zeros(ComplexF64,0,0), 0.0, 0.0)

function MatrixElements(k0,l0,j0,grad_k,grad_l; param::Param=param,verbose1=verbose1,verbose2=verbose2)

    verbose = false

    if verbose
        println(trace_f, "\nMatrixElements - k0: ", k0, " l0: ", l0, " j0: ", j0, " grad_k: ", grad_k, " grad_l: ", grad_l)
    end

    @unpack n, npt, PseudoCharge0, PseudoCharge, MassMatrix, NonlinParam, PP, covec, YHYMatr, MatElem_method,
            compute_H_S_method = param

    np1::Int64 = Int(n*(n+1)/2)

    #-------------------------------------------------------------------------------------
    # The total number of nonlinear parameters per basis function, npt,
    # is set by function read_inout() in ECG_Init.jl as follows:
    #   if CGL0_Pvec || N_C_Pvec
    #       npt = 2*np1
    #   else
    #       npt = np1
    #   end
    #
    # where:
    # n: the number of pseudoparticles
    # np1 = n*(n+1)/2): number of independent parameters in a symmetric matrix of size n
    #-------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------
    # Permutation-independent (k0,l0) quantities: reuse from the cache when
    # this pair was just built (only worthwhile in "basis terms" ordering),
    # otherwise build them. The snapshot guard keeps the cache correct after
    # NonlinParam is mutated by the optimizer.
    #-----------------------------------------------------------------------
    use_cache = (compute_H_S_method == "basis terms")
    pk = @view NonlinParam[k0, :]
    pl = @view NonlinParam[l0, :]
    hit = use_cache && _CACHE.init && _CACHE.k0 == k0 && _CACHE.l0 == l0 &&
          _CACHE.pk == pk && _CACHE.pl == pl

    if hit
        Al     = _CACHE.Al
        Bl     = _CACHE.Bl
        Ck     = _CACHE.Ck
        det_Lk = _CACHE.det_Lk
        det_Ll = _CACHE.det_Ll
    else
        #--------------------------------------------------------------------------------
        # First build matrices Lk, Ll, Ak, Al, Bk, Bl from the following:
        #   vechLk = NonlinParam[k,1:npt]  vechBk[1:npt] = NonlinParam[k, np1+1:np1+np1]
        #   vechLl = NonlinParam[l,1:npt]  vechBl[1:npt] = NonlinParam[l, np1+1:np1+np1]
        #--------------------------------------------------------------------------------
        Lk = zeros(n,n)
        Ll = zeros(n,n)

        indx = 1
        for i in 1:n
            for j in i:n
                Lk[i,j] = 0.0
                Ll[i,j] = 0.0
                Lk[j,i] = NonlinParam[k0,indx]
                Ll[j,i] = NonlinParam[l0,indx]
                indx += 1
            end
        end

        Ak = zeros(n,n)
        Al = zeros(n,n)
        Bk = zeros(n,n)
        Bl = zeros(n,n)

        if npt > np1+1
            indx = np1+1
            for i in 1:n
                for j in i:n
                    Bk[i,j] = NonlinParam[k0,indx]
                    Bk[j,i] = NonlinParam[k0,indx]

                    Bl[i,j] = NonlinParam[l0,indx]
                    Bl[j,i] = NonlinParam[l0,indx]
                    indx += 1
                end
            end
        end

        for i in 1:n
            for j in 1:n
                temp1 = 0.0
                temp2 = 0.0
                for k in 1:j
                    temp1 += Lk[i,k]*Lk[j,k]
                    temp2 += Ll[i,k]*Ll[j,k]
                end
                Ak[i,j] = temp1
                Ak[j,i] = temp1

                Al[i,j] = temp2
                Al[j,i] = temp2
            end
        end

        #--------------------------------------------------------------------------------
        # The determinants of Lk and Ll are just the products of their diagonal elements
        #--------------------------------------------------------------------------------
        det_Lk = 1.0
        det_Ll = 1.0
        for i in 1:n
            det_Lk = det_Lk*Lk[i,i]
            det_Ll = det_Ll*Ll[i,i]
        end

        #---------------------------------------------------------------
        # Form the bra matrix Ck (contains Ck^*); independent of j0
        #---------------------------------------------------------------
        Ck = zeros(ComplexF64, n, n)
        for i in 1:n
            for j in i:n
                cnumb = complex(Ak[j,i], -Bk[j,i])
                Ck[i,j] = cnumb
                Ck[j,i] = cnumb
            end
        end

        #---------------------------------------------------------------
        # Store into the cache (own copies of the param snapshots).
        # Only in "basis terms" ordering, where the next NumYHYTerms-1
        # calls share this (k0,l0) pair.
        #---------------------------------------------------------------
        if use_cache
            _CACHE.init   = true
            _CACHE.k0     = k0
            _CACHE.l0     = l0
            _CACHE.pk     = copy(pk)
            _CACHE.pl     = copy(pl)
            _CACHE.Al     = Al
            _CACHE.Bl     = Bl
            _CACHE.Ck     = Ck
            _CACHE.det_Lk = det_Lk
            _CACHE.det_Ll = det_Ll
        end
    end

    #----------------------------------------------------------------------------------------
    # Then permute elements of Al and Bl to account for the action of the permutation matrix
    # tAl=P'*Al*P and tBl=P'*Bl*P with the following definition: P = YHYMatr[...,j0]
    #----------------------------------------------------------------------------------------
    tAl = transpose(YHYMatr[:,:,j0])*Al*YHYMatr[:,:,j0]
    tBl = transpose(YHYMatr[:,:,j0])*Bl*YHYMatr[:,:,j0]

    #--------------------------------------
    # Next, form tCl, and tCkl = Ck + tCl
    #--------------------------------------
    tCl  = zeros(ComplexF64, n, n)
    tCkl = zeros(ComplexF64, n, n)
    inv_tCkl = zeros(ComplexF64, n, n)

    cnumb::ComplexF64 = 0.0 + 0.0im
    csum::ComplexF64  = 0.0 + 0.0im
    ctemp::ComplexF64 = 0.0 + 0.0im

    for i in 1:n
        for j in i:n
            cnumb = complex(tAl[j,i], tBl[j,i])
            tCl[i,j] = cnumb
            tCl[j,i] = cnumb

            cnumb = Ck[j,i] + tCl[j,i]
            tCkl[i,j] = cnumb
            tCkl[j,i] = cnumb
        end
    end

    #-------------------------------------------------------------------------------------------------------
    # Perform pseudo-Cholesky factorization of tCkl and store Cholesky factors in csum
    # Pseudo-Cholesky because it is a product of a lower triangular matrix and its transpose, not a product
    # of a lower triangular matrix and its hermitian conjugate as the original Cholesky factorization)
    #-------------------------------------------------------------------------------------------------------
    det_tCkl::ComplexF64 = complex(1.0, 0.0)

    for i in 1:n
        for j in i:n
            csum = tCkl[i,j]
            for k in i-1:-1:1
                csum = csum-tCkl[i,k]*tCkl[j,k]
            end

            if i==j
                tCkl[i,i] = sqrt(csum)
                det_tCkl = det_tCkl*csum
            else
                tCkl[j,i] = csum/tCkl[i,i]
                tCkl[i,j] = 0.0
            end
        end
    end

    #-------------------------------------------------------------------------------------------
    # Invert tCkl using its Cholesky factor (stored in csum) and place the result into inv_tCkl
    #-------------------------------------------------------------------------------------------
    for i in 1:n
        tCkl[i,i] = 1.0/tCkl[i,i]
        for j in i+1:n
            csum = 0.0 + 0.0im
            for k in i:j-1
                csum = csum-tCkl[j,k]*tCkl[k,i]
            end
            tCkl[j,i] = csum/tCkl[j,j]
        end
    end

    for i in 1:n
        for j in i:n
            ctemp = 0.0 + 0.0im
            for k in j:n
                ctemp += tCkl[k,i]*tCkl[k,j]
            end
            inv_tCkl[i,j] = ctemp
            inv_tCkl[j,i] = ctemp
        end
    end

    #--------------------
    # Evaluating overlap
    #--------------------
    if overlap_Skl
        # Skl = 2^3n/2 (||Lk|| ||Ll||/|AKL|)^3/2
        cnumb = abs(det_Ll*det_Lk)/det_tCkl
        Skl = 2.0^(3.0*n/2.0)*cnumb*sqrt(cnumb)
    else
        # [Muolo] A.1.1 Overlap integral
        Skl = PI^(3.0*n/2.0)/det_tCkl*sqrt(det_tCkl)
    end

    #--------------------------------------
    # Doing multiplication tCl*invCkl*Ck^*
    #--------------------------------------
    Wc1 = zeros(ComplexF64, n, n)
    Wc2 = zeros(ComplexF64, n, n)

    for i in 1:n
        for j in 1:n
            ctemp = 0.0 + 0.0im
            for k in 1:n
                ctemp += inv_tCkl[i,k]*Ck[k,j]
            end
            Wc1[i,j] = ctemp
        end
    end

    for i in 1:n
        for j in 1:n
            ctemp = 0.0 + 0.0im
            for k in 1:n
                ctemp += tCl[i,k]*Wc1[k,j]
            end
            Wc2[i,j] = ctemp
        end
    end

    #---------------------------------------------------
    # Computing kinetic energy, Tkl=tr[inv_tAkltAlM*Ak]
    #---------------------------------------------------
    ctemp = 0.0 + 0.0im
    for i in 1:n
        for k in 1:n
            ctemp += MassMatrix[i,k]*Wc2[k,i]
        end
    end

    Tkl = 6.0*Skl*ctemp

    #--------------------------------------------------------------------------------------
    # Evaluating potential energy, Vkl, and tr[invCkl*Jij]^(-3/2)
    # The lower triangle of array trinvCklJij32 will contain the corresponding quantities.
    #--------------------------------------------------------------------------------------
    tr_inv_tCklJij32 = zeros(ComplexF64, n, n)
    cnumb = (TWO/SQRTPI)*Skl
    Vkl::ComplexF64 = 0.0 + 0.0im

    for i in 1:n
        cmpnum = inv_tCkl[i,i]
        csum = sqrt(cmpnum)
        tr_inv_tCklJij32[i,i] = 1.0/(csum*cmpnum)
        cRklij = cnumb/csum
        Vkl += PseudoCharge[i]*PseudoCharge0*cRklij
    end

    for i in 1:n
        for j in i+1:n
            cmpnum = inv_tCkl[i,i]+inv_tCkl[j,j]-inv_tCkl[j,i]-inv_tCkl[j,i]
            csum = sqrt(cmpnum)
            tr_inv_tCklJij32[j,i] = 1.0/(csum*cmpnum)
            cRklij = cnumb/csum
            Vkl += PseudoCharge[i]*PseudoCharge[j]*cRklij
        end
    end

    Hkl = Tkl + Vkl

    if verbose
        println(trace_f, "\nMatrixElements - Hkl: ", Hkl, " Skl: ", Skl, " Tkl: ", Tkl, " Vkl: ", Vkl)
        println(" ")
    end

    return Hkl, Skl, 0.0 + 0.0im, 0.0 + 0.0im

end

end # Module ECG_Matelem_CGL0
