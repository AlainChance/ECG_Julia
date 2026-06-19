#-------------------------------------------------------------------------------------------------------------------------------------
# Author: Alain Chancé
# Date: June 4, 2023
# Version: 1.0
#
# Module ECG_Matelem_RGL0
#
# This module defines MatrixElements() which computes elements of H and S matrices for real gaussians (L=0) all particles in s-states
# for instance He and Li atoms.
#
# MatElem_method in module ECG_Param determines which computation method is used:
#    "RGL0_Pvec" : Pvec method
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
# [Muolo] Andrea Muolo, Explicitly Correlated Gaussians and the Quantum Few-Body Problem, 
# DISS. ETH NO. 25680, December 2018, 
# https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/352293/1/AMuolo.pdf, 
# 3.8 Numerical stability of complex functions, A.1.1 Overlap integral
#--------------------------------------------------------------------------------------------
module ECG_Matelem_RGL0

import ...ECG_Init: trace_f, verbose1, verbose2, verbose3, Param, set_parvec, Mini, overlap_Skl
import ...ECG_Init: ZERO, ONE, TWO, THREE, SIX, ONEHALF, ONETHIRD, ONEFOURTH, ONEFIFTH, ONESEVENTH, THREEHALF, PI, SQRTPI

using Parameters, LinearAlgebra

function MatrixElements(k0,l0,j0,grad_k,grad_l; param::Param=param,verbose1=verbose1,verbose2=verbose2)

    verbose = verbose2
    
    if verbose
        println(trace_f, "\nMatrixElements - k0: ", k0, " l0: ", l0, " j0: ", j0, " grad_k: ", grad_k, " grad_l: ", grad_l)
    end

    @unpack n, npt, PseudoCharge0, PseudoCharge, MassMatrix, NonlinParam, PP, covec, YHYMatr, MatElem_method = param
    
    Ak = zeros(n,n)
    Al = zeros(n,n)
    
    tAl = zeros(n,n)
    tAkl = zeros(n,n)
    
    inv_tAkl = zeros(n,n)
    inv_tAkltAlM = zeros(n,n)
    tr_inv_tAklJij32 = zeros(n,n)
    tr_inv_tAklJij32 = zeros(n,n)
    
    W2 = zeros(n,n)
    
    Piraised3n2 = PI^(3.0*n/2.0)
    
#------------------------------------------------------------------------------------------------------
# First we build matrices Lk, Ll, Ak, Al from vechLk = NonlinParam[k,...], vechLl = NonlinParam[l,...]
#------------------------------------------------------------------------------------------------------
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

    if verbose
        println(trace_f,"\nLk")
        show(trace_f, "text/plain", Lk)
        println(trace_f," ")

        println(trace_f,"\nLl")
        show(trace_f, "text/plain", Ll)
        println(trace_f," ")
    end
    
    for i in 1:n
        for j in i:n
            temp1 = 0.0
            temp2 = 0.0
            for k in 1:j
                temp1=temp1+Lk[i,k]*Lk[j,k]
                temp2=temp2+Ll[i,k]*Ll[j,k]
            end
            Ak[i,j]=temp1
            Ak[j,i]=temp1
            Al[i,j]=temp2
            Al[j,i]=temp2
        end
    end

    if verbose
        println(trace_f,"\nAk")
        show(trace_f, "text/plain", Ak)
        println(trace_f," ")

        println(trace_f,"\nAl")
        show(trace_f, "text/plain", Al)
        println(trace_f," ")
    end
    
#------------------------------------------------------------------------------------------------
# Then we permute elements of Al to account for the action of the permutation matrix tAl=P'*Al*P
# with the following definition: P = YHYMatr[...,j0]
# We also form matrix tAkl=Ak+tAl
#------------------------------------------------------------------------------------------------- 
    tAl = transpose(YHYMatr[:,:,j0])*Al*YHYMatr[:,:,j0]
    tAkl = Ak+tAl

    if verbose
        println(trace_f,"\ntAkl")
        show(trace_f, "text/plain", tAkl)
        println(trace_f," ")
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

    if verbose
        println(trace_f,"\ndet_Lk: ", det_Lk, " det_Ll", det_Ll)
    end

#---------------------------------------------
# Compute determinant of tAkl and invert tAkl 
#---------------------------------------------
    det_tAkl = det(tAkl)
    
    if det_tAkl < Mini
        return 0.0, 0.0, 0.0, 0.0
    end

    try
        inv_tAkl = inv(tAkl)
    catch e
        return 0.0, 0.0, 0.0, 0.0
    end

    if verbose
        println(trace_f,"\ninv_tAkl")
        show(trace_f, "text/plain", inv_tAkl)
        println(trace_f," ")
    end
    
#---------------------------------------------------
# Evaluating overlap 
    if overlap_Skl
        # Skl = 2^3n/2 (||Lk|| ||Ll||/|AKL|)^3/2
        temp1 = abs(det_Ll*det_Lk)/det_tAkl
        Skl = 2.0^(3.0*n/2.0)*temp1*sqrt(temp1)
    
    else
        # [Muolo] A.1.1 Overlap integral
        Skl = PI^(3.0*n/2.0)/det_tAkl*sqrt(det_tAkl)
        
    end
#----------------------------------------------------

    if verbose
        println(trace_f, "\ndet_Ll: ", det_Ll, " det_Lk: ", det_Lk, " det_tAkl: ", det_tAkl, " Skl: ", Skl)
    end

#----------------------------------------
# Doing multiplication W2 = inv_tAkl*tAl
#----------------------------------------
    W2 = inv_tAkl * tAl

    if verbose
        println(trace_f,"\nW2")
        show(trace_f, "text/plain", W2)
        println(trace_f," ")
    end

# Doing multiplication inv_tAkltAlM = inv_tAkl*tAl*M = W2*M
    for i in 1:n
        for j in 1:n
            temp1 = 0.0
            for k in 1:n
              temp1 += W2[j,k]*MassMatrix[k,i]
            end
            inv_tAkltAlM[j,i] = temp1
        end
    end

    if verbose
        println(trace_f,"\ninv_tAkltAlM")
        show(trace_f, "text/plain", inv_tAkltAlM)
        println(trace_f," ")
    end

# Computing kinetic energy, Tkl = tr[inv_tAkltAlM*Ak]
    Tkl = 0.0
    
    for i in 1:n
        temp1 = ZERO
        for k in 1:n
            temp1 = temp1+inv_tAkltAlM[i,k]*Ak[k,i]
        end
        Tkl = Tkl+temp1
    end

    if verbose
        println(trace_f, "\nSkl: ", Skl, " Tkl: ", Tkl)
    end
    
    Tkl = SIX*Skl*Tkl

    if verbose
        println(trace_f, "\nTkl = SIX*Skl*Tkl: ", Tkl)
    end

# Evaluating potential energy, Vkl, and tr[invCkl*Jij]^(-3/2)
# The lower triangle of array trinvCklJij32 will contain the corresponding quantities.
    temp1 = (TWO/SQRTPI)*Skl

    if verbose
        println(trace_f, "\ntemp1 = (TWO/SQRTPI)*Skl: ", temp1)
    end
    
    Vkl = 0.0
    
    for i in 1:n
        temp3 = inv_tAkl[i,i]
        temp4 = sqrt(temp3)
        tr_inv_tAklJij32[i,i] = 1.0/(temp4*temp3)
        temp5 = temp1/temp4
        Vkl = Vkl+PseudoCharge[i]*PseudoCharge0*temp5
    end

    if verbose
        println(trace_f, "\nVkl: ", Vkl)
    end

    for i in 1:n
        for j in i+1:n
            temp3 = inv_tAkl[i,i]+inv_tAkl[j,j]-inv_tAkl[j,i]-inv_tAkl[j,i]
            temp4 = sqrt(temp3)
            tr_inv_tAklJij32[j,i] = 1.0/(temp4*temp3)
            temp5 = temp1/temp4
            Vkl = Vkl+PseudoCharge[i]*PseudoCharge[j]*temp5
        end
    end

    Hkl = Tkl + Vkl

    if verbose
        println(trace_f, "\nMatrixElements - Hkl: ", Hkl, " Skl: ", Skl, " Tkl: ", Tkl, " Vkl: ", Vkl)
    end
    
    return Hkl,Skl,0,0
    
end

end # Module ECG_Matelem_RGL0