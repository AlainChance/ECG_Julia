#-------------------------------------------------------------------------------------------------------------------------------------
# Author: Alain Chancé
# Date: January 4, 2024
# Version: 1.0
#
# Module ECG_Matelem_RGL1
#
# This module defines MatrixElements() which computes elements of H and S matrices with
# two real L=1 correlated Gaussians:
# 
# fk = z_{m_k} exp[-r'(Lk*Lk')r] 
# 
# where m_k is some integer between 1 and n, the number of pseudoparticles) 
# Symmetry adaption is applied to the ket using permutation matrix P = YHYMatr[...,j0]
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
# [Bubin_3] Sergiy Bubin, and Ludwik Adamowicz, Energy and energy gradient matrix elements 
# with N-particle explicitly correlated complex Gaussian basis functions with L=1, 
# J. Chem. Phys. 128, 114107 (2008), https://doi.org/10.1063/1.2894866
#
# [Muolo] Andrea Muolo, Explicitly Correlated Gaussians and the Quantum Few-Body Problem, 
# DISS. ETH NO. 25680, December 2018, 
# https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/352293/1/AMuolo.pdf, 
#--------------------------------------------------------------------------------------------
module ECG_Matelem_RGL1

import ...ECG_Init: trace_f, verbose1, verbose2, verbose3, Param, set_parvec, Mini, overlap_Skl
import ...ECG_Init: ZERO, ONE, TWO, THREE, SIX, ONEHALF, ONETHIRD, ONEFOURTH, ONEFIFTH, ONESEVENTH, THREEHALF, PI, SQRTPI

using Parameters, LinearAlgebra

function MatrixElements(k0,l0,j0,grad_k,grad_l; param::Param=param,verbose1=verbose1,verbose2=verbose2)

    verbose = false
    
    if verbose
        println(trace_f, "\nMatrixElements - k0: ", k0, " l0: ", l0, " j0: ", j0, " grad_k: ", grad_k, " grad_l: ", grad_l)
    end

    @unpack n, npt, PseudoCharge0, PseudoCharge, MassMatrix, ZIndex, NonlinParam, PP, covec, YHYMatr, MatElem_method = param
    
    Ak = zeros(n,n)
    Al = zeros(n,n)
    
    tAl = zeros(n,n)
    tAkl = zeros(n,n)

    Z1 = zeros(n,n)
    
    W1 = zeros(n,n)
    W2 = zeros(n,n)

    tvl = zeros(n,n)

    inv_tAkl = zeros(n,n)
    inv_tAkltvl = zeros(n,n)
    
    inv_tAkltAl = zeros(n,n)
    inv_tAkltAlM = zeros(n,n)
    inv_tAklAk = zeros(n,n)
    inv_tAklAkM = zeros(n,n)

    vkinv_tAkl = zeros(n,n)
    vkinv_tAkltAlM = zeros(n,n)

    eta1 = zeros(n,n)
    sqrt_eta1 = zeros(n,n)
    eta2 = zeros(n,n)
    
    Rkl = zeros(n,n)
    
    Piraised3n2 = PI^(3.0*n/2.0)

    u1 = zeros(n,n)
    u2 = zeros(n,n)
    u3 = zeros(n,n)
    
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

#----------------------------------------------------------------------------
# After this we can do Cholesky factorization of tAkl.
# The Cholesky factor will be temporarily stored in the lower triangle of Z1
#----------------------------------------------------------------------------
    det_tAkl = 1.0
    
    for i in 1:n
        for j in i:n
            temp1 = tAkl[i,j]
            for k in i-1:-1:1
                temp1 = max(temp1-Z1[i,k]*Z1[j,k], Mini) # Avoid sqrt of negative number
            end
                
            if i==j
                Z1[i,i] = sqrt(temp1)
                det_tAkl = det_tAkl*temp1
            else
                Z1[j,i] = temp1/Z1[i,i]
                Z1[i,j] = 0.0
            end
        end
    end

    if verbose
        println(trace_f,"\nThe Cholesky factor will be temporarily stored in the lower triangle of Z1")
        show(trace_f, "text/plain", Z1)
        println(trace_f," ")
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

#-----------------------------------------------
# Computing tvl=P'*vl where P = YHYMatr[...,j0]
#-----------------------------------------------
    m_l = ZIndex[l0]
    
    for i in 1:n
#        tvl[i] = P[m_l,i]
        tvl[i] = YHYMatr[m_l,i,j0]
    end

    if verbose
        println(trace_f,"\ntvl")
        show(trace_f, "text/plain", tvl)
        println(trace_f," ")
    end
    
#----------------------------------------
# Computing inv_tAkltvl = inv_tAkl * tvl
#----------------------------------------
    for i in 1:n
        temp1 = 0.0
        for j in 1:n
            temp1 += inv_tAkl[j,i]*tvl[j]
        end
        inv_tAkltvl[i] = temp1
    end

    if verbose
        println(trace_f,"\ninv_tAkltvl")
        show(trace_f, "text/plain", inv_tAkltvl)
        println(trace_f," ")
    end

#-------------------------------------
# Computing vkinv_tAkl = vk'*inv_tAkl
#-------------------------------------
    m_k = ZIndex[k0]
    
    for i in 1:n
        vkinv_tAkl[i]=inv_tAkl[m_k,i]
    end

    if verbose
        println(trace_f,"\nvkinv_tAkl")
        show(trace_f, "text/plain", vkinv_tAkl)
        println(trace_f," ")
    end

#--------------------------------------------------------
# Computing tau3 = vkinv_tAkl*tvl
# [Bubin_3] C. Kinetic energy derivatives, equation (93)
#--------------------------------------------------------
    tau3 = 0.0

    for i in 1:n
        tau3 += vkinv_tAkl[i]*tvl[i]
    end

    if verbose
        println(trace_f,"\ntau3: ", tau3)
        println(trace_f," ")
    end
    
#--------------------
# Evaluating overlap 
#--------------------
    temp1 = det_tAkl*sqrt(det_tAkl)
    Skl = Piraised3n2 * tau3/(2.0*temp1)

    if verbose
        println(trace_f, " det_tAkl: ", det_tAkl, " Skl: ", Skl)
    end

#-------------------------------------------------
# Doing multiplication inv_tAkltAl = inv_tAkl*tAl
#-------------------------------------------------
    for i in 1:n
        for j in 1:n
            temp1 = 0.0
            for k in 1:n
                temp1 += inv_tAkl[j,k]*tAl[k,i]
            end
            inv_tAkltAl[j,i] = temp1
        end
    end

    if verbose
        println(trace_f,"\ninv_tAkltAl")
        show(trace_f, "text/plain", inv_tAkltAl)
        println(trace_f," ")
    end

#---------------------------------------------------
# Doing multiplication inv_tAkltAlM = inv_tAkltAl*M
#---------------------------------------------------
    for i in 1:n
        for j in 1:n
            temp1 = 0.0
            for k in 1:n
              temp1 += inv_tAkltAl[j,k]*MassMatrix[k,i]
            end
            inv_tAkltAlM[j,i] = temp1
        end
    end

    if verbose
        println(trace_f,"\ninv_tAkltAlM")
        show(trace_f, "text/plain", inv_tAkltAlM)
        println(trace_f," ")
    end

#---------------------------------------------------------
# Computing tau1 = tr[inv_tAkltAlM*Ak]
# [Bubin_3] C. Kinetic energy derivatives, equation (91)
#---------------------------------------------------------
    tau1 = 0.0

    for i in 1:n
        temp1 = 0.0
        for k in 1:n
            temp1 += inv_tAkltAlM[i,k]*Ak[k,i]
        end
        tau1 += temp1
    end

    if verbose
        println(trace_f,"\ntau1: ", tau1)
        println(trace_f," ")
    end

#------------------------------------------------------------------
# Computing tau2 = vk'*inv_tAkltAlM*Ak*inv_tAkltvl
# [Bubin_3] C. Kinetic energy derivatives, equation (92)
# We do it by multiplying twice the row-vector on the left
# by a matrix on the right and computing a dot product in the end.
# vkinv_tAkltAlM'=vk'*inv_tAkltAlM
#------------------------------------------------------------------
    for i in 1:n
        vkinv_tAkltAlM[i]=inv_tAkltAlM[m_k,i]
    end

    # u1 = vkinv_tAkltAlM'*Ak
    # tau2=u1'*inv_tAkltvl (storage for u1 as such is not needed, we use temp1=u1(i))
    
    tau2 = 0.0
    for i in 1:n
        temp1 = 0.0
        for j in 1:n
            temp1 += vkinv_tAkltAlM[j]*Ak[j,i]
        end
        tau2 += temp1*inv_tAkltvl[i]
    end

    if verbose
        println(trace_f,"\ntau2: ", tau2)
        println(trace_f," ")
    end

#--------------------------------------------------------
# Evaluating the kinetic energy
# [Bubin_3] 
# B. Kinetic energy integral, equation (37)
# C. Kinetic energy derivatives, equation (94)
#--------------------------------------------------------
    Tkl = Skl*(SIX*tau1 + 4.0*tau2/tau3)

    if verbose
        println(trace_f, "\nTkl: ", Tkl)
    end

#----------------------------------------------------------------
# Evaluating eta1[i,j], sqrt_eta1[i,j], eta2([i,j], Rkl[i,j],
# and the potential energy. Notice that only the lower triangles
# of eta1, sqrt_eta1, eta2, and Rkl are filled.
#----------------------------------------------------------------
    Vkl = 0.0
    temp1 = Skl*(TWO/SQRTPI)

    for i in 1:n
        temp2 = inv_tAkl[i,i]
        temp3 = sqrt(temp2)
        eta1[i,i] = temp2
        sqrt_eta1[i,i] = temp3

        # Getting row m_k of matrix inv_tAkl*Jii*inv_tAkl
        # as only this row is needed to compute eta2[i,i]

        for k in 1:n
            u1[k]=inv_tAkl[i,m_k]*inv_tAkl[k,i] 
        end
        temp4=ZERO

        for k in 1:n
            temp4 += u1[k]*tvl[k]
        end
        
        eta2[i,i] = temp4
        Rkl[i,i] = temp1*(ONE-temp4/(3.0*temp2*tau3))/temp3
        Vkl += PseudoCharge[i]*PseudoCharge0*Rkl[i,i]
    end

    if verbose
        println(trace_f, "\nVkl: ", Vkl)
    end

    for i in 1:n
        if verbose
            println(trace_f, "\ni: ", i)
        end
        for j in i+1:n
            temp2 = inv_tAkl[i,i] + inv_tAkl[j,j] - inv_tAkl[j,i] - inv_tAkl[j,i]
            if verbose
                println(trace_f, "\nj: ", j, " temp2: ", temp2, " inv_tAkl[i,i]: ", inv_tAkl[i,i], " inv_tAkl[j,j]: ", 
                    inv_tAkl[j,j], " inv_tAkl[j,i]: ", inv_tAkl[j,i], " inv_tAkl[j,i]: ", inv_tAkl[j,i])
            end
            
            if temp2 < 0
                temp2 = 0
            end
            
            temp3 = sqrt(temp2)
            eta1[j,i] = temp2
            sqrt_eta1[j,i] = temp3

            # Getting row m_k of matrix inv_tAkl*Jij*inv_tAkl
            # as only this row is needed to compute eta2[i,i]

            for k in 1:n
                u1[k] = (inv_tAkl[i,m_k] - inv_tAkl[j,m_k])*(inv_tAkl[k,i] - inv_tAkl[k,j])
            end
            temp4 = 0.0

            for k in 1:n
                temp4 += u1[k]*tvl[k]
            end

            eta2[j,i] = temp4
            Rkl[j,i] = temp1*(1.0 - temp4/(3.0*temp2*tau3))/temp3
            
            Vkl += PseudoCharge[i]*PseudoCharge[j]*Rkl[j,i]

            if verbose
                println(trace_f, "\nj: ", j, " eta2[j,i]: ", eta2[j,i], " Rkl[j,i]: ", Rkl[j,i], " Vkl: ", Vkl)
            end
        end 
    end

    if verbose
        println(trace_f, "\nVkl: ", Vkl)
    end

    Hkl = Tkl + Vkl

    if verbose
        println(trace_f, "\nMatrixElements - Hkl: ", Hkl, " Skl: ", Skl, " Tkl: ", Tkl, " Vkl: ", Vkl)
    end
    
    return Hkl,Skl,0,0
    
end

end # Module ECG_Matelem_RGL1