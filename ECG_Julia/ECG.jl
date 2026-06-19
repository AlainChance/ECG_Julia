#------------------------------------------------------------------------------------------------------------------------------
# Author: Alain Chancé
# Date: June 4, 2023
# Version: 1.0
#
# Module ECG
#
# This module includes all matrix-element submodules (ECG_Matelem_<method>) and routes
# MatrixElements to the active one (see select_matelem! / _ACTIVE_MATELEM). It defines the following functions:
#
# - store_HS() that stores and normalizes H and S matrix elements
# - norm_HS() that normalizes H and S matrices
# - compute_H_S() that fills Hamiltonian H and overlap S matrices
# - GSEPIIS() that solves secular equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy
# - solve_sec() that solves equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy
# - run_Fortran() that runs Fortran program main
# - stoch_nlp() that randomly selects non-linear basis set parameters
# - do_basis_enl() that enlarges the non-linear basis set parameters
# - do_opt_cycle() that performs an optimization cycle
# - do_action() that processes actions
#------------------------------------------------------------------------------------------------------------------------------
module ECG

import ..ECG_Param: verbose, GSEPSolutionMethod, GSEPIIS_Max_iter, compute_H_S_method, MatElem_method, do_Fortran, outfile, _EPS
import ..ECG_Param: inout_F90_file, max_print_H, do_GSEPIIS, param0, NonlinParam_file, NonlinParam_file_F90, Mini
import ..ECG_Param: NonlinParam_db_file, all_real, overlap_Skl

import ..ECG_Init: trace_f, basis, N_Pvec, N_matr, N_C_Pvec, RGL0, RGL1, CGL0, verbose1, verbose2
import ..ECG_Init: verbose3, seed, Param, param, read_NonlinParam, write_NonlinParam, basis_repl, basis_enl, basis_enl_F90
import ..ECG_Init: opt_cycle, opt_cycle_F90, full_opt1, check, action, action_list, history, history_size, history_add, history_update
import ..ECG_Init: history_print, history_plot, read_inout, write_inout

#---------------------------------------------------------------------------------------------------------
# Symmetry-term truncation state (multi-fidelity optimization, June 2026)
#
# When _TRUNCATE[] is true, compute_H_S() weights the Y^{+}Y sum with _YHYCOEFF_TRUNC[] (small-coefficient
# terms zeroed and skipped) instead of the full YHYCoeff, giving a cheaper, partially symmetrized energy
# surface. optim_nlp() turns this on only around the inner BFGS search and off for every accepted/reported
# energy, so the optimizer explores cheaply while all kept energies use the full operator. Controlled by
# ECG_Param.param0.coeff_threshold (0.0 = disabled). See ECG_Param.jl.
#---------------------------------------------------------------------------------------------------------
const _TRUNCATE = Ref(false)
const _YHYCOEFF_TRUNC = Ref{Vector{Int64}}(Int64[])

# Activate a truncation threshold: zero the symmetry-term coefficients with |coeff| < threshold
# (threshold <= 0 means full operator). Returns the number of terms kept. Used by optim_nlp's
# threshold ramp; mutates the module-global _YHYCOEFF_TRUNC.
function _set_trunc!(threshold, full_coeffs)
    if threshold > 0
        _YHYCOEFF_TRUNC[] = [abs(c) < threshold ? zero(c) : c for c in full_coeffs]
        return count(!=(0), _YHYCOEFF_TRUNC[])
    else
        _YHYCOEFF_TRUNC[] = Int64[]
        return length(full_coeffs)
    end
end

#----------------------------------------------------------------------------------
# Matrix-element variants (Phase 2): load ALL of them as unique submodules
# ECG_Matelem_<method>, then select the active one at run time from the model flags.
# Functions throughout ECG call the forwarding MatrixElements defined below, which
# routes to the active submodule's MatrixElements.
#----------------------------------------------------------------------------------
include(joinpath(@__DIR__, "ECG_Matelem_RGL0.jl"))
include(joinpath(@__DIR__, "ECG_Matelem_CGL0.jl"))
include(joinpath(@__DIR__, "ECG_Matelem_RGL1.jl"))
include(joinpath(@__DIR__, "ECG_Matelem_N_Pvec.jl"))
include(joinpath(@__DIR__, "ECG_Matelem_N_matr.jl"))
include(joinpath(@__DIR__, "ECG_Matelem_N_C_Pvec.jl"))

# Active matrix-element submodule, chosen from the model flags (set in ECG_Param).
const _ACTIVE_MATELEM = Ref{Module}(ECG_Matelem_CGL0)

function select_matelem!()
    _ACTIVE_MATELEM[] =
        CGL0     ? ECG_Matelem_CGL0     :
        N_Pvec   ? ECG_Matelem_N_Pvec   :
        N_matr   ? ECG_Matelem_N_matr   :
        N_C_Pvec ? ECG_Matelem_N_C_Pvec :
        RGL0     ? ECG_Matelem_RGL0     :
        RGL1     ? ECG_Matelem_RGL1     :
        error("Unknown MatElem_method: $MatElem_method")
    println("Using matrix-element module ", nameof(_ACTIVE_MATELEM[]))
    println(trace_f, "Using matrix-element module ", nameof(_ACTIVE_MATELEM[]))
    return _ACTIVE_MATELEM[]
end
# select_matelem!() is now called at run time from init!() (end of module), since
# it reads the model flags, which ECG_Init.init!() only sets at run time.

# Forward MatrixElements (called throughout ECG) to the active submodule.
MatrixElements(args...; kwargs...) = _ACTIVE_MATELEM[].MatrixElements(args...; kwargs...)

#----------------------------------------------------------------------------------------------------
# Include module ECG_Fortran (Fortran-interface functions split out of ECG.jl) and import its functions.
# set_ecg_module! hands ECG_Fortran a reference back to this module for the few numerical helpers it calls
# (read_matrix, write_matrix, symm, solve_eigen, compute_and_solve), avoiding a circular module dependency.
#----------------------------------------------------------------------------------------------------
include("ECG_Fortran.jl")
import .ECG_Fortran: read_complex_matrix, compute_and_solve_F90, solve_F90, run_Fortran
ECG_Fortran.set_ecg_module!(@__MODULE__)

#----------------------------------------------------
# Export functions compute_H_S, solve_sec, do_action
#----------------------------------------------------
export compute_H_S, solve_sec, do_action


#---------------------------------------------------
# Create a new instance of the Param data structure
#---------------------------------------------------
# param1 is constructed at run time in init!() (its element type depends on the
# model flags, which are only set at run time). Placeholder so the binding exists.
param1 = nothing

#------------------------------------------------------------------
# Use Parameters, DelimitedFiles, Random, Printf
# Using Printf - # https://docs.julialang.org/en/v1/stdlib/Printf/
#------------------------------------------------------------------
using Parameters, DelimitedFiles, Random, Printf

#-------------------------------
# Define vector action_list_F90 
#-------------------------------
action_list_F90::Vector{action} = []

#-----------------------------------------------------------
# Define vech() vector half vectorization
# Julia half Vectorization
# https://discourse.julialang.org/t/half-vectorization/7399
#-----------------------------------------------------------
using LinearAlgebra
function vech(A::Union{Matrix{Float64}, Matrix{ComplexF64}})
    m::Int64 = LinearAlgebra.checksquare(A)
    l::Int64 = floor(m*(m+1)/2)

    v = zeros(typeof(A[1,1]), l)
    
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k += 1] = A[i,j]
    end
    
    return v
end

#-----------------------------------------------------------------------------------------------------
# Define dist_1() that computes the following distance: dist_1(x, y) = abs(y - x)/max(abs(y), abs(x))
#-----------------------------------------------------------------------------------------------------
function dist_1(x, y)
    return abs(y - x)/max(abs(x), abs(y))
end

#-------------------------------------------------------------------------------------------------------------
# Define dist_diagH() that computes and sorts distances between diagonal elements of the Hamiltonian matrix H
#-------------------------------------------------------------------------------------------------------------
function dist_diagH(; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    dist_diagH_threshold = param0.dist_diagH_threshold
    
    H = param.H
    l = size(H,2)
    m::Int64 = floor(l*(l-1)/2)

    type_H = typeof(H[1,1])

    dist_H = zeros(Float64, m)
    map = Dict{Int64,Tuple{Int64,Int64}}()

    k = 1
    for i in 1:l
        for j in i+1:l
            map[k] = (i,j)
            dist_H[k] = dist_1(H[i,i], H[j,j])
            k += 1
        end
    end

    # Return a permutation vector that puts dist_H in sorted order
    # https://docs.julialang.org/en/v1/base/sort/
    perm = sortperm(dist_H)
    
    x = [map[perm[i]] for i in 1:m if dist_H[perm[i]] < dist_diagH_threshold]

    if verbose2
        # Print sorted distances between diagonal elements of Hamiltonian matrix H 
        println("\nSorted distances between diagonal elements of Hamiltonian matrix H")
        println(trace_f, "\nSorted distances between diagonal elements of Hamiltonian matrix H")
            
        for i in 1:min(size(perm,2), max_print_H)
            s1 = string(map[perm[i]])
            s2 = string(dist_H[perm[i]])
            println("$s1 $s2")
            println(trace_f, "$s1 $s2")
        end

        # Print diagonal elements of Hamiltonian matrix H that are distant below the minimum distance threshold threshold
        if x != Tuple{Int64, Int64}[]
            println("\nThe following diagonal elements of Hamiltonian matrix H are distant below the minimum distance threshold")
            println(trace_f, "\nThe following diagonal elements of Hamiltonian matrix H are distant below the minimum distance threshold")
            println(x)
            println(trace_f, x)
        end
    end

    return perm, map, dist_H, x
end

#---------------------------------------------------------------------------------------------------------------------
# Define dist_nlp() that computes ans sorts distances between functions as vectors of non-linear basis set parameters
# Ref. Distances.jl, A Julia package for evaluating distances (metrics) between vectors.
# https://github.com/JuliaStats/Distances.jl
#----------------------------------------------------------------------------------------------------------------------
using Distances: euclidean
function dist_functions(; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    dist_func_threshold = param0.dist_func_threshold

    NonlinParam = param.NonlinParam
    l = size(NonlinParam,1)
    
    m::Int64 = floor(l*(l-1)/2)
    
    dist_func = zeros(Float64, m)
    map = Dict{Int64,Tuple{Int64,Int64}}()

    k = 1
    for i in 1:l
        for j in i+1:l
            map[k] = (i,j)
            dist_func[k] = euclidean(NonlinParam[j,:], NonlinParam[i,:])
            k += 1
        end
    end

    # Return a permutation vector that puts dist_func in sorted order
    # https://docs.julialang.org/en/v1/base/sort/
    perm = sortperm(dist_func)

    x = [map[perm[i]] for i in 1:m if dist_func[perm[i]] < dist_func_threshold]

    if verbose2
        # Print sorted distances between functions
        println("\nSorted distances between functions")
        println(trace_f, "\nSorted distances between functions")
            
        for i in 1:min(size(perm,2), max_print_H)
            s1 = string(map[perm[i]])
            s2 = string(dist_func[perm[i]])
            println("$s1 $s2")
            println(trace_f, "$s1 $s2")
        end

        # Print functions that are distant below the minimum distance threshold
        if x != Tuple{Int64, Int64}[]
            println("\nThe following functions are distant below the minimum distance threshold")
            println(trace_f, "\nThe following functions are distant below the minimum distance threshold")
            println(x)
            println(trace_f, x)
        end
    end

    return perm, map, dist_func, x
end

#------------------------------------------------------------------------------------------------------
# Define distance_ok() that determines whether or not two diagonal elements of the Hamiltonian matrix 
# pertaining to any new function and any accepted function is greater than the minimum distance threshold
# 
# Output:
# - false: the distance is lower than the minimum distance threshold
# - true: the distance is greater than the minimum distance threshold
#---------------------------------------------------------------------------------------------------------
function distance_ok(Kstart, Kstop; param::Param=param, verbose1=verbose1, verbose2=verbose2)
    dist_diagH_threshold = param0.dist_diagH_threshold
    H = param.H

    if dist_diagH_threshold == 0.0
        return true
    end
    
    for i in 1:Kstart-1
        for j in Kstart:Kstop
            dist_diagH = dist_1(H[j,j], H[i,i])
            if dist_diagH < dist_diagH_threshold
                if verbose2
                    s = @sprintf("%.2e", dist_diagH)
                    println("\nThe distance ", s, " between H[", j, ",", j, "] and H[", i, ",", i, "] is below minimum threshold ", DistanceThreshold)
                    println(trace_f, "\nThe distance ", s, " between H[", j, ",", j, "] and H[", i, ",", i, "] is below minimum threshold ", DistanceThreshold)
                end
                return false
            end
        end
    end

    return true
end

#---------------------------------------------------------------------------------------------------------------
# Define max_overlap() that computes the maximum magnitude of the elements of the overlap matrix S that are not 
# on the diagonal, a measure of the overlap between ECGs
# Ref. Jim Mitroy et al., Theory and application of explicitly correlated Gaussians, F. Linear dependence issues
# https://www.researchgate.net/publication/258098634_Theory_and_application_of_explicitly_correlated_Gaussians
#---------------------------------------------------------------------------------------------------------------
function max_overlap(; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    overlap_threshold = param0.overlap_threshold
    
    S = param.S
    l = size(S,2)

    max_S = abs(S[2,1])
    ii = 1
    jj = 2
    
    for i in 1:l
        for j in i+1:l
            overlap = abs(S[j,i])

            if overlap > max_S
                max_S = overlap
                ii = i
                jj = j
            end
        end
    end

    if verbose1
        s = @sprintf("%.2e", max_S)
        println("\nS[", jj, ",", ii, "] is maximum overlap ", s)
        println(trace_f, "\nS[", jj, ",", ii, "] is maximum overlap ", s)
    end
    
    if max_S <= overlap_threshold
        ok = true
    else
        ok = false
    end

    return ok, ii, jj, max_S
end

#----------------------------------------------------------------------------------
# Define write_matrix() which writes a file using DelimitedFiles.writedlm function 
# https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/
#----------------------------------------------------------------------------------
function write_matrix(M; matrix_name="H", verbose1=verbose1)

    ok = true

    file = string(matrix_name, ".txt")
    
    try
        writedlm(file, M) # Write matrix M to file
        if verbose1
            println("\nMatrix $matrix_name written into file $file")
            println(trace_f, "\nMatrix $matrix_name written into file $file")
        end
    catch e
        ok = false
        if verbose1
            println("\nFailed to write matrix $matrix_name into file $file")
            println(trace_f, "\nFailed to write matrix $matrix_name into file $file")
        end
    end

    return ok
end

#---------------------------------------------------------------------------------------------
# Define read_matrix() which reads a matrix from a file using DelimitedFiles.readdlm function 
# https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/
#---------------------------------------------------------------------------------------------
function read_matrix(; matrix_name="H", verbose1=verbose1)

    ok = true
    file = string(matrix_name, ".txt")
    
    if isfile(file)
        try    
            M = readdlm(file) # Read M from H_file
            if verbose1
                println("\nRead matrix ", matrix_name, " from file ", file)
                println(trace_f, "\nRead matrix ", matrix_name, " from file ", file)
            end
            return true, M
        catch e
            if verbose1
                println("\nFailed to read matrix ", matrix_name, " from file ", file)
                println(trace_f, "\nFailed to read matrix ", matrix_name, " from file ", file)
            end
            return false, nothing
        end
    else
        if verbose1
            println("\nread_matrix - file: ", file, " not found")
            println(trace_f, "\nread_matrix - file: ", file, " not found")
        end
        return false, nothing
    end
end


#---------------------------------------------------------------------------------------
# Define write_matrix_real() which reads a matrix of complex numbers from a file 
# using DelimitedFiles.readdlm function and writes a matrix of real numbers into a file
#----------------------------------------------------------------------------------------
function write_matrix_real(; matrix_name="Evectors", suffix="_real", verbose1=verbose1)

    file = string(matrix_name, ".txt")

    if isfile(file)
    
        U = readdlm(file, '\t', ComplexF64) # Read matrix of vectors from file
        if verbose1
            println("\nReading matrix ", matrix_name, " from file ", file)
            println(trace_f, "\nReading matrix ", matrix_name, " from file ", file)
        end

        l::Int64 = size(U)[1]
        M::Array{Float64} = zeros(Float64, l, 2*l)

        for i in 1:l
            for j in 1:l
                M[i,2*j-1] = real(U[i,j])
                M[i,2*j] = imag(U[i,j])
            end
        end

        file_w = string(matrix_name, suffix, ".txt") 

        writedlm(file_w, M) # Write matrix M to file
        if verbose1
            println("\nWriting matrix ", matrix_name, " into file ", file_w)
            println(trace_f, "\nWriting matrix ", matrix_name, " into file ", file_w)
        end

    else
        if verbose1
            println("\nfile: ", file, " not found")
            println(trace_f, "\nfile: ", file, " not found")
        end
    end
end

#--------------------------------------------------------------------------------------------------------------------
# Define diff_H_90_H which prints a list of elements of (H_90 - H) that have a magnitude greater than a given number
#--------------------------------------------------------------------------------------------------------------------
function diff_H_90_H(; param::Param=param, k=1e2)

    cbs = param.cbs
    H = param.H
    typeof_H = typeof(H[1,1])

    if typeof_H == ComplexF64
        ok, H_90 = read_complex_matrix(; matrix_name="H_90", verbose1=false)
    else
        ok, H_90 = read_matrix(; matrix_name="H_90", verbose1=false)
    end

    if ok
        V = H_90 - H
        if maximum(abs.(V)) > k 
            println(trace_f, "\nList of elements of H_90 and H matrices that differ by a magnitude greater than: ", k)
            for i in 1:cbs
                for j in 1:i
                    if abs(V[i,j]) > k
                        println(trace_f, "\nH_90[", i, ",", j, "] = ", H_90[i,j])
                        println(trace_f, "H[", i, ",", j, "] = ", H[i,j])
                    end
                end
            end
        end
        return true, H_90
    else
        return false, nothing
    end
end

#------------------------------------------------------------------------------------------------------------------------------
# Define store_HS() that stores and normalizes matrix elements of the Hamiltonian, the overlap matrices and their derivatives
# k must be greater than or equal to l.
#------------------------------------------------------------------------------------------------------------------------------
function store_HS(k, l, Hkl, Skl, Dk, Dl, grad_k, grad_l; param::Param=param)
    
    if CGL0 || N_C_Pvec
        return store_HS_complex(k, l, Hkl, Skl, Dk, Dl, grad_k, grad_l; param=param)
    else
        return store_HS_real(k, l, Hkl, Skl, Dk, Dl, grad_k, grad_l; param=param)
    end
    
end

#------------------------------------------------------------------------------------------------------------
# Define store_HS_real() that stores and normalizes matrix elements of the Hamiltonian, the overlap matrices 
# and their derivatives for real correlated Gaussians. k must be greater than or equal to l.
#------------------------------------------------------------------------------------------------------------
function store_HS_real(k, l, Hkl, Skl, Dk, Dl, grad_k, grad_l; param::Param=param)
    
    @unpack trace_f, GSEP_G, GSEP_I, npt, nfru, H, diagH, S, diagS, D, ApproxEnergy = param
    
    npt2 = 2*npt
    
    if GSEP_G
        # In the case when GSEPSolutionMethod='G', the diagonal of the Hamiltonian matrix is stored in diagH
        # and the diagonal of the overlap is stored in diagS. 
        # The lower triangles of arrays H and S are used to store H and S
        if k==l
            diagS[k] = Skl
            S[k,k] = 1.0
            if abs(diagS[k]) > Mini
                diagH[k] = Hkl/diagS[k]
                H[k,k] = diagH[k]
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,k] = 2.0*Dk[1:npt2]/diagS[k]
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_real - k: ", k, " l: ", " diagS[k] ", diagS[k], " <= ", Mini)
                end
                
                return false
            end
        else
            if abs(diagS[k]*diagS[l]) > Mini
                f=1.0/sqrt(abs(diagS[k]*diagS[l]))
                
                S[k,l] = Skl*f
                S[l,k] = S[k,l]
                H[k,l] = Hkl*f
                H[l,k] = H[k,l]
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,l] = Dk[1:npt2]*f
                end
                
                if grad_l && l>nfru
                    D[1:npt2,l-nfru,k] = Dl[1:npt2]*f
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_real - k: ", k, " l: ", l, " diagS[k]*diagS[l] ", diagS[k]*diagS[l], " <= ", Mini)
                end
                
                return false
            end
        end
    end
    
    if GSEP_I
        # Function GSEPIIS solves the secular equation using the inverse iteration method.
        # Only the lower triangle of array H (including the diagonal) is used to store H-ApproxEnergy*S.
        # The entire array S is used to store S.
        if k==l
            diagS[k] = Skl
            S[k,k] = 1.0
            if abs(diagS[k]) > Mini
                
                H[k,k] = Hkl/diagS[k] - ApproxEnergy
                diagH[k] = H[k,k]
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,k] = 2.0*Dk[1:npt2]/diagS[k]
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_real - k: ", k, " l: ", l, " diagS[k] ", diagS[k], " <= ", Mini)
                end
                
                return false
            end
        else
            if abs(diagS[k]*diagS[l]) > Mini
                f=1.0/sqrt(abs(diagS[k]*diagS[l]))
                
                S[k,l] = Skl*f
                S[l,k] = S[k,l]
                H[k,l] = (Hkl - ApproxEnergy*Skl)*f
                H[l,k] = H[k,l]
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,l] = Dk[1:npt2]*f
                end
                
                if grad_l && l>nfru
                    D[1:npt2,l-nfru,k] = Dl[1:npt2]*f
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_real - k: ", k, " l : ", l, " diagS[k]*diagS[l] ", diagS[k]*diagS[l], " <= ", Mini)
                end
                
                return false
            end
        end
    end
    
    return true
end

#------------------------------------------------------------------------------------------------------------
# Define store_HS_complex() that stores and normalizes matrix elements of the Hamiltonian, the overlap matrices 
# and their derivatives for complex correlated Gaussians. k must be greater than or equal to l.
#------------------------------------------------------------------------------------------------------------
function store_HS_complex(k, l, Hkl, Skl, Dk, Dl, grad_k, grad_l; param::Param=param)
    
    @unpack trace_f, GSEP_G, GSEP_I, npt, nfru, H, diagH, S, diagS, D, ApproxEnergy = param
    
    npt2 = 2*npt
    
    if GSEP_G
        # In the case when GSEPSolutionMethod='G', the diagonal of the Hamiltonian matrix is stored in diagH
        # and the diagonal of the overlap is stored in diagS. 
        # The lower triangles of arrays H and S are used to store H and S
        if k==l
            diagS[k] = real(Skl)
            S[k,k] = complex(1.0, 0.0)
            if abs(diagS[k]) > Mini
                
                diagH[k] = real(Hkl)/diagS[k]
                H[k,k] = complex(diagH[k], 0.0)
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,k] = 2.0*real(Dk[1:npt2]/diagS[k])
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_complex - k: ", k, " l: ", " diagS[k] ", diagS[k], " <= ", Mini)
                end
                
                return false
            end
        else
            if abs(diagS[k]*diagS[l]) > Mini
                f=1.0/sqrt(abs(diagS[k]*diagS[l]))
                S[k,l] = Skl*f
                S[l,k] = conj(S[k,l])
                H[k,l] = Hkl*f
                H[l,k] = conj(H[k,l])
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,l] = conj(Dk[1:npt2]*f)
                    end
                
                if grad_l && l>nfru
                    D[1:npt2,l-nfru,k] = conj(Dl[1:npt2]*f)
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_complex - k: ", k, " l: ", l, " diagS[k]*diagS[l] ", diagS[k]*diagS[l], " <= ", Mini)
                end
                
                return false
            end
        end
    end
    
    if GSEP_I
        # Function GSEPIIS solves the secular equation using the inverse iteration method.
        # Only the lower triangle of array H (including the diagonal) is used to store H-ApproxEnergy*S.
        # The entire array S is used to store S.
        if k==l
            diagS[k] = real(Skl)
            S[k,k] = complex(1.0, 0.0)
            if abs(diagS[k]) > Mini
                
                H[k,k] = real(Hkl)/diagS[k] - ApproxEnergy
                diagH[k] = H[k,k]
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,k] = 2.0*real(Dk[1:npt2]/diagS[k])
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_complex - k: ", k, " l: ", l, " diagS[k] ", diagS[k], " <= ", Mini)
                end
                
                return false
            end
        else
            if abs(diagS[k]*diagS[l]) > Mini
                f=1.0/sqrt(abs(diagS[k]*diagS[l]))
                S[k,l] = Skl*f
                S[l,k] = conj(S[k,l])
                H[k,l] = (Hkl - ApproxEnergy*Skl)*f
                H[l,k] = conj(H[k,l])
                
                if grad_k && k>nfru
                    D[1:npt2,k-nfru,l] = conj(Dk[1:npt2]*f)
                end
                if grad_l && l>nfru
                    D[1:npt2,l-nfru,k] = conj(Dl[1:npt2]*f)
                end
            else
                param.div_by_zero = true
                param.n_div_by_zero += 1
                
                if verbose1
                    println(trace_f, "\nstore_HS_complex - k: ", k, " l : ", l, " diagS[k]*diagS[l] ", diagS[k]*diagS[l], " <= ", Mini)
                end
                
                return false
            end
        end
    end
    
    return true
end

#------------------------------------------------------------------------------------------------------------------------------
# Define norm_HS() that normalizes H and S matrices
#------------------------------------------------------------------------------------------------------------------------------
function norm_HS(; param::Param=param)
    ok = true
    
    @unpack trace_f, Nmin, Nmax, H, S = param
    
    for k in Nmin:Nmax
        for l in k:-1:1
            # Store and normalize matrix elements H[k,l] and S[k,l]
            ok = store_HS(k,l,H[k,l],S[k,l],0.0,0.0,false,false) 
        end
    end
    
    return ok
end

#---------------------------------------------------------------------
# Define print_HS() that prints Hamiltonian H and overlap S matrices
#---------------------------------------------------------------------
function print_HS(H, S)
    
    println("\nHamiltonian matrix H")
    show(stdout, "text/plain", H)
    println(" ")
    
    println(trace_f,"\nHamiltonian matrix H")
    show(trace_f, "text/plain", H)
    println(trace_f, " ")
        
    println("\nOverlap matrix S")
    show(stdout, "text/plain", S)
    println(" ")
                
    println(trace_f, "\nOverlap matrix S")
    show(trace_f, "text/plain", S)
    println(trace_f, " ")

    return
end

#----------------------------------------------------------------------------------------------------------
# Define condition_eigenvalues() that computes the condition defined as the ratio between the eigenvalue
# with largest magnitude over the one with the smallest magnitude.
#
# Ref. Andrea Muolo, Explicitly Correlated Gaussians and the Quantum Few-Body Problem, DISS. ETH NO. 25680, 
# December 2018, https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/352293/1/AMuolo.pdf
# 2.4 The stochastic variational method, 3.8 Numerical stability of complex functions
#-----------------------------------------------------------------------------------------------------------
function condition_eigenvalues(Evalues)
    
    Max_eigen = maximum(abs.(Evalues))
    Min_eigen = minimum(abs.(Evalues))
    condition = Max_eigen/Min_eigen

    return condition
end

#---------------------------------------------------------------------------------------------------------------------------------------
# Define print_eigenvalues() that prints eigenvalues, energy of the ground state and energy of first excited states
#
# According to the mini−max theorem, if the energy values are set in an increasing order, the first one provides an upper bound 
# to the exact nonrelativistic ground state energy of the system and the kth one provides an upper bound to the exact energy 
# of the (k − 1)th excited state (details of the proof can be found in ref 5).
# 
# Sergiy Bubin et al. Born−Oppenheimer and Non-Born−Oppenheimer, Atomic and Molecular Calculations with Explicitly Correlated Gaussians
# 2.5. The Variational Method, https://pubs.acs.org/doi/abs/10.1021/cr200419d
#
# (5) Suzuki, Y.; Varga, K. Stochastic Variational Approach to Quantum-Mechanical Few-Body Problems; Lecture Notes in Physics; Springer:
# Berlin, 1998.
#----------------------------------------------------------------------------------------------------------------------------------------
function print_eigenvalues(Evalues, ApproxEnergy; verbose1=verbose1, GSEP_G=false, GSEP_I=true)

    #---------------------------------------------
    # Filter eigenvalues that satisfy:
    # if GSEP_I, real(Evalues) + ApproxEnergy < 0
    # else satisfy real(Evalues) < 0
    #---------------------------------------------
    # https://docs.julialang.org/en/v1/base/collections/#Base.filter
    if GSEP_I
        X = filter(x->x<-ApproxEnergy, real(Evalues))
    else
        X = filter(x->x<0, real(Evalues))
    end

    len = min(length(X), max_print_H)
    V = X[1:len]

    #----------------------------------------------------------------------------------
    # Compute the condition as the ratio between the eigenvalue with largest magnitude 
    # over the one with the smallest magnitude
    #-----------------------------------------------------------------------------------
    condition = condition_eigenvalues(Evalues)
    
    #------------------------------------------------------------------
    # Print eigen values that satisfy real(Evalues) + ApproxEnergy < 0
    #------------------------------------------------------------------
    if verbose1 && length(X) > 0
        #---------------------------------------------
        # Print eigenvalues that satisfy:
        # if GSEP_I, real(Evalues) + ApproxEnergy < 0
        # else satisfy real(Evalues) < 0
        #---------------------------------------------
        if GSEP_I
            println("\nEigenvalues that satisfy real(Evalues) + ApproxEnergy: ", ApproxEnergy, " < 0")
            println(trace_f, "\nEigenvalues that satisfy real(Evalues) + ApproxEnergy: ", ApproxEnergy, " < 0")
        else #GSEP_G
            println("\nEigenvalues that satisfy real(Evalues) < 0")
            println(trace_f, "\nEigenvalues that satisfy real(Evalues) < 0")
        end
        
        show(stdout, "text/plain", V)
        show(trace_f, "text/plain", V)
        
        println(" ")
        println(trace_f, " ")
    
        #-------------------------------------------------------------------------
        # Print energy of the ground state and energy of the first excited states
        #-------------------------------------------------------------------------
        println("\nEnergy of the ground state and first excited states")
        println(trace_f, "\nEnergy of the ground state and first excited states")
        
        if GSEP_I
            show(stdout, "text/plain", ApproxEnergy.+V)
            show(trace_f, "text/plain", ApproxEnergy.+V)
        else #GSEP_G
            show(stdout, "text/plain", V)
            show(trace_f, "text/plain", V)
        end
        
        println(" ")
        println(trace_f, " ")
    end

    return
end

#-----------------------------------------------------------------------------------------------------
# Define sym() that symetrizes matrix H by completing the upper triangle of H with the lower triangle
# if real or with adjoint of lower triangle if CGL0 or N_C_Pvec is used
#-----------------------------------------------------------------------------------------------------
function symm(H)

    #--------------------------------------------------------------------------------------------------
    # For CGL0 and  N_C_Pvec methods, complete upper triangle of H with adjoint of lower triangle
    #--------------------------------------------------------------------------------------------------
    if CGL0 || N_C_Pvec
        for i in 1:size(H,2)
            for j in 1:i
                H[j,i] = conj(H[i,j])
            end
        end
    #---------------------------------
    # For other methods, symmetrize H
    #---------------------------------
    else                              
        for i in 1:size(H,2)
            for j in 1:i
                H[j,i] = H[i,j]
            end
        end
    end

    return H
end

#-------------------------------------------------------------------------
# Define compute_H_S() that computes Hamiltonian H and overlap S matrices
#-------------------------------------------------------------------------
# Using Printf - https://docs.julialang.org/en/v1/stdlib/Printf/
using Printf

function compute_H_S(; param::Param=param, verbose1=verbose1, verbose2=verbose2, print=true)

    @unpack trace_f, n, npt, cbs, nfru, Nmin, Nmax, NonlinParam, YHYMatr, NumYHYTerms, YHYCoeff, CurrEnergy, InvitParameter, 
    ApproxEnergy, H, S, ZIndex_used, grad_k, grad_l = param
    
#--------------
# Sanity check 
#--------------
    if !ZIndex_used && (N_Pvec || N_matr || RGL1)
        println("compute_H_S - ZIndex is needed for MatElem_method: ", MatElem_method)
        return false
    end

    EigvalTol = param.EigvalTol
    
    if verbose1
        println("\ncompute H_S method: $compute_H_S_method,  Matrix Elements method: $MatElem_method")
        println("Current basis size: ", cbs)
        println("Current energy: $CurrEnergy InvitParameter: $InvitParameter Approximate energy: $ApproxEnergy Eigen value tolerance: $EigvalTol")
        println("grad_k: ", grad_k)
        
        println(trace_f, "\ncompute H_S method: $compute_H_S_method  Matrix Elements method: $MatElem_method")
        println(trace_f, "Current basis size: ", cbs)
        println(trace_f, "Current energy: $CurrEnergy InvitParameter: $InvitParameter Approximate energy: $ApproxEnergy Eigen value tolerance: $EigvalTol")
        println(trace_f, "grad_k: ", grad_k)
        
        if grad_k
            println("Number of functions remaining unchanged - nfru: ", nfru)
            println(trace_f, "Number of functions remaining unchanged - nfru: ", nfru)
        end  
    end

#---------------------------------------------------------------------------------------
# Compute matrix elements for each basis, each line and each column of H and S matrices
#---------------------------------------------------------------------------------------
    npt2 = 2*npt
    grad_l = false             # Initialize grad_l to false
    Dlsum = 0.0
    p = ceil(Int64, cbs/10)    # Period of print statements

    # Active symmetry-term coefficients: the truncated set during the inner BFGS search of optim_nlp
    # (when _TRUNCATE[] is set), the full operator otherwise. The length guard falls back to the full
    # vector whenever the truncated one is not built for this NumYHYTerms (so this is a no-op by default).
    coeffs = (_TRUNCATE[] && length(_YHYCOEFF_TRUNC[]) == NumYHYTerms) ? _YHYCOEFF_TRUNC[] : vec(YHYCoeff)

    start = time()
    
    if basis
        if verbose1
            if CGL0 || N_C_Pvec
                println("\nk   l                      HSum                                                    Ssum                               Elapsed(s)")
                println(trace_f, "\nk   l                      HSum                                                    Ssum                               Elapsed(s)")
            else
                println("\nk   l          HSum                      Ssum                     Elapsed(s)")
                println(trace_f, "\nk   l          HSum                      Ssum                     Elapsed(s)")
            end
        end
        
        for k in Nmin:Nmax
            for l in k:-1:1
        
                Hsum = 0.0
                Ssum = 0.0
                
                # If grad_k then compute derivatives of normalized Hkl and Skl with respect to vechLk, dHkl/dvechLk and dSkl/dvechLk 
                if grad_k
                    if CGL0 || N_C_Pvec
                        Dksum = zeros(ComplexF64,npt2)
                    else
                        Dksum = zeros(Float64,npt2)
                    end
                    if l>nfru && l!=k
                        # grad_l true: compute derivatives of normalized Hkl and Skl with respect to vechLl, dHkl/dvechLl and dSkl/dvechLl
                        grad_l = true
                        if CGL0 || N_C_Pvec
                            Dlsum = zeros(ComplexF64,npt2)
                        else
                            Dlsum = zeros(Float64,npt2)
                        end
                    else
                        grad_l = false
                        Dlsum = 0.0
                    end
                else
                    Dksum = 0.0
                end
            
                for j in 1:NumYHYTerms

                    coeffs[j] == 0 && continue   # skip truncated (zero-weight) symmetry terms

                    Hkl,Skl,Dk,Dl = MatrixElements(k, l, j, grad_k, grad_l; param=param, verbose1=verbose1, verbose2=verbose2)

                    Hsum += coeffs[j]*Hkl
                    Ssum += coeffs[j]*Skl

                    # if grad_k then Dk=(dHkl/dvechLk,dSkl/dvechLk)
                    if grad_k
                        Dksum += coeffs[j]*Dk
                    end

                    # if grad_l then Dl=(dHkl/dvechLl,dSkl/dvechLl)
                    if grad_l && l>nfru && l!=k
                        Dlsum += coeffs[j]*Dl
                    end

                end
        
                if  l%p == 0 && verbose1
                    elapsed = time()-start
                    s = @sprintf("%.2f", elapsed)
                        
                    println("$k   $l    $Hsum        $Ssum             $s")
                    println(trace_f, "$k   $l    $Hsum        $Ssum             $s")
                end
            
                ok = store_HS(k,l,Hsum,Ssum,Dksum,Dlsum,grad_k,grad_l; param=param) # Store and normalize matrix elements Hsum, Ssum, Dksum, Dlsum
            end
        end
        
        if verbose1
            elapsed = time()-start
            s = @sprintf("%.2f", elapsed)
            println("\nElapsed(s): ", s)
            println(trace_f, "\nElapsed(s): ", s)
        end
    
    else # compute_H_S_method == "symmetry terms"
        start = time()
    
        if verbose1
            println("\nj     Elapsed(s)")
            println(trace_f, "\nj     Elapsed(s)")
        end
    
        for j in 1:NumYHYTerms
            coeffs[j] == 0 && continue   # skip truncated (zero-weight) symmetry terms
            for k in Nmin:Nmax
                for l in k:-1:1

                    Hkl,Skl,Dk,Dl = MatrixElements(k, l, j, false, false; param=param, verbose1=verbose1, verbose2=verbose2)

                    H[k,l] += coeffs[j]*Hkl
                    S[k,l] += coeffs[j]*Skl
                end
            end
                    
            if j%100 == 0 && verbose1
                elapsed = time()-start
                s = @sprintf("%.2f", elapsed)
                    
                println(j, "   ", s)
                println(trace_f, j, "   ", s)
            end
        end
        
        elapsed = time()-start
        s = @sprintf("%.2f", elapsed)
        
        if verbose1
            println("\nElapsed(s): ", s)
            println(trace_f, "\nElapsed(s): ", s)
        end

        # Normalize H and S matrices
        ok = norm_HS(; param=param)
    
    end
    
    #----------------------------------------------------------------
    # Print H and S matrices if print is true and cbs <= max_print_H 
    #----------------------------------------------------------------
    if print && param.cbs <= max_print_H
        print_HS(param.H, param.S)
    end
        
    return true
end

#------------------------------------------------------------------------------------------------------------------------------
# Define GSEPIIS() that solves secular equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy.
# We use Julia Linear Algebra bunchkaufman function to compute the Bunch-Kaufman factorization of a symmetric matrix to solve 
# secular equation 𝐻𝑐=𝐸𝑆𝑐, https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.bunchkaufman
#
# The solution of the system 𝐴𝑥=𝑏 where 𝐴=𝐿∗𝐷∗𝐿𝑇 is found as a result of consecutive solutions of systems 𝐿∗𝑦=𝑏 and 𝐷∗𝐿𝑇∗𝑥=𝑦
#------------------------------------------------------------------------------------------------------------------------------
using LinearAlgebra

function GSEPIIS(; param::Param=param, H=H, S=S, verbose1=verbose1, verbose2=verbose2)
    ok = true
    Max_iter = GSEPIIS_Max_iter
    
    #-----------------------------------
    # Check that the matrix H is square
    #------------------------------------
    n_rows, n_cols = size(H)   
    if n_rows != n_cols
        println("Matrix H is not square: dimensions are (", n_rows, " ", n_cols, ")")
        return false, 0
    end

    #--------------
    # Symmetrize H
    #--------------
    H = symm(H)
    
    #-----------------------------------
    # Check that the matrix S is square
    #------------------------------------
    n_rows, n_cols = size(S)   
    if n_rows != n_cols
        println("Matrix S is not square: dimensions are (", n_rows, " ", n_cols, ")")
        return false, 0
    end 
    
    #-------------------------------------------
    # Retrieve parameters from param structure
    #-------------------------------------------
    @unpack ApproxEnergy, EigvalTol, LastEigvector = param
    
    Evalue = ApproxEnergy
    
    #-----------------------------------------------------------------
    # Compute the Bunch-Kaufman factorization of symmetrized matrix H
    #-----------------------------------------------------------------
    try
        bkfH = bunchkaufman(Symmetric(H, :L)) # Bunch-Kaufman factorization
    
        L = bkfH.L  # Get L (unit lower triangular) part of Bunch-Kaufman factorization of H
        if verbose2
            println("\nlower triangular part of Bunch-Kaufman factorization of H")
            show(stdout, "text/plain", L)
            println(" ")
        end

        D = bkfH.D  # Get D (diagonal) part of Bunch-Kaufman factorization of H
        if verbose2
            println("\nDiagonal part of Bunch-Kaufman factorization of H")
            show(stdout, "text/plain", D)
            println(" ")
        end
    
        DLT = D*transpose(L)
    
        LDLT = L*DLT
    
        ST = transpose(S)
    
#--------------------------------------------------------------------------------------------------
# Do inverse iterations until the process converges with relative accuracy EigvalTol, 
# or until the number of iterations exceeds the maximum Max_iter
# The solution of the system A*x=b where A=L*D*LT is found as a result of consecutive solutions of
# systems L*y=b and D*LT*x=y. We use the Julia \ operation to solve these linear systems.
#---------------------------------------------------------------------------------------------------
        x = LastEigvector
    
        Diffprev = 10000
        ok = false
    
        for i in 1:Max_iter
            v = x
            w = ST*v
            if verbose2
                println("\ni: ", i,  " w = np.transpose(S)*v: ")
                show(stdout, "text/plain", w)
                println(" ")
            end
 
            y = L\w                           # The \ operation solves the linear solution
            if verbose2
                println("\ny = solve(L,w)")
                show(stdout, "text/plain", y)
                println(" ")
            end
    
            x = DLT\y
            if verbose2
                println("\nx = solve(DLT,y)")
                show(stdout, "text/plain", x)
                println(" ")
            end
            
            t1 = maximum(abs.(x))
            if verbose2
                println("\nt1: ", t1)
            end
        
            x = x/t1
            if verbose2
                println("\nx = x/t1")
                show(stdout, "text/plain", x)
                println(" ")
            end
        
            tc = (transpose(x)*v) / (transpose(v)*v)     # tc=(x^{T}v)/(v^{T}v)
        
            Diff = norm(x-tc*v) / norm(x)
            if verbose2
                println("\nDiff: ", Diff)
            end
        
            if Diff < EigvalTol
                ok = true
                if verbose1
                    println("\nGSEPIIS converged in ", i, " iterations")
                    println(trace_f, "\nGSEPIIS converged in ", i, " iterations")
                end
                break
            end
        end
    
        if !ok
            if verbose1
                println("\nGSEPIIS did not converge after ", Max_iter, " iterations")
                println(trace_f, "\nGSEPIIS did not converge after ", Max_iter, " iterations")
                return false, Evalue
            end
        end
        
        param.LastEigvector=x # Save eigenvector
    
        # Compute x^{T}Sx
        t1 = transpose(x)*S*x
        if verbose2
            println("t1 = transpose(x)*S*x: ", t1)
        end
        
        # Compute x^{T}LDLTx
        t2 = transpose(x)*LDLT*x
        if verbose2
            println("t2 = transpose(x)*LDLT*x: ", t2)
        end
        
        # Rayleigh quotient
        Evalue = (t2/t1) + ApproxEnergy
        if verbose1
            println("\nGSEPIIS - Evalue: ", Evalue)
            println(trace_f, "\nGSEPIIS - Evalue: ", Evalue)
        end
    
        return ok, Evalue
    
    catch e
        println("\nGSEPIIS - bunchkaufman failed")
        return false, 0
    end

end # function GSEPIIS

#------------------------------------------------------------------------------------------------------------------------------
# Define solve_eigen() that solves equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy using Julia linear algebra eigen() function which solves
# a standard or generalized eigenvalue problem for a complex Hermitian or real symmetric matrix: 
# https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedEigen.
#------------------------------------------------------------------------------------------------------------------------------
function solve_eigen(; param::Param=param, H=H, S=S, verbose1=verbose1, verbose2=verbose2)

    @unpack trace_f, GSEP_G, GSEP_I = param
    
    #-----------------------------------
    # Check that the matrix H is square
    #------------------------------------
    n_rows, n_cols = size(H)   
    if n_rows != n_cols
        println("Matrix H is not square: dimensions are (", n_rows, " ", n_cols, ")")
        println(trace_f, "Matrix H is not square: dimensions are (", n_rows, " ", n_cols, ")")
        return false, nothing, nothing, nothing
    end
    
    #-----------------------------------
    # Check that the matrix S is square
    #------------------------------------
    n_rows, n_cols = size(S)   
    if n_rows != n_cols
        println("Matrix S is not square: dimensions are (", n_rows, " ", n_cols, ")")
        println(trace_f, "Matrix S is not square: dimensions are (", n_rows, " ", n_cols, ")")
        return false, nothing, nothing, nothing
    end
    
    #--------------
    # Symmetrize H
    #--------------
    H = symm(H)
    
    #------------------------------------------------------------------------------------------------------------
    # Solve with Julia linear algebra eigen function
    # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedEigen
    # the eigenvalues can be obtained via F.values and the eigenvectors as the columns of the matrix F.vectors. 
    # (The kth eigenvector can be obtained from the slice F.vectors[:, k].)
    #------------------------------------------------------------------------------------------------------------
    F = nothing
    try
        F = eigen(H, S)

        #------------------------
        # Update param structure
        #------------------------
        param.Evectors = F.vectors                     # Eigenvectors
        param.Evalues = real(F.values)
        param.Evalue_eigen = minimum(real(F.values))

        if GSEP_I
            param.CurrEnergy = param.Evalue_eigen + param.ApproxEnergy
        else
            param.CurrEnergy = param.Evalue_eigen
        end

        #----------------------------------------------------------------------------------
        # Compute the condition as the ratio between the eigenvalue with largest magnitude 
        # over the one with the smallest magnitude
        #----------------------------------------------------------------------------------
        condition = condition_eigenvalues(F.values)
        param.condition = condition

        #---------------------
        # Print eigen values
        #---------------------
        if verbose1
            print_eigenvalues(F.values, param.ApproxEnergy; verbose1=verbose1, GSEP_G=GSEP_G, GSEP_I=GSEP_I)
        end
    
        return true, param.Evalue_eigen, param.Evalues, condition
        
    catch e
        println("\nsolve_eigen - eigen failed")
        catch_backtrace()
        return false, nothing, nothing, nothing
    end
end

#------------------------------------------------------------------------------------------------------------------------------
# Define solve_sec() that solves equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy
#
# solve_sec() uses the following methods to solve secular equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy:
#
# - GSEPIIS() function which finds the solution of the system 𝐴𝑥=𝑏 where 𝐴=𝐿∗𝐷∗𝐿𝑇 as a result of consecutive solutions of 
#   systems 𝐿∗𝑦=𝑏 and 𝐷∗𝐿𝑇∗𝑥=𝑦. The Julia \ operation solves the linear solution.
#
# - Julia linear algebra eigen() function which solves a standard or generalized eigenvalue problem for a complex Hermitian or
#   real symmetric matrix: https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedEigen.
#------------------------------------------------------------------------------------------------------------------------------
function solve_sec(; param::Param=param, H=H, S=S, verbose1=verbose1, verbose2=verbose2)
    
    ok = true
    Evalue_GSEPIIS = nothing
    
    #-------------------------------------------
    # Retrieve parameters from param structure
    #-------------------------------------------
    @unpack CurrEnergy, GSEP_G, GSEP_I = param
    
    #-----------------------------------
    # Check that the matrix H is square
    #------------------------------------
    n_rows, n_cols = size(H)   
    if n_rows != n_cols
        println("Matrix H is not square: dimensions are (", n_rows, " ", n_cols, ")")
        return false, nothing, nothing, nothing
    end
    
    #-----------------------------------
    # Check that the matrix S is square
    #------------------------------------
    n_rows, n_cols = size(S)   
    if n_rows != n_cols
        println("Matrix S is not square: dimensions are (", n_rows, " ", n_cols, ")")
        return false, nothing, nothing, nothing
    end 

    #---------------------------------
    # Solve with solve_eigen function
    #---------------------------------
    ok, Evalue_eigen, Evalues, condition = solve_eigen(; param=param, H=H, S=S, verbose1=verbose1, verbose2=verbose2)
    if ok
        Evalue_GSEPIIS = CurrEnergy
    end
    
    #-----------------------------
    # Solve with GSEPIIS function
    #-----------------------------
    if !CGL0 &&!N_C_Pvec && do_GSEPIIS
        ok, Evalue_GSEPIIS = GSEPIIS(H=H, S=S, verbose1=verbose1, verbose2=verbose2)
        if ok
            Evalue_GSEPIIS = max(Evalue_GSEPIIS, param0.TargetEnergy)
        end
    end
    
    return ok, param.CurrEnergy, Evalue_GSEPIIS, Evalue_eigen, Evalues, condition
    
end

#---------------------------------------------------------------------------------------------------------
# Define compute_and_solve() that computes H and S matrices and solves equation 𝐻𝑐=𝐸𝑆𝑐 to find the energy
#---------------------------------------------------------------------------------------------------------
function compute_and_solve(; param::Param=param, verbose1=verbose1, verbose2=verbose2, print=true, H_name="H", S_name="S")
    
    #-------------------------------------------
    # Retrieve parameters from param structure
    #-------------------------------------------
    @unpack GSEP_G, GSEP_I = param
    
    #--------------------------
    # Compute H and S matrices  
    #--------------------------
    ok = compute_H_S(; param=param, verbose1=verbose1, verbose2=verbose2, print=print)
    if !ok
        println("\ncompute_and_solve - compute_H_S() failed")
        return ok
    end

    #--------------------------------------
    # Write H and S matrices to text files 
    #--------------------------------------
    if H_name != nothing
        write_matrix(param.H; matrix_name=H_name, verbose1=verbose1)
    end
    
    if S_name !=nothing
        write_matrix(param.S; matrix_name=S_name, verbose1=verbose1)
    end
    
    #--------------------------------------------------
    # Solve secular equation with solve_eigen function
    #--------------------------------------------------
    ok, Evalue_eigen, Evalues, condition = solve_eigen(; param=param, H=param.H, S=param.S, verbose1=verbose1, verbose2=verbose2)
    if !ok
        if verbose1
            println("\ncompute_and_solve - solve_eigen failed")
        end
    end

    return ok

end # compute_and_solve


function save_state(; param::Param=param, param1::Param=param1)
    
    param1.ZIndex_used = param.ZIndex_used
    # Copy ZIndex
    if param.ZIndex_used
        param1.ZIndex = copy(param.ZIndex)
    end
    
    param1.NonlinParam = copy(param.NonlinParam)
        
    # Copy H, diagH, S, diagS and D
    param1.H = copy(param.H)
    param1.diagH = copy(param.diagH)
    param1.S = copy(param.S)
    param1.diagS = copy(param.diagS)
    param1.D = copy(param.D)

    param1.CurrEnergy = param.CurrEnergy
    param1.InvitParameter = param.InvitParameter
    param1.ApproxEnergy = param.ApproxEnergy
    param1.EigvalTol = param.EigvalTol

    # Copy LastEigvector
    param1.LastEigvector = copy(param.LastEigvector)

    # Copy grad_k and grad_l flags
    param1.grad_k = param.grad_k
    param1.grad_l = param.grad_l

    # Copy matrix Evectors and Evalues
    param1.Evectors = copy(param.Evectors)
    param1.Evalue_eigen = param.Evalue_eigen
    param1.Evalues = copy(param.Evalues)
    param1.condition = param.condition

    return
end

function restore_state(; param::Param=param, param1::Param=param1)
    
    param.ZIndex_used = param1.ZIndex_used
    # Restore Zindex
    if param.ZIndex_used
        param.ZIndex = param1.ZIndex
    end
    
    param.NonlinParam = param1.NonlinParam
        
    # Restore H, diagH, S, diagS and D
    param.H = param1.H
    param.diagH = param1.diagH
    param.S = param1.S
    param.diagS = param1.diagS
    param.D = param1.D

    param.CurrEnergy = param1.CurrEnergy
    param.InvitParameter = param1.InvitParameter
    param.ApproxEnergy = param1.ApproxEnergy
    param.EigvalTol = param1.EigvalTol

    # Restore LastEigvector
    param.LastEigvector = param1.LastEigvector

    param.grad_k = param1.grad_k
    param.grad_l = param1.grad_l

    # Restore matrix Evectors and Evalues
    param.Evectors = param1.Evectors
    param.Evalue_eigen = param1.Evalue_eigen
    param.Evalues = param1.Evalues
    param.condition = param1.condition
    
    return
end

#--------------------------------------------------------------------------------------------------------------------
# Define show_status() that prints current energy, maximum overlap between functions and distances between functions
#--------------------------------------------------------------------------------------------------------------------
function show_status(; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    s = @sprintf("%.2E", param.condition)
    s1 = @sprintf("%.2E", maximum(abs.(param.NonlinParam)))

    CurrEnergy = param.CurrEnergy
    InvitParameter = param.InvitParameter
    ApproxEnergy = param.ApproxEnergy
    TargetEnergy = param0.TargetEnergy
            
    println("\nCurrent energy: $CurrEnergy InvitParameter: $InvitParameter Approximate Energy: $ApproxEnergy Target energy: $TargetEnergy")
    println("Eigenvalue tolerance: ", param.EigvalTol, " Overlap threshold: ", param0.overlap_threshold,
    " Condition: ", s, " Condition maximum threshold: ", param0.condition_max)

    println("Maximum magnitude of non-linear basis set parameters: ", s1, " Non-linear basis set parameter threshold: ", param0.nlp_threshold)
            
    println(trace_f, "\nCurrent energy: $CurrEnergy InvitParameter: $InvitParameter Approximate Energy: $ApproxEnergy Target energy: $TargetEnergy")
    println(trace_f, "Eigenvalue tolerance: ", param.EigvalTol, " Overlap threshold: ", param0.overlap_threshold,
    " Condition: ", s, " Condition maximum threshold: ", param0.condition_max)
    println(trace_f, "Maximum magnitude of non-linear basis set parameters: ", s1, " Non-linear basis set parameter threshold: ", param0.nlp_threshold)

    # Show maximum overlap between functions
    i_S, j_S, max_S = max_overlap(; param=param, verbose1=verbose1, verbose2=verbose2)

    # Get distance between diagonal elements of Hamiltonian matrix H
    perm_H, map_H, dist_H, x_H = dist_diagH(; param=param, verbose1=verbose1, verbose2=true)

    # Get distances between functions
    perm_func, map_func, dist_func, x_func = dist_functions(; param=param, verbose1=verbose1, verbose2=true)

    return
end

#---------------------------------------------------------------------------------------------------------------------
# gen_trial_param!  --  Julia port of the Fortran subroutine GenerateTrialParam [ATOMMOLnonBO].
#
# Fills rows Kstart:Kstop of param.NonlinParam with trial nonlinear parameters generated from the template basis
# NonlinParam_t (size_t rows). Two stochastic generation methods (Sharkey/Bubin RGL0 code):
#   method 1 (probability param0.RG_p1):   each new parameter = (1 + RG_s1*Z)*template, Z~N(0,1) drawn
#                                           independently per parameter (independent normal scaling);
#   method 2 (probability 1 - RG_p1):       all parameters of the selected template function are multiplied by a
#                                           single common factor (1 + RG_s2*Z), redrawn until |factor| lies outside
#                                           (0.8, 1.2) so the candidate is not almost linearly dependent.
# A template function is selected at random: when size_t < nfun (too few to take a consecutive block) each output
# function picks its own template index; otherwise a consecutive block k0 : k0+nfun-1 is used. When size_t == 0 the
# parameters are drawn uniformly in [-0.5, 0.5) (only the np = n(n+1)/2 lower-triangle entries; any imaginary/B
# entries stay 0), as in the Fortran Glob_CurrBasisSize == 0 case. Returns ksel, the per-output selected template
# index (used by the caller to copy the ZIndex prefactor); ksel entries are 0 in the size_t == 0 case.
#
# Fortran mapping: random_number -> rand() (uniform [0,1)), drnor() -> randn() (standard normal),
# int(r*N)+1 -> floor(Int, rand()*N)+1, Glob_NonlinParam(j,k) -> NonlinParam_t[k, j], Glob_CurrBasisSize -> size_t.
#
# Reference
# [ATOMMOLnonBO] S. Bubin, L. Adamowicz, *Computer program* `atom-mol-nonBO`…, J. Chem. Phys. **152**, 204102 (2020). 
# [doi:10.1063/1.5144268](https://doi.org/10.1063/1.5144268) · [GitHub](https://github.com/sbubin/ATOM-MOL-nonBO)
#---------------------------------------------------------------------------------------------------------------------
function gen_trial_param!(param::Param, NonlinParam_t, size_t::Int, Kstart::Int, Kstop::Int)
    npt  = param.npt
    np   = Int(param.n*(param.n + 1)/2)
    p1   = param0.RG_p1
    s1   = param0.RG_s1
    s2   = param0.RG_s2
    nfun = Kstop - Kstart + 1
    ksel = zeros(Int, nfun)

    if size_t == 0
        # Glob_CurrBasisSize == 0: uniform L in [-0.5, 0.5), imaginary/B entries left at 0
        for (c, i) in enumerate(Kstart:Kstop)
            row = zeros(npt)
            for j in 1:np
                row[j] = rand() - 0.5
            end
            param.NonlinParam[i, :] = row
        end
        return ksel
    end

    if size_t < nfun
        # Not enough template functions for a consecutive block: pick one method, then a random template per output
        method1 = rand() < p1
        for (c, i) in enumerate(Kstart:Kstop)
            k = floor(Int, rand()*size_t) + 1
            ksel[c] = k
            if method1
                param.NonlinParam[i, :] = (s1 .* randn(npt) .+ 1.0) .* NonlinParam_t[k, :]
            else
                r = s2*randn() + 1.0
                while (abs(r) > 0.8) && (abs(r) < 1.2)
                    r = s2*randn() + 1.0
                end
                param.NonlinParam[i, :] = r .* NonlinParam_t[k, :]
            end
        end
    else
        k0 = floor(Int, rand()*(size_t - nfun + 1)) + 1   # consecutive block start
        if rand() < p1
            for (c, i) in enumerate(Kstart:Kstop)
                k = k0 + (c - 1); ksel[c] = k
                param.NonlinParam[i, :] = (s1 .* randn(npt) .+ 1.0) .* NonlinParam_t[k, :]
            end
        else
            r = s2*randn() + 1.0                           # one common factor for the whole block
            while (abs(r) > 0.8) && (abs(r) < 1.2)
                r = s2*randn() + 1.0
            end
            for (c, i) in enumerate(Kstart:Kstop)
                k = k0 + (c - 1); ksel[c] = k
                param.NonlinParam[i, :] = r .* NonlinParam_t[k, :]
            end
        end
    end
    return ksel
end

#--------------------------------------------------------------------------
# Define stoch_nlp() that randomly selects non linear basis set parameters
#--------------------------------------------------------------------------
function stoch_nlp(; param::Param=param, Kstart=1, Kstop=1, MaxEnergyEval=100, seed=nothing, verbose1=verbose1, verbose2=verbose2)

    #------------------------------------------
    # Exit if param.CurrEnergy <= param0.TargetEnergy
    #------------------------------------------
    if param.CurrEnergy <= param0.TargetEnergy
        if verbose1
            println("Exiting since current energy ", param.CurrEnergy, " is lower than target energy ", param0.TargetEnergy)
            println(trace_f, "\nstoch_nlp - Exiting since current energy ", param.CurrEnergy, " is lower than target energy ", param0.TargetEnergy)
        end
        return false, 0
    end

    #---------------------------------------
    # Ensure Kstop is in range[Kstart, cbs]
    #---------------------------------------
    cbs = size(param.NonlinParam)[1]
    Kstop = min(Kstop, cbs)
    
    if Kstop < Kstart
        if verbose1
            println("Exiting since Kstop ", Kstop, " is less than Kstart ", Kstart)
            println(trace_f, "\nstoch_nlp - Exiting since Kstop ", Kstop, " is less than Kstart ", Kstart)
        end
        return false, 0
    end

    #---------------------------------------------------------------------------------------------------
    # Check that the number of basis set parameters per line in param.NonlinParam is equal to param.npt
    #---------------------------------------------------------------------------------------------------
    npt = size(param.NonlinParam)[2]
    if npt != param.npt
        if verbose1
            println("\nNumber of parameters per line in NonlinParam, npt ", npt, " is not equal to the one in param structure ", param.npt)
            println(trace_f, "\nNumber of parameters per line in NonlinParam, npt ", npt, " is not equal to the one in param structure ", param.npt)
        end
        return false, 0
    end

    
    #-----------------------------------------------
    # Update nfru, Nmin and Nmax in param structure
    #-----------------------------------------------
    param.nfru = max(0, Kstart-1)
    param.Nmin = Kstart
    param.Nmax = Kstop

    # Compute number of functions to randomly select
    nfunc = Kstop-Kstart+1
    
    #------------------------------
    # Set gradient flags to false 
    #------------------------------
    grad_k = param.grad_k
    grad_l = param.grad_l
    
    param.grad_k = false
    param.grad_l = false

    #----------------------------------------------------------
    # Reseed the random number generator if a seed is provided
    #----------------------------------------------------------
    if seed != nothing
        Random.seed!(seed)
    end
    
    #------------------------------------------------------------------------------------------
    # Get overlap_threshold, nlp_threshold and dist_diagH_threshold from param0 data structure
    #------------------------------------------------------------------------------------------
    overlap_threshold = param0.overlap_threshold
    test_overlap = overlap_threshold > 0.0

    test_condition = param0.condition_max > 0
    
    nlp_threshold = param0.nlp_threshold
    test_nlp = nlp_threshold > 0.0
    
    dist_diagH_threshold = param0.dist_diagH_threshold
    test_dist_diagH = dist_diagH_threshold > 0.0

    #------------------
    # Print parameters
    #------------------
    if verbose1

        s = @sprintf("%.2E", param.condition)
        s1 = @sprintf("%.2E", maximum(abs.(param.NonlinParam)))
        
        if seed == nothing
            println("\nRandomly selecting basis functions from $Kstart to $Kstop Max energy evaluation: $MaxEnergyEval")
        else
            println("\nRandomly selecting basis functions from $Kstart to $Kstop Max energy evaluation: $MaxEnergyEval seed: $seed")
        end
        
        println("\nCurrent energy: ", param.CurrEnergy, " InvitParameter: ", param.InvitParameter, " Approximate Energy: ", param.ApproxEnergy, 
                " Target energy: ", param0.TargetEnergy)
        
        println("Eigenvalue tolerance: ", param.EigvalTol, " Overlap threshold: ", param0.overlap_threshold, " Condition: ", s, 
            " Condition maximum threshold: ", param0.condition_max)

        println("Maximum magnitude of non-linear basis set parameters: ", s1, " Non-linear basis set parameter threshold: ", nlp_threshold)

        if seed == nothing
            println(trace_f, "\nRandomly selecting basis functions from $Kstart to $Kstop Max energy evaluation: $MaxEnergyEval")
        else
            println(trace_f, "\nRandomly selecting basis functions from $Kstart to $Kstop Max energy evaluation: $MaxEnergyEval seed: $seed")
        end
        
        println(trace_f, "\nCurrent energy: ", param.CurrEnergy, " InvitParameter: ", param.InvitParameter, " Approximate Energy: ", param.ApproxEnergy, 
                " Target energy: ", param0.TargetEnergy)

        println(trace_f, "Eigenvalue tolerance: ", param.EigvalTol, " Overlap threshold: ", param0.overlap_threshold, " Condition: ", s, 
            " Condition maximum threshold: ", param0.condition_max)

        println(trace_f, "Maximum magnitude of non-linear basis set parameters: ", s1, " Non-linear basis set parameter threshold: ", nlp_threshold)
    end
    
    #--------------------------------------------------------------------------
    # Copy eigen values, current energy, NonLinParam, ZIndex, H and S matrices
    #--------------------------------------------------------------------------
    # Copy Evalue_eigen and Evalue_GSEPIIS
    Evalue_eigen = param.Evalue_eigen

    # Copy matrix Evectors and Evalues
    Evectors = copy(param.Evectors)
    Evalues = copy(param.Evalues)

    # Copy approximate and current energy
    ApproxEnergy = param.ApproxEnergy
    CurrEnergy = param.CurrEnergy
    
    # Copy NonLinParam
    NonlinParam = copy(param.NonlinParam)

    # Copy ZIndex
    if param.ZIndex_used
        ZIndex = copy(param.ZIndex)
    end

    size_db = size(param.NonlinParam_db)[1]
    
    # Set-up template NonLinParam and template ZIndex.
    # ZIndex_t_defined records whether a valid template ZIndex array (ZIndex_t) was actually assigned,
    # so the two ZIndex-copy sites below use one consistent, correct condition. (Previously the two
    # sites used different guards that did not match this assignment, so a new basis function could
    # either keep its default ZIndex = 1 (the prefactor BFPI[1,:] = (1,2,3)) instead of the template's
    # intended prefactor, or reference an undefined ZIndex_t. The valid prefactors for an atom with n
    # pseudoparticles are all BFPI rows whose indices are <= n: rows 1-35 (all C(7,3)=35 triples) for
    # Nitrogen, n=7.)
    ZIndex_t_defined = false
    if param.nlp0 || Kstop > size_db
        NonlinParam_t = param.NonlinParam
        size_t = cbs

        if param.ZIndex_used
            ZIndex_t = param.ZIndex
            ZIndex_t_defined = true
        end
    else
        NonlinParam_t = param.NonlinParam_db
        size_t = size_db

        if param.ZIndex_used && param.ZIndex_used_db
            ZIndex_t = param.ZIndex_db
            ZIndex_t_defined = true
        end
    end
    
    # Copy H, diagH, S and diagS
    H = copy(param.H)
    diagH = copy(param.diagH)
    S = copy(param.S)
    diagS = copy(param.diagS)
    
    # shift vector
    shift = 0.5.*ones(nfunc, param.npt)

    # Text to be displayezd if doubly degenerate eigen values are found 
    if param0.discard_degenerate
        s_degenerate = " discarded"
    else
        s_degenerate = ""
    end
    
    ok = true
    found = false

    # Initialize number of energy evaluations
    neval = 0
    
    for u in 1:MaxEnergyEval
        neval += 1
        #--------------------------------------------------------------------------------------------
        # Update non linear basis set parameters from Kstart to Kstop
        #
        # Set up trial nonlinear parameters for functions to be updated in basis set
        # Create an array of pseudorandom numbers from the uniform distribution over the range [0 1)
        # Julia Base.rand function https://docs.julialang.org/en/v1/stdlib/Random/
        #--------------------------------------------------------------------------------------------
        if nfunc == 1
            if param0.gen_NonlinParam
                # First method to try: Julia port of the Fortran GenerateTrialParam (perturb a random template).
                ksel = gen_trial_param!(param, NonlinParam_t, size_t, Kstop, Kstop)

                # Copy ZIndex from the selected template function (the prefactor BFPI row it used).
                if ZIndex_t_defined && ksel[1] > 0
                    param.ZIndex[Kstop] = ZIndex_t[ksel[1]]
                end
            else
                x = rand(Float64,(nfunc, param.npt)) - shift
                r = rand(0.8:1.2)

                # Get one basis set parameter from NonlinParam
                v = mod(u,size_t) + 1
                param.NonlinParam[Kstop,:] = r.*(param0.coeff_nlp.*x[1,:] + NonlinParam_t[v,:])

                # Copy ZIndex from the template (only when a valid template ZIndex array was set up),
                # so the new function inherits the template's prefactor index (a valid BFPI row).
                if ZIndex_t_defined
                    param.ZIndex[Kstop] = ZIndex_t[v]
                end
            end

        else
            if param0.gen_NonlinParam
                # First method to try: Julia port of the Fortran GenerateTrialParam for the whole Kstart:Kstop block.
                ksel = gen_trial_param!(param, NonlinParam_t, size_t, Kstart, Kstop)

                if ZIndex_t_defined
                    for (c, i) in enumerate(Kstart:Kstop)
                        ksel[c] > 0 && (param.ZIndex[i] = ZIndex_t[ksel[c]])
                    end
                end
            else
                if param0.shuffle_NonlinParam
                    # Shuffle NonlinParam
                    ix = shuffle(Vector(1:size_t))
                else
                    ix = Vector(1:size_t)
                end

                # Set up items from Kstart to Kstop in NonlinParam
                for i in Kstart:Kstop
                    x = rand(Float64,(nfunc, param.npt)) - shift
                    r = rand(0.8:1.2)
                    m = minimum(abs.(NonlinParam_t[ix[i],:]))
                    param.NonlinParam[i,:] = r.*(m.*param0.coeff_nlp.*x[i-Kstart+1,:] + NonlinParam_t[ix[i],:])

                    # Copy ZIndex from the template (only when a valid template ZIndex array was set up),
                    # so the new function inherits the template's prefactor index (a valid BFPI row).
                    if ZIndex_t_defined
                        param.ZIndex[i] = ZIndex_t[ix[i]]
                    end
                end
            end
        end

        # all_real == "T" with the complex N_C_Pvec method: keep the trial functions purely real by zeroing the
        # imaginary (B) block of the generated rows (matches the N_C_Pvec matelem, which ignores B when all_real == "T").
        if N_C_Pvec && all_real == "T"
            np1 = Int(param.n*(param.n+1)/2)
            for i in Kstart:Kstop
                param.NonlinParam[i, np1+1:param.npt] .= 0.0
            end
        end

        #-----------------------------------------------------
        # Compute H and S matrices and solve secular equation
        #-----------------------------------------------------
        ok = compute_and_solve(; param=param, verbose1=false, verbose2=false, print=false, H_name=nothing, S_name=nothing)


        # Get distances between functions
        perm_func, map_func, dist_func, x_func = dist_functions(; param=param, verbose1=verbose1, verbose2=verbose2)

        #--------------------------------------------
        # If compute and solve failed, then continue
        #--------------------------------------------
        if !ok
            if verbose1
                println("\nstoch_nlp - compute_and_solve failed")
            end

        #----------------------------------------------------------------------------------------------------
        # If the minimum distance between functions is below dist_func_threshold, continue to next iteration
        #----------------------------------------------------------------------------------------------------
        elseif dist_func[perm_func[1]] < param0.dist_func_threshold
            continue

        #--------------------------------------------------------------------------
        # If the distance threshold is below threshold, continue to next iteration
        #--------------------------------------------------------------------------
        elseif test_dist_diagH && !distance_ok(Kstart, Kstop; param=param, verbose1=verbose1, verbose2=verbose2)
            continue

        #--------------------------------------------------------------------------------------------------------
        # If the maximum magnitude of the non-linear basis set parameters exceeds nlp_threshold, print a message
        #--------------------------------------------------------------------------------------------------------
        elseif test_nlp && maximum(abs.(param.NonlinParam)) > nlp_threshold
            if verbose2
                println("\nMaximum magnitude of non-linear basis set parameters ", maximum(abs.(param.NonlinParam)), 
                    " exceeds non-linear coefficient threshold ", LinCoeffThreshold)
                println(trace_f, "\nstoch_nlp - Maximum magnitude of non-linear basis set parameters ", maximum(abs.(param.NonlinParam)),
                    " exceeds non-linear coefficient threshold ", LinCoeffThreshold)
            end
                
        #-------------------------------------------------------
        # If the divide by zero flag is raised, print a message 
        #-------------------------------------------------------
        elseif param.div_by_zero
            
            param.n_div_by_zero += 1
            param.div_by_zero = false    # Reset flag for next iteration
            
            if param.n_div_by_zero > 10
                if verbose1
                    println("\nDivide by zero in store_HS() occured more than ten times")
                    println(trace_f, "\nstoch_nlp - Divide by zero in store_HS occured more than ten times")
                end
                break
            end

        #-----------------------------------------------------------------------------
        # If the energy is lower than param0.TargetEnergy, continue to next iteration
        #-----------------------------------------------------------------------------
        elseif param.CurrEnergy < param0.TargetEnergy
            continue

         #--------------------------------------------------
         # If the energy is NaN, continue to next iteration
         #--------------------------------------------------
         elseif isnan(param.CurrEnergy)
            if verbose1
                println("\nEnergy is ", param.CurrEnergy)
                println(trace_f, "\nEnergy is ", param.CurrEnergy)
            end
            continue
            
        #----------------------------------------------------------------------------------------------------------
        # If a lower value of the energy is found, copy current energy, eigenvector, NonLinParam, H and S matrices
        #----------------------------------------------------------------------------------------------------------
        elseif param.CurrEnergy < CurrEnergy - param.EigvalTol

            # Compute H and S matrices and solve secular equation with the complete basis set
            param.Nmin = 1
            ok = compute_and_solve(; param=param, verbose1=false, verbose2=false, print=false, H_name=nothing, S_name=nothing)

            if param.CurrEnergy >= CurrEnergy - param.EigvalTol
                continue
            end

            # Get maximum overlap between functions
            ok_overlap = true
            if test_overlap 
                ok_overlap, ii, jj, max_S = max_overlap(; param=param, verbose1=false, verbose2=verbose2)
            end

            #------------------------------------------------------------------
            # If the overlap exceeds threshold, continue to the next iteration
            #------------------------------------------------------------------
            if !ok_overlap
                continue

            #------------------------------------------------------------------------------------------------
            # If the condition of the eigenvalues exceeds param0.condition_max, print a message and continue
            #------------------------------------------------------------------------------------------------
            elseif test_condition && param.condition > param0.condition_max
                if verbose2
                    s = @sprintf("%.2E", param.condition)
                    println("\nCondition ", s, " exceeds maximum threshold ")
                    println(trace_f, "\nstoch_nlp - condition ", s, " exceeds maximum threshold ")
                end
                continue

            #-------------------------------------------------------------------------------------
            # If this is the first time a function is found which lowers the energy, print header
            #-------------------------------------------------------------------------------------
            else
                if found == false && verbose1
                    println("\nIteration    Best energy")
                    println(trace_f, "\nIteration    Best energy")
                end
                found = true

                #------------------------------------------------------------------------------------------
                # If there are at least two eigenvalues and the absolute value of their difference is less 
                # than twice the eigenvalue tolerance, then print a message
                #------------------------------------------------------------------------------------------
                if size(param.Evalues)[1] > 1 && abs(param.Evalues[1] - param.Evalues[2]) < 2.0 * param.EigvalTol
                    if verbose1
                        println(u, "         ", real(param.Evalues[1]), " ", real(param.Evalues[2]), " degenerate state", s_degenerate)
                        println(trace_f, u, "         ", real(param.Evalues[1]), " ", real(param.Evalues[2]), " degenerate state", s_degenerate)
                    end

                    # If param0.discard_degenerate is true, then continue to next iteration
                    if param0.discard_degenerate 
                        continue
                    end
                end

                # Reset Nmin to Kstart for next iteration
                param.Nmin = Kstart
                
                # Update current energy
                CurrEnergy = param.CurrEnergy

                if verbose1
                    println("$u         $CurrEnergy")
                    println(trace_f, "$u         $CurrEnergy")
                end
            
                # Update Evalue_eigen
                Evalue_eigen = param.Evalue_eigen

                # Copy matrix Evectors and Evalues
                Evectors = copy(param.Evectors)
                Evalues = copy(param.Evalues)
            
                # Copy NonLinParam
                NonlinParam = copy(param.NonlinParam)

                # Copy ZIndex
                if param.ZIndex_used
                    ZIndex = copy(param.ZIndex)
                end
            
                # Copy H and S matrices
                H = copy(param.H)
                diagH = copy(param.diagH)
                S = copy(param.S)
                diagS = copy(param.diagS)
            end
        else
            continue
        end
    end

    
    #-------------------------------------------------------------------------------------
    # Update param data structure with saved values which can be either of the following:
    # - values saved before entering the loop if no lower energy was found
    # - values pertaining to the best lower energy found
    #-------------------------------------------------------------------------------------
    param.Evectors = Evectors
    param.Evalue_eigen = Evalue_eigen
    param.Evalues = Evalues
    param.CurrEnergy = CurrEnergy
    param.NonlinParam = NonlinParam
    
    if param.ZIndex_used
        param.ZIndex = ZIndex
    end
    
    param.H = H
    param.diagH = diagH
    param.S = S
    param.diagS = diagS

    #----------------------------------------------
    # Reset gradient flags to their initial values
    #----------------------------------------------
    param.grad_k = grad_k
    param.grad_l = grad_l
    
    if found 
        #--------------------------------------------------
        # A function that lowers the energy has been found
        #--------------------------------------------------
        # Write H and S matrices to text files 
        write_matrix(H; matrix_name="H", verbose1=verbose1)
        write_matrix(S; matrix_name="S", verbose1=verbose1)

        # Write matrix of eigenvectors to a text file
        write_matrix(param.Evectors; matrix_name="Evectors", verbose1=verbose1)

         # And write it also to a text file with real part and imaginary part following one another
        if CGL0 || N_C_Pvec
            write_matrix_real(; matrix_name = "Evectors", suffix="_real", verbose1=false)
        end

        # Write non-linear parameters to text file 
        write_NonlinParam(param.cbs, param.npt, param.ZIndex, param.NonlinParam, ZIndex_used=param.ZIndex_used, verbose1=verbose1)

        #----------------------------------------------
        # Print H and S matrices if cbs <= max_print_H 
        #----------------------------------------------
        if param.cbs <= max_print_H
            print_HS(param.H, param.S)
        end

        #-----------------------------------------------------------------------------------------------
        # Print final current energy, maximum overlap between functions and distances between functions 
        #-----------------------------------------------------------------------------------------------
        if verbose1  
             show_status(; param=param, verbose1=verbose1, verbose2=verbose2)
        end

        #-------------------
        # Print eigenvalues
        #-------------------
        print_eigenvalues(param.Evalues, param.ApproxEnergy; verbose1=verbose1, GSEP_G=param.GSEP_G, GSEP_I=param.GSEP_I)

        #------------------------------------------
        # Print eigenvectors if cbs <= max_print_H 
        #------------------------------------------
        if param0.print_eigenvectors && param.cbs <= max_print_H
            println(trace_f,"\nEigenvectors matrix of eigenvectors")
            show(trace_f, "text/plain", param.Evectors)
            println(trace_f, " ")
        end

    elseif verbose1
        #------------------------------------
        # No function found, print a message
        #------------------------------------
        println("\nstoch_nlp did not find a function that lowers the energy")
        println(trace_f, "\nstoch_nlp did not find a function that lowers the energy")
    end
    
    return ok && found, neval

end # function stoch_nlp

#------------------------------------------------------------------------------------------------------------------------------
# Define do_basis_repl() that replaces the basis set
#------------------------------------------------------------------------------------------------------------------------------
function do_basis_repl(action_item; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    ok = true

    it = action_item
    ncycles = it.ntrials
    Kstart = it.Kstart
    Kstop = it.Kstop
    seed = it.seed
    ntrials = it.ntrials
    MaxEnergyEval = it.MaxEnergyEval
    param0.nlp0 = it.nlp0
    param0.coeff_nlp = it.coeff_nlp

    # Initialize number of energy evaluations
    neval = MaxEnergyEval

    if verbose1
        println("\nReplacing basis from $Kstart to $Kstop, seed: $seed")
        println(trace_f, "\nReplacing basis from $Kstart to $Kstop, seed: $seed")
    end

    #------------------------------------------------
    # Randomly select functions from Kstart to Kstop
    #------------------------------------------------
    for k in 1:ntrials
        if verbose1
            println("\nTrial: $k out of $ntrials")
            println(trace_f, "\nTrial: $k out of $ntrials")
        end
        
        ok, neval = stoch_nlp(Kstart=Kstart, Kstop=Kstop, MaxEnergyEval=MaxEnergyEval, seed=seed, verbose1=verbose1, verbose2=verbose2)
        
        if !ok
            break
        end
    end

    if ok
        #-------------------------------
        # Success - Update history list
        #-------------------------------
        for k in Kstart:Kstop
            history_update(param.CurrEnergy, ncycles, Kstart, neval, seed; rank=k)
        end
    end
    
    return ok
end

#------------------------------------------------------------------------------------------------------------------------------
# Define do_basis_enl() that enlarges the basis set
#------------------------------------------------------------------------------------------------------------------------------
function do_basis_enl(action_item; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    ok = true

    it = action_item

    #--------------------------------------------------------
    # Retrieve parameters from param structure
    # npt - number of nonlinear parameters per basis function
    #---------------------------------------------------------
    @unpack nfru, Nmin, Nmax, cbs, n, npt, H, diagH, S, diagS, D, ZIndex, NonlinParam, LastEigvector = param

    #------------------------------
    # Update Kstart in action item
    #------------------------------
    it.Kstart = param.cbs + 1

    #------------------------------------------------------
    # Get Kstart, Kstop, MaxEnergyEval, nlp0 and coeff_nlp
    #------------------------------------------------------
    Kstart = it.Kstart
    Kstop = it.Kstop
    MaxEnergyEval = it.MaxEnergyEval
    neval = MaxEnergyEval
    param0.nlp0 = it.nlp0
    param0.coeff_nlp = it.coeff_nlp
    
    #---------
    # Get nfa 
    #---------
    nfa = max(it.nfa, it.Kstart)
    
    if verbose1
        println("\ndo_basis_enl - Enlarging basis from $(Kstart-1) to $nfa seed: $seed")
        println(trace_f, "\ndo_basis_enl - Enlarging basis from $(Kstart-1) to $nfa seed: $seed")
    end
    
    #----------------------------
    # Set up enlarged H1, diagH1 
    #----------------------------
    if CGL0 || N_C_Pvec
        H1 = zeros(ComplexF64,nfa,nfa)
    else
        H1 = zeros(Float64,nfa,nfa)
    end
    diagH1 = zeros(Float64,nfa)

    #----------------------------
    # Set up enlarged S1, diagS1 
    #----------------------------
    if CGL0 || N_C_Pvec
        S1 = zeros(ComplexF64,nfa,nfa)
    else
        S1 = zeros(Float64,nfa,nfa)
    end
    diagS1 = zeros(Float64,nfa)
    
    #-----------------------------------
    # Initialize H1, diagH1, S1, diagS1 
    #-----------------------------------
    for i in 1:cbs
        diagH1[i] = diagH[i]
        diagS1[i] = diagS[i]
        for j in 1:i
            H1[i,j] = H[i,j]
            S1[i,j] = S[i,j]
        end
        for j in i+1:cbs
            S1[i,j] = S[i,j]
        end
    end
    
    #-----------------------------------------------------------------------------------------------------------
    # Create new matrix D which contains the derivatives of the Hamiltonian H and the overlap matrix elements S
    # D(1:np,i,j) contains dHij/dvechLi
    # D(np+1:2*np,i,j) contains dSij/dvechLi 
    #-----------------------------------------------------------------------------------------------------------
    if CGL0 || N_C_Pvec
        D1 = zeros(ComplexF64,2*npt,nfa,nfa)
    else
        D1 = zeros(Float64,2*npt,nfa,nfa)
    end
    
    #-------------------------------
    # Create ZIndex and NonlinParam
    #-------------------------------
    ZIndex1 = ones(Int64, nfa)
    NonlinParam1 = zeros(nfa, npt)
    
    #-------------------------------------
    # Initialize ZIndex1 and NonlinParam1
    #-------------------------------------
    ZIndex1[1:cbs] = ZIndex[1:cbs]
    NonlinParam1[1:cbs, 1:npt] = NonlinParam[1:cbs, 1:npt]

    #-------------------------------------
    # If action item is of type basis_enl
    #-------------------------------------
    if it.Type == basis_enl

        #------------------------
        # Update param structure
        #------------------------
        param.nfru = cbs
        param.Nmin = cbs+1
        param.Nmax = nfa
        param.cbs = nfa
        param.H = H1
        param.diagH = diagH1
        param.S = S1
        param.diagS = diagS1
        param.D = D1
        param.ZIndex = ZIndex1
        param.NonlinParam = NonlinParam1
        param.LastEigvector = ones(Float64, nfa)
        
        #------------------------------------------------------------------
        # Call stoch_nlp to randomly select functions from Kstart to Kstop
        #------------------------------------------------------------------
        ntrials = it.ntrials
        for k in 1:ntrials
            if verbose1
                println("\nTrial: $k out of $ntrials")
                println(trace_f, "\nTrial: $k out of $ntrials")
            end
        
            ok, neval = stoch_nlp(Kstart=Kstart, Kstop=Kstop, MaxEnergyEval=MaxEnergyEval, seed=it.seed, verbose1=verbose1, verbose2=verbose2)

            if !ok
                break
            end
        end

    else
        #------------------------------------------------------------------------------
        # Action type is assumed to be basis_enl_F90
        # Create a new action list for Fortran program and push an action item into it
        #------------------------------------------------------------------------------
        nfa = max(it.nfa, it.Kstart)
        
        action_list_F90 = []
        action_item_F90 = action(Type=it.Type, solver_type=GSEPSolutionMethod, nfa=nfa, nfo=1, ntrials=it.ntrials,
            MaxEnergyEval=it.MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=it.Kstep, seed=nothing)

        push!(action_list_F90, action_item_F90)

        #--------------------------
        # Write file inout_F90.txt
        #--------------------------
        write_inout(; param=param, inout_file="inout_F90.txt", action_list=action_list_F90)

        #------------------------------------------
        # Write non-linear parameters to text file 
        #------------------------------------------
        write_NonlinParam(param.cbs, param.npt, param.ZIndex, param.NonlinParam, ZIndex_used=param.ZIndex_used, verbose1=verbose1)

        #------------------------
        # Update param structure
        #------------------------
        param.nfru = cbs
        param.Nmin = cbs+1
        param.Nmax = nfa
        param.cbs = nfa
        param.H = H1
        param.diagH = diagH1
        param.S = S1
        param.diagS = diagS1
        param.D = D1
        param.ZIndex = ZIndex1
        param.NonlinParam = NonlinParam1
        param.LastEigvector = ones(Float64, nfa)
        
        #------------------
        # Call run_Fortran   
        #------------------
        ok, H_90, S_90 = run_Fortran(; param=param, verbose1=verbose1, verbose2=verbose2)
    end

    if ok
        #---------
        # Success 
        #---------
        # Add new history items to the list
        for k in Kstart:Kstop
            history_add(param.CurrEnergy, it.ntrials, Kstart, neval, it.seed)
        end
        
    else
        #----------------------------------------------------
        # Failure - Restore parameters to the previous state
        #----------------------------------------------------
        if verbose1 && it.Type == basis_enl
            println("\ndo_basis_enl - stoch_nlp() failed - Restoring H, S, non-linear basis set parameters")
            println(trace_f, "\ndo_basis_enl - stoch_nlp() failed - Restoring H, S, non-linear basis set parameters")
        end
        
        # Restore nfru, Nmin, Nmax, cbs, n, npt, H, diagH, S, diagS, D,  ZIndex, NonlinParam, LastEigvector
        param.nfru = nfru
        param.Nmin = Nmin
        param.Nmax = Nmax
        param.cbs = cbs
        param.n = n
        param.npt = npt
        param.H = H
        param.diagH = diagH
        param.S = S
        param.diagS = diagS
        param.D = D
        param.ZIndex = ZIndex
        param.NonlinParam = NonlinParam
        param.LastEigvector = LastEigvector
        
    end
    
    return ok
    
end # function do_basis_enl

#--------------------------------------------------------------------------------------------------------
# Define loss_nlp() that takes as input a basis element with index param.nfru, calls compute_and_solve()
# and returns param.CurrEnergy as loss (variational energy minimization).
#
# Energies below param0.TargetEnergy violate the variational bound (TargetEnergy
# is set at or slightly below the exact energy): they are spurious solutions of
# the generalized eigenvalue problem caused by a near-singular overlap matrix S
# (near-linearly-dependent basis functions). The loss is mirrored about
# TargetEnergy for such energies, so the optimizer is repelled from the
# collapse region instead of diving into it:
#     E >= TargetEnergy  →  loss = E
#     E <  TargetEnergy  →  loss = 2*TargetEnergy - E   (grows as E collapses)
# The mirror is continuous at E = TargetEnergy.
# (Before June 2026 the loss was abs(CurrEnergy - TargetEnergy), which pinned
# the energy to TargetEnergy regardless of the variational bound.)
#--------------------------------------------------------------------------------------------------------
function loss_nlp(x::Vector{Float64}; param::Param=param, verbose1=verbose1, verbose2=verbose2)
    param.NonlinParam[param.nfru,:] = x
    param.Nmin = 1
    ok = compute_and_solve(; param=param, verbose1=false, verbose2=false, print=false, H_name=nothing, S_name=nothing)

    E = param.CurrEnergy
    loss::Float64 = E >= param0.TargetEnergy ? E : 2.0 * param0.TargetEnergy - E

    return loss
end

#-------------------------------------------------------------------------------------------------------------------
# Analytic gradient of the optimization loss w.r.t. the nonlinear parameters of the function being optimized (nfru).
#
# Uses the raw (unnormalized) generalized eigenproblem Hr c = E Sr c with the S-normalized ground eigenvector c
# (E is congruence-invariant, so this equals the normalized loss energy). Type-generic in the matrix-element scalar:
# Float64 for the real matelems (N_Pvec) and ComplexF64 for the complex/Hermitian one (N_C_Pvec). With H,S Hermitian
# and only row/col i varying,
#     dE/dvechL_i = 2 Re( conj(c_i) Σ_l c_l (gH_l - E gS_l) ),  gH_l,gS_l = Σ_j coeffs[j] * d(H_il,S_il)/dvechL_i
# where gH_l,gS_l come from the matelem's Dk (requires an AD/correct-gradient matelem). For real H,S,c this reduces
# exactly to 2 c_i Σ_l c_l (gH_l - E gS_l). The active coeffs match loss_nlp (truncated during the BFGS search), and
# the collapse-guard sign is mirrored so this is exactly d(loss)/dx.
#
# Verified against central finite differences of loss_nlp: N_Pvec (real, N_test/grad_check.jl) and N_C_Pvec
# (complex, incl. a nonzero-B trial, N_C_Pvec_test/grad_check*.jl) both agree to ~1e-7. Passed to Optim only when
# param0.analytic_grad is true; otherwise Optim uses its finite-difference gradient (unchanged default).
#-------------------------------------------------------------------------------------------------------------------
function loss_grad!(G, x; param::Param=param, verbose1=verbose1)
    i   = param.nfru
    cbs = param.cbs
    np  = param.npt
    NYHY = param.NumYHYTerms
    param.NonlinParam[i, :] = x

    coeffs = (_TRUNCATE[] && length(_YHYCOEFF_TRUNC[]) == NYHY) ? _YHYCOEFF_TRUNC[] : vec(param.YHYCoeff)

    # Element type of the matrix elements: Float64 (real matelems) or ComplexF64 (N_C_Pvec). This makes the
    # assembly type-generic so it works for both: real symmetric and complex Hermitian H/S.
    htmp, _, _, _ = MatrixElements(1, 1, 1, false, false; param=param)
    T = typeof(htmp)

    # raw (unnormalized) H, S over the active operator (Hermitian: H[l,k] = conj(H[k,l]))
    Hr = zeros(T, cbs, cbs); Sr = zeros(T, cbs, cbs)
    for k in 1:cbs, l in 1:k
        h = zero(T); s = zero(T)
        for j in 1:NYHY
            coeffs[j] == 0 && continue
            hh, ss, _, _ = MatrixElements(k, l, j, false, false; param=param)
            h += coeffs[j]*hh; s += coeffs[j]*ss
        end
        Hr[k,l] = h; Hr[l,k] = conj(h); Sr[k,l] = s; Sr[l,k] = conj(s)
    end

    F  = eigen(Hermitian(Hr), Hermitian(Sr))     # real eigenvalues; S-normalized eigenvectors (c'Sr c = 1)
    gi = argmin(F.values); E = real(F.values[gi]); c = F.vectors[:, gi]

    # sv = Σ_l c[l] (gH_l - E gS_l), where gH_l, gS_l = Σ_j coeffs[j] d(H_il, S_il)/dvechL_i from the AD Dk.
    sv = zeros(T, np)
    for l in 1:cbs
        gHl = zeros(T, np); gSl = zeros(T, np)
        for j in 1:NYHY
            coeffs[j] == 0 && continue
            _, _, Dk, _ = MatrixElements(i, l, j, true, false; param=param)
            @views gHl .+= coeffs[j].*Dk[1:np]
            @views gSl .+= coeffs[j].*Dk[np+1:2np]
        end
        @. sv += c[l]*(gHl - E*gSl)
    end
    # dE/dvechL_i = 2 Re( conj(c_i) Σ_l c_l (gH_l - E gS_l) ). For real H/S this reduces to 2 c_i Σ_l c_l (gH_l - E gS_l).
    grad = 2 .* real.(conj(c[i]) .* sv)

    if E < param0.TargetEnergy                 # loss = 2*Target - E  ->  d(loss)/dx = -dE/dx
        grad .= .-grad
    end

    # all_real == "T" with the complex N_C_Pvec method: the imaginary (B) parameters are frozen at zero, so their
    # gradient must be zero too (defensive; the N_C_Pvec matelem already returns zero B-derivatives).
    if N_C_Pvec && all_real == "T"
        np1 = Int(param.n*(param.n+1)/2)
        grad[np1+1:np] .= 0.0
    end

    G .= grad
    return G
end

#----------------------------------------------------------
# Define optim_nlp() that optimizes a non linear basis set
#----------------------------------------------------------
using Optim
function optim_nlp(Kstart, Kstop, ncycles; param::Param=param, verbose1=verbose1)

    if verbose1
        println("\noptim_nlp - Optimizing non linear basis set")
        println(trace_f, "\noptim_nlp - Optimizing non linear basis set")
    end

    nfru = param.nfru

    # Wall-clock timer for the optimization phase (reported at the end of optim_nlp)
    t_optim = time()

    # Save current state
    save_state(; param=param, param1=param1)

    #-----------------------------------------------------------------------------------------------
    # Multi-fidelity setup: GUIDE the inner BFGS search with a truncated Y^{+}Y operator (terms with
    # |YHYCoeff| < threshold are zeroed and skipped in compute_H_S); every accepted/reported energy is
    # recomputed with the full operator (_TRUNCATE[] = false).
    #
    # coeff_threshold = 0 disables truncation (full operator throughout). When coeff_threshold > 0 and
    # coeff_ramp_tol > 0, an automatic ramp starts at coeff_threshold (coarse) and lowers the threshold
    # to the next-finer coefficient tier -- and finally to 0 (full) -- whenever a cycle's total energy
    # improvement falls below coeff_ramp_tol. With coeff_ramp_tol = 0 the threshold stays fixed.
    #-----------------------------------------------------------------------------------------------
    _TRUNCATE[] = false
    full_coeffs = vec(param.YHYCoeff)
    T0 = param0.coeff_threshold
    ramp_tol = param0.coeff_ramp_tol

    # Ramp schedule of thresholds (coarse -> fine), used only when T0 > 0:
    # start at T0, step down through the coefficient tiers strictly below T0, end at 0 (full operator).
    ramp = Float64[]
    if T0 > 0
        tiers = sort(unique(abs.(full_coeffs)))     # ascending distinct |coeff| values
        mintier = tiers[1]
        push!(ramp, float(T0))
        for t in sort(tiers, rev=true)
            (t < T0 && t > mintier) && push!(ramp, float(t))
        end
        push!(ramp, 0.0)
        unique!(ramp)
    end

    stage = 1
    if T0 > 0
        nkept = _set_trunc!(ramp[stage], full_coeffs)
        if verbose1
            println("\noptim_nlp - coeff_threshold = $(ramp[stage]): guiding the BFGS search with $nkept of $(length(full_coeffs)) symmetry terms (accepted energies use the full operator)")
            println(trace_f, "\noptim_nlp - coeff_threshold = $(ramp[stage]): guiding the BFGS search with $nkept of $(length(full_coeffs)) symmetry terms (accepted energies use the full operator)")
            if ramp_tol > 0
                println("optim_nlp - threshold ramp enabled (coeff_ramp_tol = $ramp_tol): schedule $ramp")
                println(trace_f, "optim_nlp - threshold ramp enabled (coeff_ramp_tol = $ramp_tol): schedule $ramp")
            end
        end
    else
        _YHYCOEFF_TRUNC[] = Int64[]
    end

    # Compute initial loss (full operator)
    param.nfru = 1
    loss = loss_nlp(param.NonlinParam[param.nfru,:]; param=param, verbose1=verbose1)

    #-------------------------------------------------------------------------------------------------
    # Decide whether to use the analytic gradient. It requires a matelem that returns a vector Dk;
    # probe the active matelem once. If analytic_grad is requested but the matelem returns a scalar
    # (e.g. the complex N_C_Pvec matelem, which has no AD gradient yet), fall back to Optim's finite
    # differences with a warning instead of crashing in loss_grad!.
    #-------------------------------------------------------------------------------------------------
    use_analytic = param0.analytic_grad
    if param0.analytic_grad
        _, _, Dk_probe, _ = MatrixElements(Kstart, Kstart, 1, true, false; param=param, verbose1=false, verbose2=false)
        if !(Dk_probe isa AbstractVector)
            use_analytic = false
            println("\noptim_nlp - WARNING: analytic_grad=true but MatElem_method=$MatElem_method does not return analytic Dk; using finite differences instead.")
            println(trace_f, "\noptim_nlp - WARNING: analytic_grad=true but MatElem_method=$MatElem_method does not return analytic Dk; using finite differences instead.")
        end
    end

    #---------------------------------------------------------------------------------------------
    # Perform an optimization cycle of functions in the non-linear basis set from Kstart to Kstop
    #
    # Use Optim.jl, Univariate and multivariate optimization in Julia.
    # https://julianlsolvers.github.io/Optim.jl/stable/
    #
    # Use finite differences for BFGS
    # https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/
    #
    # Optim general options
    # https://julianlsolvers.github.io/Optim.jl/stable/user/config/
    #----------------------------------------------------------------------------------------------
    for u in 1:ncycles
        if verbose1
            println("\noptim_nlp - Optimization cycle: $u out of $ncycles")
            println(trace_f, "\noptim_nlp - Optimization cycle: $u out of $ncycles")
        end

        # Table header is printed lazily, before the first improvement row,
        # so cycles without any improvement stay quiet.
        header_printed = false

        # Total energy improvement accumulated this cycle (drives the threshold ramp).
        cycle_improvement = 0.0

        for i in Kstart:Kstop

            param.nfru = i
            _TRUNCATE[] = true    # cheap, partially symmetrized surface for the BFGS search only
            opts = Optim.Options(iterations = param0.optim_iter, show_warnings = true, time_limit = param0.optim_time_limit)
            if use_analytic
                result = optimize(loss_nlp, loss_grad!, param.NonlinParam[i,:], BFGS(), opts)   # analytic gradient
            else
                result = optimize(loss_nlp, param.NonlinParam[i,:], BFGS(), opts)                # Optim finite differences
            end
            param.NonlinParam[i,:] = result.minimizer
            _TRUNCATE[] = false   # accepted / reported energies use the full operator

            # Compute loss (full operator) for the accept/reject decision and the reported energy
            loss_ = loss_nlp(param.NonlinParam[i,:]; param=param, verbose1=verbose1)

            # Accept only genuine variational improvements: the energy must not
            # fall below TargetEnergy (the physical lower bound) — energies
            # below it come from a near-singular overlap matrix S.
            if loss_ < loss && param.CurrEnergy >= param0.TargetEnergy
                improvement = loss - loss_
                cycle_improvement += improvement
                loss = loss_
                # Save state
                save_state(; param=param, param1=param1)

                # Print a table row on improvement, occasionally (first 10 functions or every 10th)
                if verbose1 && (i - Kstart <= 10 || i % 10 == 0)
                    if !header_printed
                        @printf("%25s   %14s   %14s\n", "Optimized function number", "Current Energy", "Improvement")
                        @printf(trace_f, "%25s   %14s   %14s\n", "Optimized function number", "Current Energy", "Improvement")
                        header_printed = true
                    end
                    @printf("%25d   %14.6f   %14.4e\n", i, real(param.CurrEnergy), improvement)
                    @printf(trace_f, "%25d   %14.6f   %14.4e\n", i, real(param.CurrEnergy), improvement)
                end
            else
                # Restore state
                restore_state(; param=param, param1=param1)
            end
        end

        #-------------------------------------------------------------------------------------------
        # Threshold ramp: when this cycle's total improvement falls below coeff_ramp_tol, lower the
        # threshold to the next-finer tier (more symmetry terms; eventually the full operator).
        #-------------------------------------------------------------------------------------------
        if ramp_tol > 0 && stage < length(ramp) && cycle_improvement < ramp_tol
            stage += 1
            nkept = _set_trunc!(ramp[stage], full_coeffs)
            if verbose1
                lbl = ramp[stage] > 0 ? "$nkept of $(length(full_coeffs)) symmetry terms" : "the full operator"
                @printf("optim_nlp - cycle improvement %.4e < coeff_ramp_tol %.4e: lowering threshold to %g (%s)\n",
                        cycle_improvement, ramp_tol, ramp[stage], lbl)
                @printf(trace_f, "optim_nlp - cycle improvement %.4e < coeff_ramp_tol %.4e: lowering threshold to %g (%s)\n",
                        cycle_improvement, ramp_tol, ramp[stage], lbl)
            end
        end
    end

    _TRUNCATE[] = false   # safety: never leave truncation on outside the BFGS search
    param.nfru = nfru

    if verbose1
        @printf("optim_nlp - wall-clock time: %.2f s\n", time()-t_optim)
        @printf(trace_f, "optim_nlp - wall-clock time: %.2f s\n", time()-t_optim)
    end

    return
end

#-----------------------------------------------------------
# Define do_opt_cycle() that performs an optimization cycle
#-----------------------------------------------------------
function do_opt_cycle(action_item; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    cbs = param.cbs
    it = action_item

    #------------------------------------------------------
    # Get Kstart, Kstop, MaxEnergyEval, nlp0 and coeff_nlp
    #------------------------------------------------------
    Kstart = it.Kstart
    Kstop = it.Kstop

    #---------------------------------------
    # Ensure Kstop is in range[Kstart, cbs]
    #---------------------------------------
    cbs = size(param.NonlinParam)[1]
    Kstop = min(Kstop, cbs)

    ncycles = it.ntrials
    MaxEnergyEval = it.MaxEnergyEval
    neval = MaxEnergyEval
    ntrials = it.ntrials
    
    param0.nlp0 = it.nlp0
    param0.coeff_nlp = it.coeff_nlp
    
    nfa = max(it.nfa, Kstart)

    ok = true

    #-------------------------------------
    # If action item is of type opt_cycle
    #-------------------------------------
    if it.Type == opt_cycle
        #---------------
        # Run optim_nlp
        #---------------
        optim_nlp(Kstart, Kstop, ncycles; param=param, verbose1=verbose1)
        history_update(param.CurrEnergy, ncycles, Kstart, neval, action_item.seed)
    end

    if it.Type == opt_cycle_F90 && do_Fortran
        #------------------------------------------------------------------------------
        # Create a new action list for Fortran program and push an action item into it
        #------------------------------------------------------------------------------
        action_list_F90 = []
        action_item_F90 = action(Type=opt_cycle, solver_type=GSEPSolutionMethod, nfa=nfa, nfo=1, ntrials=ntrials,
            MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=it.Kstep, seed=nothing)

        push!(action_list_F90, action_item_F90)

        #--------------------------
        # Write file inout_F90.txt
        #-------------------------
        write_inout(; param=param, inout_file="inout_F90.txt", action_list=action_list_F90)

        #------------------------------------------
        # Write non-linear parameters to text file 
        #------------------------------------------
        write_NonlinParam(param.cbs, param.npt, param.ZIndex, param.NonlinParam, ZIndex_used=param.ZIndex_used, verbose1=verbose1)

        #-------------------------------------------------
        # Run Fortran program that computes H_90 and S_90  
        #-------------------------------------------------
        ok, H_90, S_90 = run_Fortran(; param=param, verbose1=verbose1, verbose2=verbose2)

        if ok
            #------------------------------------------------
            # Success - Update last history item in the list
            #------------------------------------------------
            history_update(param.CurrEnergy, ncycles, Kstart, neval, action_item.seed)
        end
    end
    
    return ok
    
end # function do_opt_cycle

#----------------------------------------------------------------------------------------------------------
# Define do_check() that checks the accuracy by computing maximum(.abs((H - e*S)*v)) for each eigenvalue e
# and corresponding eigenvector v which should be close to zero.
#
# https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedEigen
# the eigenvalues can be obtained via F.values and the eigenvectors as the columns of the matrix F.vectors. 
# (The kth eigenvector can be obtained from the slice F.vectors[:, k].)
#-----------------------------------------------------------------------------------------------------------
function do_check(action_item; param::Param=param, verbose1=verbose1, verbose2=verbose2)

    @unpack H, S, GSEP_I, Evectors, Evalues, ApproxEnergy = param

    #---------------------------------------------
    # Filter eigenvalues that satisfy:
    # if GSEP_I, real(Evalues) + ApproxEnergy < 0
    # else satisfy real(Evalues) < 0
    #---------------------------------------------
    # https://docs.julialang.org/en/v1/base/collections/#Base.filter
    if GSEP_I
        X = filter(x->x<-ApproxEnergy, real(Evalues))
    else
        X = filter(x->x<0, real(Evalues))
    end

    len = min(length(X), max_print_H)

    if verbose1
        println("\nEvaluating the accuracy of the computed eigenvalues and eigenvectors")
        println(trace_f, "\nEvaluating the accuracy of the computed eigenvalues and eigenvectors")
        println("\nk   evalue e   maximum(.abs((H - e*S)*v))")
        println(trace_f, "\nk   evalue e   maximum(.abs((H - e*S)*v))")
    end

    for k in 1:len
        e = X[k]
        v = Evectors[:, k]
        u = (H - e*S)*v
        d = maximum(abs.(u))
        if verbose1
            s = @sprintf("%i   %.5f    %0.4e", k, e, d)
            println(s)
            println(trace_f, s)
        end
    end

    return true
    
end # function do_check

#---------------------------------------------
# Define do_action() that process action list
#---------------------------------------------
function do_action(; param::Param=param)

    ok = true

    # Wall-clock timer for the whole calculation (reported at the end of do_action)
    t_do_action = time()

    #-----------------------------
    # If data_init() failed, exit
    #-----------------------------
    if !param.data_init_ok
        if verbose1
            println("\nExiting since data_init() failed")
            println(trace_f, "\ndo_action - exiting since data_init() failed")
        end
        return false
    end
    
    #-----------------------------------------------------
    # Compute H and S matrices and solve secular equation
    #-----------------------------------------------------
    param.ApproxEnergy = param0.ApproxEnergy
    
    ok = compute_and_solve(; param=param, verbose1=verbose1, verbose2=verbose2)
    
    if !ok
        if verbose1
            println("\ndo_action - compute_and_solve() failed")
            println(trace_f, "\ndo_action - compute_and_solve() failed")
        end
        return false
    else
        if param.CurrEnergy < param0.TargetEnergy
            if verbose1
                println("\nCurrent Energy ", param.CurrEnergy, " is lower than target energy ", param0.TargetEnergy)
                println(trace_f, "\nCurrent Energy ", param.CurrEnergy, " is lower than target energy ", param0.TargetEnergy)
            end
            return false
        elseif verbose1
            println("\nUpdating last item in history with current energy: ", param.CurrEnergy)
            println(trace_f, "\ndo_action - updating last item in history with current energy: ", param.CurrEnergy)
        end
        history_update(param.CurrEnergy, 1, 1, 1, nothing; rank=param.cbs)
    end

    #----------------------------------------------------------------------------------------
    # Show current energy, maximum overlap between functions and distances between functions
    #----------------------------------------------------------------------------------------
    show_status(; param=param, verbose1=verbose1, verbose2=verbose2)
    
    #--------------------------------
    # Process actions in action_list
    #--------------------------------
    n_actions = size(action_list)[1]   # Get number of elements in vector action_list

    if n_actions == 0
        @printf("\ndo_action - total wall-clock time: %.2f s\n", time()-t_do_action)
        @printf(trace_f, "\ndo_action - total wall-clock time: %.2f s\n", time()-t_do_action)
        return ok
    end

    for i in 1:n_actions
        action_item = action_list[i]
        ok = true
        
        #------------------------------------------------
        # Replace set of non-linear basis set parameters 
        #------------------------------------------------
        if action_item.Type == basis_repl
            ok = do_basis_repl(action_item; param=param, verbose1=verbose1, verbose2=verbose2)
        
        #------------------------------------------------
        # Enlarge set of non-linear basis set parameters
        #------------------------------------------------
        elseif action_item.Type == basis_enl || action_item.Type == basis_enl_F90
            ok = do_basis_enl(action_item; param=param, verbose1=verbose1, verbose2=verbose2)

        #-------------------------------------------------
        # Optimize set of non-linear basis set parameters
        #-------------------------------------------------
        elseif action_item.Type == opt_cycle || action_item.Type == opt_cycle_F90
            ok = do_opt_cycle(action_item; param=param, verbose1=verbose1, verbose2=verbose2)

        #-----------------------------------------------------------------
        # Check the accuracy of the computed eigenvalues and eigenvectors
        #-----------------------------------------------------------------
        elseif action_item.Type == check
            ok = do_check(action_item; param=param, verbose1=verbose1, verbose2=verbose2)
        end
    end
    
    #---------------
    # Print history
    #---------------
    if history_size() > 1
        println("\nHistory list")
        println(trace_f, "\nHistory list")
        history_print()
    end

    #-------------------------------------------------
    # Plot the energy as a function of the basis size
    #-------------------------------------------------
    history_plot()

    #------------------------------------------------------------------------
    # Update ecg_config.json with the executed actions, ready for a follow-on
    # run. ECG_Config.write_config_actions! is a no-op when no config was read.
    #------------------------------------------------------------------------
    let parent = parentmodule(@__MODULE__)
        if isdefined(parent, :ECG_Config) && isdefined(parent.ECG_Config, :write_config_actions!) &&
           parent.ECG_Config.write_config_actions!(action_list)
            println("\ndo_action - updated ecg_config.json with $(length(action_list)) action(s)")
            println(trace_f, "\ndo_action - updated ecg_config.json with $(length(action_list)) action(s)")
        end
    end

    #------------------
    # Write inout file
    #------------------
    write_inout(; param=param, inout_file="inout_1.txt", action_list=action_list)

    #-----------------------------------
    # Report total wall-clock time
    #-----------------------------------
    @printf("\ndo_action - total wall-clock time: %.2f s\n", time()-t_do_action)
    @printf(trace_f, "\ndo_action - total wall-clock time: %.2f s\n", time()-t_do_action)

    return
end

#------------------------------------------------------------------------------------------------------------------------------
# init!() : run-time initialisation for module ECG. Selects the active matrix-element
# submodule from the model flags and constructs the param1 scratch instance with the
# matching element type. Must run AFTER ECG_Init.init!() has set the flags. Called at
# module-load for now (preserves current behaviour); Phase 3d moves this into run_ecg.
#------------------------------------------------------------------------------------------------------------------------------
function init!()
    global param1
    select_matelem!()
    param1 = ((CGL0 || N_C_Pvec) ? Param{ComplexF64} : Param{Float64})()
    return nothing
end # function init!

# init!() is called at run time by run_ecg (after ECG_Init.init!()), so this
# module precompiles cleanly without a configuration.

end # module ECG