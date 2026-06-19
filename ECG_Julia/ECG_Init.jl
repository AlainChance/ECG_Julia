#------------------------------------------------------------------------------------------------------------------------------
# Author: Alain Chancé
# Date: June 4, 2023
# Version: 1.0
#
# Module ECG_Init performs the following tasks:
#
# - Import a selection of variables from module ECG_Param
# - Create or open a trace file
# - Print parameters
# - Define global variables
# - Define constants
# - Generate Basis function prefactor indices, BFPI
#
# - Define the following functions:
#   - read_Param() that reads parameters from a file
#   - setup_nlp() that sets up trial nonlinear parameters for a given number of basis functions
#   - write_NonlinParam that writes non-linear parameters into file NonlinParam_file
#   - read_inout() that reads an inout.txt file using DelimitedFiles
#
# - Define mutable structure action
# - Define structure Param that is used to pass parameters using Parameters https://github.com/mauro3/Parameters.jl
# - Create an instance param of the Param structure
#
# - Define the following functions:
#   - reduced_Mass() that constructs the reduced mass matrix
#   - read_matrix_3D() that reads a 3D matrix from a file
#   - read_NonlinParam() that reads non-linear parameters from a file
#   - set_Transposit() that constructs a 4D matrix of all pair permutations
#   - set_Young() that sets up Young operators, list and count their independent terms
#   - data_init() that initializes data structures
#
# - Initialize data structures
# - Define set_parvec() that creates parvec matrix
#------------------------------------------------------------------------------------------------------------------------------
module ECG_Init

#------------------------------------------------------------------------------------------------------------------------------
# Import module ECG_Param parameters
#------------------------------------------------------------------------------------------------------------------------------
import ..ECG_Param: verbose, compute_H_S_method, MatElem_method, GSEPSolutionMethod, do_GSEPIIS, do_Fortran, outfile, inout_file
import ..ECG_Param: trace_file, param0, param_file, Mass_file, max_print_H, grad_k, grad_l, PseudoCharge_file, NonlinParam_file 
import ..ECG_Param: NonlinParam_db_file, _EPS, Mini, read_param_file, YOperatorStringLength, overlap_Skl, config_actions

#------------------------------------------------------------------------------------------------------------------------------
# Include module SymmetryOperators (symmetry_operators.jl) used by data_init()
# to compute the Y / Y†Y operator terms when the reference text files
# YCoeff.txt, YMatr.txt, YHYCoeff.txt, YHYMatr.txt are not present.
#------------------------------------------------------------------------------------------------------------------------------
include("symmetry_operators.jl")
import .SymmetryOperators

#------------------------------------------------------------------------------------------------------------------------------
# Export module ECG_Init parameters
#------------------------------------------------------------------------------------------------------------------------------
export trace_f, basis, N_Pvec, N_matr, N_C_Pvec, RGL0, RGL1, CGL0, verbose1, verbose2, verbose3
export Param, param, seed, ZERO, ONE, TWO, THREE, SIX, ONEHALF, ONETHIRD, ONEFOURTH, ONEFIFTH, ONESEVENTH, THREEHALF, PI, SQRTPI
export Mini, read_NonlinParam, write_NonlinParam, basis_repl, basis_enl, basis_enl_F90, opt_cycle, opt_cycle_F90, Full_opt1, check
export action, action_list, history, history_size, history_add, history_update, history_print, history_plot, read_inout, write_inout
export action_type, actions_to_config, actions_from_config

# Using Printf - https://docs.julialang.org/en/v1/stdlib/Printf/
using Printf

# Delimited Files, https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/
using DelimitedFiles

using Random

#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
# Module-level placeholders for configuration-dependent globals.
# These are populated at run time by init!() (defined at the end of this module),
# so that ECG_Init can be precompiled without a configuration and without an open
# trace file. The trace file is opened, and these values are computed, in init!().
#------------------------------------------------------------------------------------------------------------------------------
trace_f = devnull
seed = 0
basis = false
N_Pvec = false; N_matr = false; N_C_Pvec = false
RGL0 = false; RGL1 = false; CGL0 = false
verbose1 = false; verbose2 = false; verbose3 = false
npart = 0; n = 0; npt = 0; cbs = 0; NumYHYTerms = 0; ApproxEnergy = 0.0
param = nothing

# The symmetry (Young / Y†Y) operators are read from the YCoeff/YHYCoeff/YMatr/YHYMatr
# text files when present, otherwise built by module SymmetryOperators (symmetry_operators.jl).


#------------------------------------------------------------------------------------------------------------------------------
# Define numerical constants
# Ref. [Bubin], Fortran module globvars, https://github.com/sbubin/ATOM-MOL-nonBO/blob/master/src/globvars.f90
#------------------------------------------------------------------------------------------------------------------------------
const ZERO = 0.0::Float64
const ONE = 1.0::Float64
const TWO = 2.0::Float64
const THREE = 3.0::Float64
const FOUR = 4.0::Float64
const FIVE = 5.0::Float64
const SIX = 6.0::Float64
const SEVEN = 7.0::Float64
const EIGHT = 8.0::Float64
const NINE = 9.0::Float64
const TEN = 10.0::Float64
const ONEHALF = ONE/TWO::Float64
const ONETHIRD = ONE/THREE::Float64
const ONEFOURTH = ONE/FOUR::Float64
const THREEHALF = THREE/TWO::Float64
const ONEFIFTH = ONE/FIVE::Float64
const ONESEVENTH = ONE/SEVEN::Float64
const ONEEIGHTTH = ONE/EIGHT::Float64
const PI = 3.1415926535897932384626433832795029::Float64  
const SQRTPI = 1.7724538509055160272981674833411452::Float64
const FineStructConst = 7.2973525376E-03::Float64          #CODATA 2006 value

#------------------------------------------------------------------------------------------------------------------------------
# Generate the Basis Function Prefactor Indices (BFPI) array for atoms from Lithium (3 pseudoparticles) to Nitrogen (7).
#
# Each ECG basis function for these L=0, M=0 states carries an angular prefactor multiplying the correlated Gaussian; the
# prefactor is built from the coordinates (z-components) of THREE distinct pseudoparticles. BFPI is the catalogue of those
# index triples: row r holds a triple (i,j,k) of three distinct pseudoparticle indices, and a basis function's ZIndex selects
# which row (which prefactor) it uses. See Sharkey & Adamowicz, "An algorithm for nonrelativistic quantum-mechanical
# finite-nuclear-mass variational calculations of nitrogen atom in L = 0, M = 0 states using all-electrons explicitly
# correlated Gaussian basis functions", J. Chem. Phys. 140, 174112 (2014); https://doi.org/10.1063/1.4873916.
#
# The 36x3 table enumerates every triple of distinct indices drawn from {1,...,7} (C(7,3) = 35 triples; row 36 is spare),
# grouped by the LARGEST index in the triple (the pseudoparticle at which that index first becomes available):
#     largest = 3 -> row  1            (Lithium,   n = 3, 1 triple  = C(2,2))
#     largest = 4 -> rows 2-4          (Beryllium, n = 4, 3 triples = C(3,2))
#     largest = 5 -> rows 5-10         (Boron,     n = 5, 6 triples = C(4,2))
#     largest = 6 -> rows 11-20        (Carbon,    n = 6, 10 triples = C(5,2))
#     largest = 7 -> rows 21-35        (Nitrogen,  n = 7, 15 triples = C(6,2))
# Because the blocks are CUMULATIVE, an atom with n pseudoparticles may use every row whose three indices are <= n, i.e.
# rows 1 .. (end of its block): Lithium row 1 only, ..., Nitrogen all 35 rows. (So a valid Nitrogen ZIndex is any of 1..35,
# not only the 21-35 block, which is merely the subset of triples that contain index 7.)
#
# set_parvec() (this module) looks up the bra and ket triples via BFPI[ZIndex[k0],:] and BFPI[ZIndex[l0],:], permutes the ket
# triple through PP[:,j0], and assembles the 36-row parvec matrix that ECG_Matelem_N_Pvec / N_C_Pvec sum (with the covec
# signs) to evaluate the prefactor-weighted overlap, kinetic and potential matrix elements.
#------------------------------------------------------------------------------------------------------------------------------
BFPI = zeros(Int64, 36,3)

# Lithium
BFPI[1,1]=ONE
BFPI[1,2]=TWO
BFPI[1,3]=THREE
    
# Berylium
BFPI[2,1]=ONE
BFPI[2,2]=TWO
BFPI[2,3]=FOUR

BFPI[3,1]=ONE
BFPI[3,2]=THREE
BFPI[3,3]=FOUR

BFPI[4,1]=TWO
BFPI[4,2]=THREE
BFPI[4,3]=FOUR
    
# Boron
BFPI[5,1]=ONE
BFPI[5,2]=TWO
BFPI[5,3]=FIVE

BFPI[6,1]=ONE
BFPI[6,2]=THREE
BFPI[6,3]=FIVE

BFPI[7,1]=ONE
BFPI[7,2]=FOUR
BFPI[7,3]=FIVE

BFPI[8,1]=TWO
BFPI[8,2]=THREE
BFPI[8,3]=FIVE

BFPI[9,1]=TWO
BFPI[9,2]=FOUR
BFPI[9,3]=FIVE

BFPI[10,1]=THREE
BFPI[10,2]=FOUR
BFPI[10,3]=FIVE
    
# Carbon
BFPI[11,1]=ONE
BFPI[11,2]=TWO
BFPI[11,3]=SIX

BFPI[12,1]=ONE
BFPI[12,2]=THREE
BFPI[12,3]=SIX 

BFPI[13,1]=ONE
BFPI[13,2]=FOUR
BFPI[13,3]=SIX

BFPI[14,1]=ONE
BFPI[14,2]=FIVE
BFPI[14,3]=SIX

BFPI[15,1]=TWO
BFPI[15,2]=THREE
BFPI[15,3]=SIX

BFPI[16,1]=TWO
BFPI[16,2]=FOUR
BFPI[16,3]=SIX

BFPI[17,1]=TWO
BFPI[17,2]=FIVE
BFPI[17,3]=SIX

BFPI[18,1]=THREE
BFPI[18,2]=FOUR
BFPI[18,3]=SIX

BFPI[19,1]=THREE
BFPI[19,2]=FIVE
BFPI[19,3]=SIX

BFPI[20,1]=FOUR
BFPI[20,2]=FIVE
BFPI[20,3]=SIX

# Nitrogen
BFPI[21,1]=ONE
BFPI[21,2]=TWO
BFPI[21,3]=SEVEN

BFPI[22,1]=ONE
BFPI[22,2]=THREE
BFPI[22,3]=SEVEN

BFPI[23,1]=ONE
BFPI[23,2]=FOUR
BFPI[23,3]=SEVEN

BFPI[24,1]=ONE
BFPI[24,2]=FIVE
BFPI[24,3]=SEVEN

BFPI[25,1]=ONE
BFPI[25,2]=SIX
BFPI[25,3]=SEVEN

BFPI[26,1]=TWO
BFPI[26,2]=THREE
BFPI[26,3]=SEVEN

BFPI[27,1]=TWO
BFPI[27,2]=FOUR
BFPI[27,3]=SEVEN

BFPI[28,1]=TWO
BFPI[28,2]=FIVE
BFPI[28,3]=SEVEN

BFPI[29,1]=TWO
BFPI[29,2]=SIX
BFPI[29,3]=SEVEN

BFPI[30,1]=THREE
BFPI[30,2]=FOUR
BFPI[30,3]=SEVEN

BFPI[31,1]=THREE
BFPI[31,2]=FIVE
BFPI[31,3]=SEVEN

BFPI[32,1]=THREE
BFPI[32,2]=SIX
BFPI[32,3]=SEVEN

BFPI[33,1]=FOUR
BFPI[33,2]=FIVE
BFPI[33,3]=SEVEN

BFPI[34,1]=FOUR
BFPI[34,2]=SIX
BFPI[34,3]=SEVEN

BFPI[35,1]=FIVE
BFPI[35,2]=SIX
BFPI[35,3]=SEVEN

#----------------------------------------------------------
# Define read_Param() that reads parameters file Param.txt
#----------------------------------------------------------
function read_Param(; param_file=param_file, trace_f=trace_f)
    
    if isfile(param_file) && read_param_file
        if verbose1
            println("\nReading parameters from file: $param_file")
            println(trace_f, "\nRead_Param - Reading parameters from file: $param_file")
        end
        
        f = open(param_file)
        lines = readlines(f)
        
        npart = parse(Int64, lines[2])
        cbs = parse(Int64, lines[3])
        NumYHYTerms = parse(Int64, lines[4])
        ApproxEnergy = parse(Float64, lines[5])

        close(param_file)
        
    else
        if verbose1
            println("\nReading parameters from ECG_Param.param0")
            println(trace_f, "\nReading parameters from ECG_Param.param0")
        end
        
        npart = param0.npart
        cbs = param0.cbs
        NumYHYTerms = param0.NumYHYTerms
        
        param0.ApproxEnergy = max(param0.ApproxEnergy, param0.TargetEnergy)
        ApproxEnergy = param0.ApproxEnergy

        TargetEnergy = param0.TargetEnergy
    
        n::Int64 = npart-1
        np1::Int64 = Int(n*(n+1)/2)
    
        if CGL0 || N_C_Pvec
            npt = 2*np1
        else
            npt = np1
        end
    end
    
    if verbose1
        println("Number of particles, npart: $npart")
        println("Current basis size, cbs: $cbs")
        println("Number of terms in the simplified Y^{+}Y operator, NumYHYTerms: $NumYHYTerms")
        println("Approximate energy, ApproxEnergy: $ApproxEnergy")
        println("Target energy, TargetEnergy: $TargetEnergy")
        println("Number of non-linear parameters per basis function, npt: $npt")
        
        println(trace_f, "Number of particles, npart: $npart")
        println(trace_f, "Current basis size, cbs: $cbs")
        println(trace_f, "Number of terms in the simplified Y^{+}Y operator, NumYHYTerms: $NumYHYTerms")
        println(trace_f, "Approximate energy, ApproxEnergy: $ApproxEnergy")
        println(trace_f, "Target energy, TargetEnergy: $TargetEnergy")
        println(trace_f, "Number of non-linear parameters per basis function, npt: $npt")
    end
    
    return npart, n, npt, cbs, NumYHYTerms, ApproxEnergy
end

#---------------------------------
# Read parameters from param_file
#---------------------------------

#-----------------------------------------------------------------------------------------
# Define write_NonlinParam() that writes non-linear parameters into file NonlinParam_file
#-----------------------------------------------------------------------------------------
function write_NonlinParam(cbs, npt, ZIndex, NonlinParam; NonlinParam_file=NonlinParam_file, ZIndex_used=true, 
        verbose1=verbose1, trace_f=trace_f, mode="w")

    nline = size(NonlinParam,1)

    if cbs < 1 || nline != cbs
        if verbose1
            println("\nError - cbs: ", cbs, " is not equal to nline: ", nline, " the number of lines in NonlinParam file")
            println(trace_f, "\nError - cbs: ", cbs, " is not equal to nline: ", nline, " the number of lines in NonlinParam file")
        return false
        end
    else
        if verbose1
            println("\nWriting non-linear basis set parameters into file: $NonlinParam_file")
            println(trace_f, "\nWriting non-linear basis set parameters into file: $NonlinParam_file")
        end
    end

    NonlinParam_f = open(NonlinParam_file, mode) # Open NonlinParam file
                
    if ZIndex_used
        for i in 1:cbs
            s = string(i, " ", ZIndex[i])
            #s = @sprintf(" %.6i %.6i", i, ZIndex[i])
            for j in 1:npt
                s1 = @sprintf("%.16E", NonlinParam[i,j])
                s = string(s, " ", s1)
            end
            println(NonlinParam_f, s)
        end
    else
        for i in 1:cbs
            s = string(i)
            #s = @sprintf("%.6i", i)
            for j in 1:npt
                s1 = @sprintf("%.16E", NonlinParam[i,j])
                s = string(s, " ", s1)
            end
            println(NonlinParam_f, s)
        end
    end
        
    close(NonlinParam_f)

    return true
end

#-----------------------------------------------------------------------------------------------
# Define read_YHYCoeff that reads file YHYCoeff.txt and returns NumYHYTerms = size(YHYCoeff)[1]
#-----------------------------------------------------------------------------------------------
function read_YHYCoeff(; trace_f=trace_f)
    
    #--------------------------------------
    # Read YHYCoeff from file YHYCoeff.txt
    #--------------------------------------
    if isfile("YHYCoeff.txt")
        YHYCoeff = readdlm("YHYCoeff.txt", Int)
        NumYHYTerms = size(YHYCoeff)[1]
        if verbose1
            println("\nReading file YHYCoeff.txt - Set number of terms in the simplified Y^{+}Y operator, NumYHYTerms = size(YHYCoeff)[1]: $NumYHYTerms")
            println(trace_f, "\nReading file YHYCoeff.txt - Set number of terms in the simplified Y^{+}Y operator, NumYHYTerms = size(YHYCoeff)[1]: $NumYHYTerms")
        end
        if verbose2
            println(trace_f, "YHYCoeff: $YHYCoeff")
            println(trace_f, " ")
        end
        return true, NumYHYTerms
    else
        return false, 0
    end
end


#---------------------------------
# Define mutable structure action
#---------------------------------
const basis_enl::Int64 = 1
const opt_cycle::Int64 = 2
const full_opt1::Int64 = 3
const check::Int64 = 4
const basis_repl::Int64 = 5
const basis_enl_F90::Int64 = 6
const opt_cycle_F90::Int64 = 7

action_type = ["BASIS_ENL", "OPT_CYCLE", "FULL_OPT1", "CHECK", "BASIS_REPL", "BASIS_ENL_F90", "OPT_CYCLE_F90"]

Base.@kwdef mutable struct action
    Type::Int64 # basis_enl, opt_cycle or full_opt1
    solver_type::String # G or I
    nfa::Int64
    nfo::Int64
    ntrials::Int64
    MaxEnergyEval::Int64
    Kstart::Int64
    Kstop::Int64
    Kstep::Int64
    seed::Union{Int64, Nothing}
    nlp0::Bool = param0.nlp0
    coeff_nlp::Float64 = param0.coeff_nlp
end

#---------------------------
# Define vector action_list 
#---------------------------
action_list::Vector{action} = []

#-----------------------------------------------------
# Define action_add_check() that adds an action check
#-----------------------------------------------------
function action_add_check(; action_list::Vector{action}=action_list, verbose1=verbose1)

    add_check = false
        
    n_actions = size(action_list)[1]   # Get number of elements in vector action_list

    if n_actions > 0
        if action_list[n_actions].Type != check
            add_check = true
        end
    else
        add_check = true
    end

    if add_check
        action_item = action(Type=check, solver_type=GSEPSolutionMethod, nfa=1, nfo=1, ntrials=1, 
                MaxEnergyEval=1, Kstart=1, Kstop=1, Kstep=1, seed=nothing, nlp0=param0.nlp0, coeff_nlp=param0.coeff_nlp)
        push!(action_list, action_item)
        
        if verbose1
            println("\nAdded action check of the accuracy of the eigenvalues and eigenvectors")
            println(trace_f, "\ndo_action - Added action check of the accuracy of the eigenvalues and eigenvectors")
        end
    end

    return
end

#-------------------------------------------------------
# Define action_print() that prints the list of actions
#-------------------------------------------------------
# Using Printf - https://docs.julialang.org/en/v1/stdlib/Printf/
function action_print(; action_list::Vector{action}=action_list)
    
    n_actions = size(action_list)[1]   # Get number of elements in vector action_list
    
    if n_actions > 0
        
        println("\nAction list")
        println(trace_f, "\nAction list")
        
        for it in action_list
            s = " "
            if it.seed == nothing
                s_seed = " No seed"
            else
                s_seed = string(it.seed)
            end
            
            if it.Type == basis_repl
                s = @sprintf(" %s %s %i %i %i %i %s %i %.2f", action_type[it.Type], it.solver_type, it.Kstart, it.Kstop, it.ntrials, 
                    it.MaxEnergyEval, s_seed, it.nlp0, it.coeff_nlp)
            
            elseif it.Type == basis_enl || it.Type == basis_enl_F90
                s = @sprintf(" %s %s %i %i %i %i %.2f %.2f %s %i %.2f", action_type[it.Type], it.solver_type, it.nfa, it.nfo, 
                    it.ntrials, it.MaxEnergyEval, 0.98, 2.0, s_seed, it.nlp0, it.coeff_nlp)
                
            elseif it.Type == opt_cycle || it.Type == opt_cycle_F90
                s = @sprintf(" %s %s %i %i %i %i %i %i %.2f %.2f %i", action_type[it.Type], it.solver_type, it.Kstart, it.Kstop, 
                    it.nfo, 1, it.ntrials, it.MaxEnergyEval, 0.98, 2.0, 1)

            elseif it.Type == check
                s = @sprintf(" %s", action_type[it.Type])
                
            end
            
            println(s)
            println(trace_f, s)
        end
    end
    return
end

#------------------------------------------------------------------------------
# actions_to_config() / actions_from_config() — convert between the action_list
# (Vector{action}) and a JSON-friendly Vector of Dicts, used by the ecg_config.json
# "actions" section (an alternate to the legacy inout.txt command script).
# In the JSON form, "Type" is the action NAME (e.g. "BASIS_ENL", an entry of
# action_type) rather than the internal integer index, and "seed" is an integer
# or null (nothing).
#------------------------------------------------------------------------------
function actions_to_config(action_list::Vector{action}=action_list)
    return [Dict{String,Any}(
                "Type"          => action_type[a.Type],
                "solver_type"   => a.solver_type,
                "nfa"           => a.nfa,
                "nfo"           => a.nfo,
                "ntrials"       => a.ntrials,
                "MaxEnergyEval" => a.MaxEnergyEval,
                "Kstart"        => a.Kstart,
                "Kstop"         => a.Kstop,
                "Kstep"         => a.Kstep,
                "seed"          => a.seed,           # Int or nothing -> JSON null
                "nlp0"          => a.nlp0,
                "coeff_nlp"     => a.coeff_nlp,
            ) for a in action_list]
end

function actions_from_config(arr)
    action_list = Vector{action}()
    arr === nothing && return action_list
    for (i, d) in enumerate(arr)
        # "Type": accept the action name (preferred) or an integer index.
        tval = d["Type"]
        if tval isa Integer
            Type = Int(tval)
        else
            Type = findfirst(==(String(tval)), action_type)
            Type === nothing && error("actions_from_config: unknown action Type \"$tval\" in action $i (expected one of $action_type)")
        end
        # "seed": null/missing -> nothing, otherwise an integer.
        sval = get(d, "seed", nothing)
        seed = sval === nothing ? nothing : Int(sval)
        push!(action_list, action(
            Type          = Type,
            solver_type   = String(get(d, "solver_type", GSEPSolutionMethod)),
            nfa           = Int(get(d, "nfa", 0)),
            nfo           = Int(get(d, "nfo", 0)),
            ntrials       = Int(get(d, "ntrials", 0)),
            MaxEnergyEval = Int(get(d, "MaxEnergyEval", 0)),
            Kstart        = Int(get(d, "Kstart", 0)),
            Kstop         = Int(get(d, "Kstop", 0)),
            Kstep         = Int(get(d, "Kstep", 1)),
            seed          = seed,
            nlp0          = Bool(get(d, "nlp0", param0.nlp0)),
            coeff_nlp     = Float64(get(d, "coeff_nlp", param0.coeff_nlp)),
        ))
    end
    return action_list
end

#----------------------------------
# Define mutable structure history
#----------------------------------
Base.@kwdef mutable struct history
    energy::Float64              # energy at the end of the calculation
    ncycles::Int64               # Number of cycles done
    init::Int64                  # Initial function at last step
    neval::Int64                 # Number of energy evaluations
    seed::Union{Int64, Nothing}  # Random seed or nothing
end

#----------------------------
# Define vector history_list
#----------------------------
history_list::Vector{history} = []  # History list

#------------------------------------------------------------------------
# Define history_size() that returns the size of the vector history_list
#------------------------------------------------------------------------
function history_size(; history_list::Vector{history}=history_list)
    return size(history_list)[1]  # Return number of elements in vector history_list
end

#----------------------------------------------------------------
# Define history_add() that adds a new item to the history_list
#----------------------------------------------------------------
function history_add(energy::Float64, ncycles::Int64, init::Int64, neval::Int64, seed::Union{Int64, Nothing}; 
        history_list::Vector{history}=history_list)
    
    history_item = history(energy=energy, ncycles=ncycles, init=init, neval=neval, seed=seed)
    push!(history_list, history_item)
    
    return
end

#-------------------------------------------------------
# Define history_update() that updates the history_list
#-------------------------------------------------------
function history_update(energy::Float64, ncycles::Int64, init::Int64, neval::Int64, seed::Union{Int64, Nothing}; 
        history_list::Vector{history}=history_list, rank::Int64=0)

    n_history = size(history_list)[1]   # Get number of elements in vector history_list

    if rank <= 0
        # Update last element
        rank = n_history
    end
        
    if rank <= n_history
        # Update element
        history_item = history(energy=energy, ncycles=ncycles, init=init, neval=neval, seed=seed)
        history_list[rank] = history_item
        
    else
        # Add a new element
        history_add(energy, ncycles, init, neval, seed; history_list=history_list)
    end

    return
end

#-----------------------------------------------------
# Define history_print() that prints the history list
#-----------------------------------------------------
function history_print(; history_list::Vector{history}=history_list)
    
    n_history = size(history_list)[1]   # Get number of elements in vector history_list
    
    if n_history > 0
        i = 1
        for hc in history_list
            s = @sprintf(" %i %.16E %i %i %i", i, hc.energy, hc.ncycles, hc.init, hc.neval)
            println(s)
            println(trace_f, s)
            i += 1
        end
    end
    return
end

#-----------------------------------------------------------------------------
# Define history_plot() that plots the energy as a function of the basis size
#-----------------------------------------------------------------------------
using Plots # or StatsPlots
function history_plot(; history_list::Vector{history}=history_list)
    
    n_history = size(history_list)[1]   # Get number of elements in vector history_list
    Basis_size = [1:n_history]
    Energy = [history_list[i].energy for i in 1:n_history]
    
    if n_history > 0
        println("\nHistory plot")
        println(trace_f, "\nHistory plot")
        
        plot1 = plot(Basis_size, Energy, lw=3, label=string("Target energy ", param0.TargetEnergy), 
            title = string("Ground state energy ", param0.name), xlabel = ("Basis size"), ylabel = ("Energy"), 
            grid = gridlinewidth=1)
        
        println(" ")
        display(plot(plot1))
    end
    return
end

#--------------------------------------------------------------------------------------------------------------------------
# BASIS_ENL
# Enlarge the basis by stochastic selection of new basis functions followed by their optimization.
#
# Example: 
# BASIS_ENL I      6      7      1    500    200  0.98  0.25E+02
#
# Parameters:
#
# I - Eigenvalue solver type: I or G. I stands for an iterative solver based on the inverse iteration method which uses
# as approximate eigenvalue the product of CURRENT_ENERGY and INVITPARAMETER. Function GSEPIIS solves the secular equation 
# using the inverse iteration method.
#
# 6 - Current basis size, cbs.
#
# 7 - Target basis size, number of functions attempted, nfa.
#
# 1 - Number of functions to be randomly selected and added to the basis at each step, nfo.
#
# 500 - Number of random trials for the stochastic selection.
#
# 200 - Maximum number of energy evaluations in the optimization of the nonlinear parameters of the best new function 
# candidate that follows stochastic selection.
#
# 0.98 - Pair overlap threshold.
#
# 0.25E+02 - Threshold for the linear coefficients before normalized basis functions.
#--------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
# OPT_CYCLE
# Perform a cyclic optimization of the current basis (one or several functions at a time) a certain number of times.
#
# Example:
# OPT_CYCLE I      7      1      7      1      1     80    120  0.98  0.25E+02      3
#
# I - Eigenvalue solver type: I or G.
# 
# 7 - Current basis size
#
# 1 - Function number from which the optimization cycle should begin, Kstart.
#
# 7 - Function number at which the optimization cycle should end, Kstop.
#
# 1 - Number of functions whose parameters are to be optimized simultaneously, nfo.
#
# 1 - Number of functions to be shifted at each step of the process. Normally this is set to 1.
#
# 80 - Number of optimization cycles to be made.
#
# 120 - Maximum number of energy evaluations at each step of the process.
#
# 0.98 - Pair overlap threshold.
#
# 0.25E+02 - Threshold for the linear coefficients before normalized basis functions.
#
# 3 - Frequency of saving the updated basis in the input/output file.
#------------------------------------------------------------------------------------------------------------------------ 

#------------------------------------------------------------------------------------------------------------------------
# FULL_OPT1
# Perform full (i.e. simultaneous) optimization of all basis functions or a selected subset of basis functions..
#
# Example:
# FULL_OPT1 G      3      1      3 999999  0.9800000000000000E+00  0.3000000000000000E+01   3600   3600  hess.dat
#
# G - Eigenvalue solver type: I or G.
# 
# 3 - Current basis size
#
# 1 - Function number from which the full optimization cycle should begin, Kstart.
#
# 3 - Function number at which the full optimization cycle should end, Kstop.
#
# 999999 - Maximum number of energy evaluations at each step of the process.
#
# 0.98 - Pair overlap threshold.
#
# 0.30E+01 - Threshold for the linear coefficients before normalized basis functions.
#
# 3600 - Time interval in seconds for saving current best basis (e.g. 600 means every ten minutes) in the input/output file.
#------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------
# Define mutable structure Param that is used to pass parameters
#----------------------------------------------------------------
using Parameters

# Single parametric Param{T} (T = ComplexF64 for complex methods CGL0/N_C_Pvec,
# Float64 for the real methods). Replaces the old all_real-gated pair of structs so the
# type is fixed by a type parameter at construction, not by a runtime branch at definition.
@with_kw mutable struct Param{T<:Number}
    trace_f::IOStream = trace_f
    npart::Int64 = npart                            # Number of particles
    n::Int64 = n                                    # Number of pseudoparticles
    npt::Int64 = npt
    Mass::Vector{Float64} = zeros(Float64,npart)
    MassMatrix::Matrix{Float64} = zeros(n,n)
    PseudoCharge0::Float64 = 1
    PseudoCharge::Vector{Float64} = zeros(Float64, n)
    cbs::Int64 = cbs
    nfru::Int64 = 1
    Nmin::Int64 = 1                                  # MatrixElements() loops for k in Nmin:Nmax
    Nmax::Int64 = cbs
    ZIndex::Vector{Int64} = ones(Int64, cbs)         # Set ZIndex vector to all ones
    ZIndex_used::Bool = true
    NonlinParam::Matrix{Float64} = zeros(cbs, npt)
    ZIndex_db::Vector{Int64} = ones(Int64, cbs)      # Set ZIndex_db vector to all ones
    ZIndex_used_db::Bool = true
    NonlinParam_db::Matrix{Float64} = zeros(cbs, npt)
    YOperatorString::String = " "
    NumYTerms::Int = NumYHYTerms
    NumYHYTerms::Int = NumYHYTerms
    YCoeff::Matrix{Int64} = zeros(Int64,NumYTerms,1)
    YHYCoeff::Matrix{Int64} = zeros(Int64,NumYHYTerms,1)
    YMatr::Array{Float64} = zeros(n,n,NumYTerms)
    YHYMatr::Array{Float64} = zeros(n,n,NumYHYTerms)
    Transposit::Array{Int64,4} = zeros(Int64,n,n,npart,npart)
    PP::Array{Int64} = zeros(Int64,2*n,NumYHYTerms)
    covec::Vector{Int64} = zeros(Int64,36)
    Indentity::Vector{Int64} = zeros(Int64,2*n)
    H::Matrix{T} = zeros(T,cbs,cbs)
    diagH::Vector{Float64} = zeros(Float64,cbs)
    S::Matrix{T} = zeros(T,cbs,cbs)
    diagS::Vector{Float64} = zeros(Float64,cbs)
    D::Array{T} = zeros(T,2*npt,cbs,cbs)
    GSEP_G = (GSEPSolutionMethod == "G")                      # GSEP Solution method set in module ECG_Param
    GSEP_I = (GSEPSolutionMethod == "I")
    CurrEnergy::Float64 = ApproxEnergy
    WhichEigenvalue::Int64 = 1
    InvitParameter::Float64 = 1 + 1e-14
    ApproxEnergy::Float64 = ApproxEnergy
    EigvalTol::Float64 = _EPS
    LastEigvector::Vector{T} = ones(T, cbs)
    RG::Vector{Float64} = zeros(Float64,3)
    compute_H_S_method::String = compute_H_S_method           # Compute_H_S method set in module ECG_Param
    MatElem_method::String = MatElem_method                   # MatElem method set in module ECG_Param
    grad_k::Bool = grad_k
    grad_l::Bool = grad_l
    Evalue_GSEPIIS::Float64 = 0.0
    Evalue_eigen::Float64 = 0.0
    Evalues::Vector{T} = zeros(T,cbs)        # eigenvalues returned by the Julia eigen function
    Evectors::Matrix{T} = zeros(T,cbs,cbs)   # eigenvectors returned by the Julia eigen function
    condition::Float64 = 1.0
    data_init_ok::Bool = true
    div_by_zero::Bool = false
    n_div_by_zero::Int64 = 0
    nlp0::Bool = param0.nlp0
    do_setup_nlp::Bool = false
end

#----------------------------------------------------------------------------------
# Define read_NonlinParam() that reads non-linear basis set parameters from a file
#----------------------------------------------------------------------------------
function read_NonlinParam(; param::Param=param, NonlinParam_file=NonlinParam_file, verbose1=verbose1)

    npt = param.npt
    nptdiv2 = floor(Int64,npt/2)

    n_nlp = npt
    
    n = param.n
    np1::Int64 = Int(n*(n+1)/2)
    
    ZIndex_used = false
    
    if isfile(NonlinParam_file)

        m = readdlm(NonlinParam_file)
        
        basis_size = size(m)[1]
        size_m_2 = size(m)[2]
        
        if size_m_2 == npt + 1
            ZIndex = ones(Int64, basis_size)
            NonlinParam = m[:,2:end]
            n_nlp = npt
            
        elseif size_m_2 == npt + 2
            ZIndex_used = true
            ZIndex = Int64.(m[:,2])
            NonlinParam = m[:,3:end]
            n_nlp = npt

            if verbose1 && ZIndex_used
                println("\nZIndex is used")
                println(trace_f, "\nZIndex is used")
            end

        elseif (CGL0 || N_C_Pvec) && size_m_2 == nptdiv2 + 1
            ZIndex = ones(Int64, basis_size)
            NonlinParam = zeros(basis_size, npt)
            NonlinParam[:,1:nptdiv2] = m[:,2:end]
            n_nlp = nptdiv2

        elseif (CGL0 || N_C_Pvec) && size_m_2 == nptdiv2 + 2
            ZIndex_used = true
            ZIndex = Int64.(m[:,2])
            NonlinParam = zeros(basis_size, npt)
            NonlinParam[:,1:nptdiv2] = m[:,3:end]
            n_nlp = nptdiv2

            if verbose1 && ZIndex_used
                println("\nZIndex is used")
                println(trace_f, "\nZIndex is used")
            end
        
        else
            if verbose1
                println("\nNumber of non-linear basis set parameters per line in file ", NonlinParam_file, " is not the one expected, npt ", npt)
                println(trace_f, "\nNumber of non-linear basis set parameters per line in file ", NonlinParam_file, 
                    " is not the one expected, npt ", npt)
            end
            return false, nothing, nothing, nothing
            
        end

        if verbose1
            println("\nRead ", basis_size, " of ", n_nlp, " non-linear basis set parameters from file ", NonlinParam_file)
            println(trace_f, "\nRead ", basis_size, " of ", n_nlp, " non-linear basis set parameters from file ", NonlinParam_file)
        end
    
        if ZIndex_used && verbose1 && basis_size <= max_print_H
            println("\nZIndex")
            show(stdout, "text/plain", ZIndex)
            println(" ")

            println(trace_f, "\nZIndex")
            show(trace_f, "text/plain", ZIndex)
            println(trace_f, " ")
        end
        
        if verbose1 && basis_size <= max_print_H
            println("\nNonlinParam")
            show(stdout, "text/plain", NonlinParam)
            println(" ")

            println(trace_f, "\nNonlinParam")
            show(trace_f, "text/plain", NonlinParam)
            println(trace_f, " ")
        end

        return true, ZIndex, NonlinParam, ZIndex_used
        
    else
        if verbose1
            println("\nFile ", NonlinParam_file, " not found")
            println(trace_f, "\nFile ", NonlinParam_file, " not found")
        end

        return false, nothing, nothing, nothing
        
    end
end

#-------------------------------------------------------------------
# Read non-linear basis set parameters from file NonlinParam_db.txt
#-------------------------------------------------------------------
function read_NonlinParam_db(; param::Param=param, NonlinParam_db_file=NonlinParam_db_file, verbose1=verbose1)

    ok = true
    
    if isfile(NonlinParam_db_file)
        ok, ZIndex_db, NonlinParam_db, ZIndex_used_db = read_NonlinParam(; param=param, NonlinParam_file=NonlinParam_db_file, verbose1=verbose1)
        
        if ok
            param.NonlinParam_db = NonlinParam_db
            param.ZIndex_db = ZIndex_db
            param.ZIndex_used_db = ZIndex_used_db
        end
    else
        param0.nlp0 = true
        
        if verbose1
            println("\nFile ", NonlinParam_db_file, " not found")
            println(trace_f, "\nread_NonlinParam_db - File ", NonlinParam_db_file, " not found")
        end
    end

    return ok
end

#--------------------------------------------------------------------------------------------------
# Define setup_nlp() that sets up trial nonlinear parameters for a given number of basis functions.
#
# Input
#    nfunc - number of basis functions
#    npt - Total number of nonlinear parameters per basis function
#
# Output
#
#    m(1:nfunc) - an 1D-array containing generated Z-indicies of the basis functions.
#    x(1:npt,1:nfunc) - a 2D-array containing generated nonlinear parameters of the basis functions.
#---------------------------------------------------------------------------------------------------
using Random

function setup_nlp(nfunc::Int64; param::Param=param, verbose1=verbose1)
    
    #--------------------
    # Exit if nfunc is 0
    #--------------------
    if nfunc == 0
        println("\nsetup_nlp - nfunc = 0 - Exiting")
        println(trace_f, "\nsetup_nlp - nfunc = 0 - Exiting")
        return nothing
    end

    cbs = nfunc
    size_db = size(param.NonlinParam_db)[1]
    npt = size(param.NonlinParam)[2]

    # Set-up template NonLinParam
    if param0.nlp0
        if verbose1
            if nfunc == 1
                println("\nRandomly selecting one basis function with seed: ", seed)
                println(trace_f, "\nsetup_nlp - Randomly one selecting one basis function with seed: ", seed)
            else
                println("\nRandomly selecting ", nfunc, " basis functions with seed: ", seed)
                println(trace_f, "\nsetup_nlp - Randomly selecting ", nfunc, " basis functions with seed: ", seed)
            end
        end
        NonlinParam_t = param.NonlinParam
        size_t = cbs
    else
        if verbose1
            println("\nSet-up non linear basis set parameters from NonlinParam_db")
            println(trace_f, "\nsetup_nlp - Set-up non linear basis set parameters from NonlinParam_db")
        end
        NonlinParam_t = param.NonlinParam_db
        size_t = size_db
    end
    
    # shift vector
    shift = 0.5.*ones(nfunc, npt)  

    if param0.shuffle_NonlinParam
        # Shuffle NonlinParam
        ix = shuffle(Vector(1:size_t))
    else
        ix = Vector(1:size_t)
    end

    Kstart = 1
    Kstop = nfunc
    
    # Set up items from Kstart to Kstop in NonlinParam
    for i in Kstart:Kstop
        x = rand(Float64,(nfunc, param.npt)) - shift
        r = rand(0.8:1.2)
        param.NonlinParam[i,:] = r.*(param0.coeff_nlp.*x[i-Kstart+1,:] + NonlinParam_t[ix[i],:])
    end

    # Copy ZIndex
    if param.ZIndex_used
        param.ZIndex = ones(Int64, nfunc)
    end
    
    return
    
end # function setup_nlp

#------------------------------------------------------------------------------------------------------------------------------
# Define read_inout() that reads an inout.txt file
# [Bubin] Sergiy Bubin and Ludwik Adamowicz, Computer program ATOM-MOL-nonBO for performing calculations of ground and excited 
# states of atoms and molecules without assuming the Born–Oppenheimer approximation using all-particle complex explicitly 
# correlated Gaussian functions. J. Chem. Phys. 152, 204102 (2020), 26 May 2020, https://doi.org/10.1063/1.5144268 
# GitHub https://github.com/sbubin/ATOM-MOL-nonBO/tree/master/src
# ATOM_MOL-nonBO, module workproc: https://github.com/sbubin/ATOM-MOL-nonBO/blob/master/src/workproc.f90
#-------------------------------------------------------------------------------------------------------------------------------
function read_inout(; inout_file=inout_file, NonlinParam_file=NonlinParam_file, Mass_file=Mass_file, trace_f=trace_f,
        PseudoCharge_file=PseudoCharge_file, history_list::Vector{history}=history_list, verbose1=verbose1)

    #------------------------------------------------
    # Create an instance of the Param data structure
    #------------------------------------------------
    param = ((CGL0 || N_C_Pvec) ? Param{ComplexF64} : Param{Float64})()

    #-----------------------
    # Create an action_list
    #-----------------------
    action_list::Vector{action} = []
    
    if !isfile(inout_file)
        println("\nRead_inout - Missing inout file: ", inout_file)
        return false, param, action_list
    end
    
    ok = true
    
    if verbose1
        println("\nRead_inout - Reading parameters from file: $inout_file")
        println(trace_f, "\nRead_inout - Reading parameters from file: $inout_file")
    end
        
    #---------------------------------------------------------------------------------------
    # Phase 1 - Create a dictionary with first item in line as key and line number as value
    # Open inout file, read it line by line, populate the dictionary 
    # Get number of particles, basis size and actions if any
    # Close the inout file
    #---------------------------------------------------------------------------------------
    f = open(inout_file)
        
    dict = Dict{String, Int32}()
    
    i = 1
    
    for x in eachline(f)
        z = split(x) # Julia split, https://docs.julialang.org/en/v1/base/strings/
        if isempty(z)
            # Count empty lines too: the dict line numbers index into the raw
            # readlines() array of Phase 2, so every line must be counted.
            # (Previously empty lines were skipped, shifting all subsequent
            # lookups by one — e.g. an inout.txt with a blank line in the
            # actions block made read_inout fail on the parameter rows.)
            i += 1
        end
        if !isempty(z)

            dict[z[1]] = i
            len_z = length(z)
            
            if verbose2
                println(i, " ", z[1])
            end

            i += 1

            #-----------
            # PARTICLES
            #-----------
            if z[1] == "PARTICLES"
                npart = parse(Int64,z[2])   # Number of particles
                if npart < 2
                    npart = param.npart
                    if verbose1
                        println("\nPARTICLES - Number of particles, npart: ", npart, " less than 2, set to: ", param.npart)
                        println(trace_f, "\nPARTICLES - Number of particles, npart: ", npart, " less than 2, set to: ", param.npart)
                    end
                else
                    if verbose1
                        println("\nPARTICLES - Number of particles, npart: ", npart)
                        println(trace_f, "\nPARTICLES - Number of particles, npart: ", npart)
                    end
                end
                param.npart = npart
            
                n = npart-1                 # Number of pseudoparticles
                param.n = n
                
                if verbose1
                    println("Number of pseudo particles, n: ", n)
                    println(trace_f, "Number of pseudo particles, n: ", n)
                end
            
                np1 = Int(n*(n+1)/2)
                
                if verbose1
                    println("Number of independent parameters in a symmetric matrix of size (", n,"x", n,"): ", np1)
                    println(trace_f, "Number of independent parameters in a symmetric matrix of size (", n,"x", n,"): ", np1)
                end
                
                # Total number of nonlinear parameters per basis function
                if CGL0 || N_C_Pvec
                    npt = 2*np1
                else
                    npt = np1
                end
                param.npt = npt
                
                if verbose1
                    println("Number of non-linear parameters per basis function, npt: ", npt)
                    println(trace_f, "Number of non-linear parameters per basis function, npt: ", npt)
                end
            end

            #------------
            # BASIS_SIZE
            #------------
            if z[1] == "BASIS_SIZE"
                cbs = parse(Int64, z[2])
                param.cbs = cbs
                
                if verbose1
                    println("\nBASIS_SIZE -  Current basis size, cbs: ", cbs)
                    println(trace_f, "\nBASIS_SIZE -  Current basis size, cbs: ", cbs)
                end

                if cbs == 0
                # Create an action item to replace initial set of non-linear parameters 
                    Kstop = param0.cbs
                    
                    action_item = action(Type=basis_repl, solver_type=GSEPSolutionMethod, nfa=Kstop, nfo=1, ntrials=param0.ntrials, 
                        MaxEnergyEval=param0.MaxEnergyEval, Kstart=1, Kstop=Kstop, Kstep=1, seed=seed, nlp0=param0.nlp0, coeff_nlp=param0.coeff_nlp)
                    push!(action_list, action_item)

                    if verbose1
                        println("Creating an action item to replace ", param0.cbs, " non-linear parameters")
                        println(trace_f, "Creating an action item to replace ", param0.cbs, " non-linear parameters")
                    end
                end
            end

            #-----------
            # BASIS_ENL
            #-----------
            if z[1] == "BASIS_ENL" || z[1] == "BASIS_ENL_F90"
                solver_type = z[2]
                Kstart = parse(Int64, z[3])
                nfa = parse(Int64, z[4])
                nfo = parse(Int64, z[5])
                ntrials = parse(Int64, z[6])
                MaxEnergyEval = parse(Int64, z[7])
                
                Kstop = nfa
                Kstep = nfo

                if len_z >= 10
                    seed1 = parse(Int64, z[10])
                else
                    seed1 = nothing
                end

                if len_z >= 11
                    param0.nlp0 = parse(Bool, z[11])
                end

                if len_z >= 12
                    param0.coeff_nlp = parse(Float64, z[12])
                end

                if z[1] == "BASIS_ENL"
                    action_item = action(Type=basis_enl, solver_type=GSEPSolutionMethod, nfa=nfa, nfo=nfo, ntrials=ntrials, 
                        MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=Kstep, seed=seed1, nlp0=param0.nlp0, 
                        coeff_nlp=param0.coeff_nlp)
                else
                    action_item = action(Type=basis_enl_F90, solver_type=GSEPSolutionMethod, nfa=nfa, nfo=nfo, ntrials=ntrials,
                        MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=Kstep, seed=nothing, nlp0=param0.nlp0, 
                        coeff_nlp=param0.coeff_nlp)
                end

                push!(action_list, action_item)
            end

            #-----------
            # OPT_CYCLE
            #-----------
            if z[1] == "OPT_CYCLE" || z[1] == "OPT_CYCLE_F90"
                Kstart = parse(Int64, z[4])
                Kstop = parse(Int64, z[5])
                nfo = parse(Int64, z[6])
                Kstep = nfo
                ntrials = parse(Int64, z[8])
                MaxEnergyEval = parse(Int64, z[9])

                if len_z >= 13
                    seed1 = parse(Int64, z[13])
                else
                    seed1 = nothing
                end

                if len_z >= 14
                    param0.nlp0 = parse(Bool, z[14])
                end

                if len_z >= 15
                    param0.coeff_nlp = parse(Float64, z[15])
                end

                if z[1] == "OPT_CYCLE"
                    action_item = action(Type=opt_cycle, solver_type=GSEPSolutionMethod, nfa=Kstart, nfo=nfo, ntrials=ntrials, 
                    MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=Kstep, seed=seed1, nlp0=param0.nlp0, 
                    coeff_nlp=param0.coeff_nlp)
                else
                    action_item = action(Type=opt_cycle_F90, solver_type=GSEPSolutionMethod, nfa=Kstart, nfo=nfo, ntrials=ntrials, 
                    MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=Kstep, seed=seed1, nlp0=param0.nlp0, 
                    coeff_nlp=param0.coeff_nlp)
                end

                push!(action_list, action_item)
            end

            #-----------
            # FULL_OPT1
            #-----------
            if z[1] == "FULL_OPT1"
                Kstart = parse(Int64, z[4])
                Kstop = parse(Int64, z[5])
                nfo = Kstop - Kstart
                Kstep = nfo
                MaxEnergyEval = parse(Int64, z[6])
                ntrials = 1

                if len_z >= 13
                    seed1 = parse(Int64, z[13])
                else
                    seed1 = nothing
                end

                if len_z >= 14
                    param0.nlp0 = parse(Bool, z[14])
                end

                if len_z >= 15
                    param0.coeff_nlp = parse(Float64, z[15])
                end
            
                action_item = action(Type=full_opt1, solver_type=GSEPSolutionMethod, nfa=Kstart, nfo=nfo, ntrials=ntrials, 
                    MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=Kstep, seed=seed1, nlp0=param0.nlp0, 
                    coeff_nlp=param0.coeff_nlp)

                push!(action_list, action_item)
            end

            #-------
            # CHECK
            #-------
            if z[1] == "CHECK"
                action_item = action(Type=check, solver_type=GSEPSolutionMethod, nfa=1, nfo=1, ntrials=1, 
                    MaxEnergyEval=1, Kstart=1, Kstop=1, Kstep=1, seed=nothing, nlp0=param0.nlp0, 
                    coeff_nlp=param0.coeff_nlp)

                push!(action_list, action_item)
            end
                
            #-------------------
            # BASIS_REPL
            # Basis replacement
            #-------------------
            if z[1] == "BASIS_REPL"
                solver_type = z[2]
                Kstart = parse(Int64, z[3])
                nfa = parse(Int64, z[4])
                nfo = parse(Int64, z[5])
                ntrials = parse(Int64, z[6])
                MaxEnergyEval = parse(Int64, z[7])
                
                Kstop = nfa
                Kstep = nfo

                if len_z >= 10
                    seed1 = parse(Int64, z[10])
                else
                    seed1 = nothing
                end

                if len_z >= 11
                    param0.nlp0 = parse(Bool, z[11])
                end

                if len_z >= 12
                    param0.coeff_nlp = parse(Float64, z[12])
                end

                action_item = action(Type=basis_repl, solver_type=GSEPSolutionMethod, nfa=nfa, nfo=nfo, ntrials=ntrials, 
                    MaxEnergyEval=MaxEnergyEval, Kstart=Kstart, Kstop=Kstop, Kstep=Kstep, seed=seed1, nlp0=param0.nlp0, 
                    coeff_nlp=param0.coeff_nlp)

                push!(action_list, action_item)
            end

            #------------------------------
            # History current or list item
            #------------------------------
            if z[1] != "MASSES" && z[1] != "CHARGES" && (len_z == 5 || len_z == 6) && length(z[1]) < 5

                rank = parse(Int64, z[1])
                
                z[2] = replace(z[2], "D"=>"E")
                energy = parse(Float64, z[2])
                
                ncycles = parse(Int64, z[3])
                init = parse(Int64, z[4])
                neval = parse(Int64, z[5])
                
                if len_z == 6
                    seed1 = z[6]
                else
                    seed1 = nothing
                end
                
                if rank <= size(history_list)[1] + 1
                    history_add(energy, ncycles, init, neval, seed1; history_list=history_list)
                end
            end
        end
    end
        
    if verbose2
        println(trace_f, dict)
        println(trace_f, " ")
    end
        
    close(f)
        
    #--------------------------------------------------------------------------------------------------------
    # Phase 2 - Re-open inout_file, read all lines at once and leverage the dictionary to access its content 
    #--------------------------------------------------------------------------------------------------------
    f = open(inout_file)
    lines = readlines(f)

    #-----------------------------------------------------
    # Retrieve npart, n, npt and cbs from param structure
    #-----------------------------------------------------
    npart = param.npart
    n = param.n
    np1 = Int(n*(n+1)/2)
    npt = param.npt
    cbs = param.cbs
        
    #-----------
    # PARTICLES
    #-----------
    v = get(dict, "PARTICLES", nothing)
    if v == nothing
        ok = false
        if verbose1
            println("PARTICLES missing")
            println(trace_f, "PARTICLES missing")
        end
    end
        
    #--------
    # MASSES
    #--------
    v = get(dict, "MASSES", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            
            Mass = zeros(Float64, npart)

            s = string(" MASSES ")
            
            for i in 1:npart
                z[i+1] = replace(z[i+1], "D"=>"E")
                Mass[i] = parse(Float64, z[i+1])

                s1 = @sprintf("%.16E", Mass[i])
                s = string(s, " ", s1)
            end
            
            param.Mass = Mass
            
            if verbose1
                println("\nMASSES")
                show(stdout, "text/plain", Mass)
                println(" ")

                println(trace_f, "\nMASSES")
                show(trace_f, "text/plain", Mass)
                println(trace_f, " ")
            end
        
            Mass_f = open(Mass_file,"w") # Open Mass file in write mode
            if verbose1
                println("\nWriting Mass data into file: ", Mass_file)
                println(trace_f, "\nWriting Mass data into file: ", Mass_file)
            end
            println(Mass_f, s)
            close(Mass_f)
        end
    else
        ok = false
        if verbose1
            println("MASSES missing")
            println(trace_f, "MASSES missing")
        end
    end
        
    #---------
    # CHARGES
    #---------
    v = get(dict, "CHARGES", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            
            if verbose1
                println("\nWriting PseudoCharge data into file: ", PseudoCharge_file)
                println(trace_f, "\nWriting PseudoCharge data into file: ", PseudoCharge_file)
            end

            PseudoCharge_f = open(PseudoCharge_file,"w") # Open PseudoCharge file in write mode

            z[2] = replace(z[2], "D"=>"E")
            PseudoCharge0 = parse(Float64, z[2])
            param.PseudoCharge0 = PseudoCharge0

            if verbose1
                println("\nCHARGES")
                println("PseudoCharge: ")
                println("1: ", PseudoCharge0)
            end
            
            s1 = @sprintf("%.16E", PseudoCharge0)
            s = string(" CHARGES ", s1)

            if verbose1
                println(trace_f, "\nCHARGES")
                println(trace_f, "PseudoCharge: ")
                println(trace_f, "1: ", PseudoCharge0)
            end
            
            PseudoCharge = zeros(Float64, n)
            param.PseudoCharge = PseudoCharge
                    
            for k in 2:n+1
                z[k+1] = replace(z[k+1], "D"=>"E")
                PseudoCharge[k-1] = parse(Float64, z[k+1])
                
                if verbose1
                    println(k, ": ", PseudoCharge[k-1])
                end

                s1 = @sprintf("%.16E", PseudoCharge[k-1])
                s = string(s, " ", s1)
            end
            
            println(PseudoCharge_f, s)
            
            close(PseudoCharge_f)
        end
    else
        ok = false
        if verbose1
            println("CHARGES missing")
            println(trace_f, "CHARGES missing")
        end
    end
    
    #----------
    # SYMMETRY
    #----------
    v = get(dict, "SYMMETRY", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            YOperatorString = strip(z[2])
            param.YOperatorString = YOperatorString
            if verbose1
                println("\nSYMMETRY - YOperatorString: ", YOperatorString)
                println(trace_f, "\nSYMMETRY - YOperatorString: ", YOperatorString)
            end
        end
    else
        ok = false
        if verbose1
            println("SYMMETRY missing")
            println(trace_f, "SYMMETRY missing")
        end
    end
        
    #------------
    # BASIS_SIZE
    #------------
    v = get(dict, "BASIS_SIZE", nothing)
    if v == nothing
        ok = false
        if verbose1
            println("BASIS_SIZE missing")
            println(trace_f, "BASIS_SIZE missing")
        end
    end
        
    #----------------
    # CURRENT_ENERGY
    #----------------
    CurrEnergy = 0.0
    
    v = get(dict, "CURRENT_ENERGY", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            z[2] = replace(z[2], "D"=>"E")
            CurrEnergy = parse(Float64, z[2])
            if verbose1
                println("\nCURRENT_ENERGY - CurrEnergy: ", CurrEnergy)
                println(trace_f, "\nCURRENT_ENERGY - CurrEnergy: ", CurrEnergy)
            end
        end
    else
        ok = false
        if verbose1
            println("CURRENT_ENERGY missing")
            println(trace_f, "CURRENT_ENERGY missing")
        end
    end
    
    param.CurrEnergy = min(CurrEnergy, 0.0)
        
    #-------------------------------------------------------------------------------------------------------------------
    # WHICH_EIGENVALUE
    # Defines which eigenvalue will be calculated. The eigenvalues are numbered in the ascending order. 
    # So eigenvalue #1 is the lowest one and corresponds to the ground state. The first excited state would be #2.
    # However, keep in mind that it is only possible to target eigenvalue #K when the basis is at least K functions. 
    # So, one must target a different (smaller) eigenvalue before the basis of at least K basis functions is generated.
    #-------------------------------------------------------------------------------------------------------------------
    WhichEigenvalue = 1
    
    v = get(dict, "WHICH_EIGENVALUE", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            # Set WhichEigenvalue to be the maximum of 1 and the required value in the inout file
            WhichEigenvalue = max(1, parse(Int64, z[2]))
            if verbose1
                println("\nWHICH_EIGENVALUE - WhichEigenvalue: ", WhichEigenvalue)
                println(trace_f, "\nWHICH_EIGENVALUE - WhichEigenvalue: ", WhichEigenvalue)
            end
        end
    end
    
    param.WhichEigenvalue = WhichEigenvalue
        
    #------------------
    # EIGVAL_TOLERANCE
    #------------------
    EigvalTol = _EPS
    
    v = get(dict, "EIGVAL_TOLERANCE", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            # Set EigvalTol to the maximum of _EPS and the required value in the inout file
            z[2] = replace(z[2], "D"=>"E") 
            EigvalTol = max(_EPS, parse(Float64, z[2]))
            if verbose1
                println("\nEIGVAL_TOLERANCE - EigvalTol: ", EigvalTol)
                println(trace_f, "\nEIGVAL_TOLERANCE - EigvalTol: ", EigvalTol)
            end
        end
    end
    
    param.EigvalTol = EigvalTol
        
    #---------------------------------------------------------------------------------------------------------------------
    # INVITPARAMETER
    # The inverse iteration eigensolver requires an approximate eigenvalue, which is taken to be equal to CURRENT_ENERGY 
    # multiplied by INVITPARAMETER
    #---------------------------------------------------------------------------------------------------------------------
    InvitParameter = 1.0 + 1e-14
    
    v = get(dict, "INVITPARAMETER", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            # Set InvitParameter to the maximum of _EPS and the required value in the inout file
            z[2] = replace(z[2], "D"=>"E")
            InvitParameter = max(_EPS, parse(Float64, z[2]))
            if verbose1
                println("\nINVITPARAMETER - InvitParameter: ", InvitParameter)
                println(trace_f, "\nINVITPARAMETER - InvitParameter: ", InvitParameter)
            end
         end
    end
    param.InvitParameter = InvitParameter
    
    if CurrEnergy < 0.0
        param.ApproxEnergy = max(CurrEnergy * InvitParameter, param0.TargetEnergy)
        if verbose1
            println("\nCurrEnergy * InvitParameter: ", CurrEnergy * InvitParameter)
            println("\nApproxEnergy: ", param.ApproxEnergy)

            println(trace_f, "\nCurrEnergy * InvitParameter: ", CurrEnergy * InvitParameter)
            println(trace_f, "\nApproxEnergy: ", param.ApproxEnergy)
        end
    end
    
    #---------------------------------------------------------------------------------------------------------------------
    # GENERATOR_PARAM
    # Parameters of the random generator that is used to generate new basis function candidates 
    #---------------------------------------------------------------------------------------------------------------------
    v = get(dict, "GENERATOR_PARAM", nothing)
    if v !== nothing
        x = lines[v]
        if x != 0
            z = split(x)
            for i in 1:length(z)-1
                z[1+i] = replace(z[1+i], "D"=>"E")
                param.RG[i] = parse(Float64, z[1+i])
            end
            if verbose1
                println("\nGENERATOR_PARAM - RG: ", param.RG)
                println(trace_f, "\nGENERATOR_PARAM - RG: ", param.RG)
            end
         end
    end
        
    #---------------------------------
    # Non-linear basis set parameters
    #---------------------------------
    if cbs > 0
        # Create ZIndex and NonlinParam data structures
        ZIndex_used = false
        ZIndex = ones(Int64, cbs)         # Set ZIndex vector to all ones
        NonlinParam = zeros(cbs, npt)
        
        for i in 1:cbs
            v = get(dict, string(i), nothing)
            if v !== nothing
                x = lines[v]
                if x != 0
                    z = split(x)
                    
                    # If CGL0 or N_C_Pvec and the number of elements is half of what is expected 
                    if (CGL0 || N_C_Pvec) && (length(z) == np1+1 || length(z) == np1+2) 
                        # Reset npt to actual number of elements in a line
                        npt = np1
                        param.npt = np1
                        if i == 1 && verbose1
                            println("\nResetting number of non-linear parameters per basis function: ", np1)
                            println(trace_f, "\nResetting number of non-linear parameters per basis function: ", np1)
                        end
                    # If the number of elements in the first line is consistent with the expected number npt
                    elseif length(z) == npt+1 || length(z) == npt+2
                        if length(z) == npt+2
                            ZIndex_used = true
                        end
                    # Invalid number of non-linear basis set parameters
                    else
                        ok = false
                        if verbose1
                            println("\nInvalid number of non-linear parameters per basis function: ", length(z) - 1)
                            println(trace_f, "\nInvalid number of non-linear parameters per basis function: ", length(z) - 1)
                        end
                        break
                    end  
                    
                    if ZIndex_used
                        z[2] = replace(z[2], "D"=>"E")
                        ZIndex[parse(Int64,z[1])] = parse(Int64,z[2])
                        param.ZIndex = ZIndex
                        
                        for j in 1:npt
                            z[j+2] = replace(z[j+2], "D"=>"E")
                            fl = parse(Float64,z[j+2])
                            NonlinParam[parse(Int64,z[1]),j]=fl
                        end
                    else
                        for j in 1:npt
                            z[j+1] = replace(z[j+1], "D"=>"E")
                            fl = parse(Float64,z[j+1])
                            NonlinParam[parse(Int64,z[1]),j]=fl
                        end
                    end
                end
            end
        end

        param.ZIndex_used = ZIndex_used
        param.ZIndex = ZIndex
        param.NonlinParam = NonlinParam
    
    else  # Current basis size, cbs = 0
        cbs = max(param0.cbs, 3)  # Set new basis size to be a minimum of 3
        param.cbs = cbs

        if verbose1
            println("\nSetting up basis set of size cbs = ", cbs, " with trial non-linear basis set parameters")
            println(trace_f, "\nSetting up basis set of size cbs = ", cbs, " with trial non-linear basis set parameters")
        end

        if N_Pvec || N_matr || RGL1
            param.ZIndex_used = true  # ZIndex is needed for MatElem_method N_Pvec, N_matr or RGL1
        else
            param.ZIndex_used = false
        end

        param.NonlinParam = zeros(cbs, npt)
        param.ZIndex = ones(Int64, cbs)         # Set ZIndex vector to all ones

        # Add new history items to the list
        for k in 1:cbs
            history_add(CurrEnergy, 1, 1, 1, param0.seed)
        end

        # Set flag do_setup_nlp to true
        param.do_setup_nlp = true
    end

    # Write non-linear basis set parameters into file NonlinParam_file
    if ok
        write_NonlinParam(cbs, npt, param.ZIndex, param.NonlinParam; NonlinParam_file=NonlinParam_file, ZIndex_used=param.ZIndex_used, verbose1=verbose1)
    end
    
    if ok && verbose1 && cbs <= max_print_H
        if param.ZIndex_used
            println("\nZIndex is used")
            println("\nZIndex")
            show(stdout, "text/plain", param.ZIndex)
            println(" ")
        end
        println("\nNonlinParam")
        show(stdout, "text/plain", param.NonlinParam)
        println(" ")
    end
    
    close(f)
    
    # Fill in param structure
    param.YOperatorString = YOperatorString
    param.Nmax = param.cbs
    param.LastEigvector = ones(Float64, param.cbs)

    #-------------------------------------------------
    # Add an action check at the end if there is none
    #-------------------------------------------------
    action_add_check(; action_list=action_list, verbose1=verbose1)
    
    #-------------------
    # Print action list
    #-------------------
    if verbose1
        action_print(; action_list=action_list)
    end

    #-----------------------
    # Print history current
    #-----------------------
    n_history = size(history_list)[1]   # Get number of elements in vector history_list
    
    if verbose1 && n_history != 0
        hc = last(history_list)             # Get last element in history_list
        s = @sprintf(" %i %.16E %i %i %i", n_history, hc.energy, hc.ncycles , hc.init, hc.neval)
        
        println("\nHistory current")
        println(s)
        
        println(trace_f, "\nHistory current")
        println(trace_f, s)

        println("\nHistory")
        println(trace_f, "\nHistory")
        history_print(; history_list=history_list)
    end
                
    return ok, param, action_list

end

#----------------------------------------------------------------------------------------------------------------------------
# Define write_inout() that writes an inout.txt file
# Check Ref. [Bubin] and the Fortran module workproc: https://github.com/sbubin/ATOM-MOL-nonBO/blob/master/src/workproc.f90
#----------------------------------------------------------------------------------------------------------------------------
function write_inout(; param::Param=param, inout_file="inout_F90.txt", action_list=action_list_F90, history_list=history_list, verbose1=verbose1)

    if verbose1
        println("\nWriting inout file ", inout_file)
        println(trace_f, "\nWriting inout file ", inout_file)
    end

    cbs = param.cbs
    npt = param.npt
    ZIndex = param.ZIndex
    NonlinParam = param.NonlinParam
    ZIndex_used = param.ZIndex_used

    line_sep = " =============================="

    inout_f = open(inout_file,"w") # Open NonlinParam file in write mode
    
    println(inout_f, " PARTICLES ", param.npart)

    s = " MASSES"
    for mass in param.Mass
        s1 = @sprintf(" %.16E", mass)
        s = string(s, s1)
    end
    println(inout_f, s)

    s = @sprintf(" CHARGES %.16E", param.PseudoCharge0)
    for charge in param.PseudoCharge
        s1 = @sprintf("%.16E", charge)
        s = string(s, " ", s1)
    end
    println(inout_f, s)

    println(inout_f, " SYMMETRY ", param.YOperatorString)
    println(inout_f, " BASIS_SIZE ", param.cbs)

    s = @sprintf(" CURRENT_ENERGY %.16E", param.CurrEnergy)
    println(inout_f, s)
    
    println(inout_f, " WHICH_EIGENVALUE ", param.WhichEigenvalue)

    s = @sprintf(" EIGVAL_TOLERANCE %.16E", param.EigvalTol)
    println(inout_f, s)

    s = @sprintf(" INVITPARAMETER %.16E", param.InvitParameter)
    println(inout_f, s)

    s = @sprintf(" LAST_EIGVAL_TOL %.16E", param.EigvalTol)
    println(inout_f, s)

    s = @sprintf(" BEST_EIGVAL_TOL %.16E", param.EigvalTol)
    println(inout_f, s)

    s = @sprintf(" WORST_EIGVAL_TOL %.16E", param.EigvalTol)
    println(inout_f, s)

    s = " GENERATOR_PARAM"
    for rg in param.RG
        s1 = @sprintf("%.16E", rg)
        s = string(s, " ", s1)
    end
    println(inout_f, s)
    
    println(inout_f, line_sep)

    n_history = size(history_list)[1]   # Get number of elements in vector history_list
    if n_history != 0
        hc = last(history_list)         # Get last element in history_list
        s = @sprintf(" %i %.16E %i %i %i", n_history, hc.energy, hc.ncycles , hc.init, hc.neval)
        println(inout_f, s)
    end
    
    println(inout_f, line_sep)

    for it in action_list
        s = " "
        
        if it.seed == nothing
            s_seed = " No seed"
        else
            s_seed = string(it.seed)
        end
        
        if it.Type == basis_repl
            s = @sprintf(" %s %s %i %i %i %i %s %i %.2f", action_type[it.Type], GSEPSolutionMethod, it.Kstart, it.Kstop, it.ntrials, 
                it.MaxEnergyEval, s_seed, it.nlp0, it.coeff_nlp)
            
        elseif it.Type == basis_enl
            s = @sprintf(" %s %s %i %i %i %i %i %.2f %.2f %s %i %.2f", action_type[it.Type], GSEPSolutionMethod, param.cbs, it.nfa, it.nfo, 
                it.ntrials, it.MaxEnergyEval, 0.98, 2.0, s_seed, it.nlp0, it.coeff_nlp)

        elseif it.Type == basis_enl_F90
            s = @sprintf(" %s %s %i %i %i %i %i %.2f %.2f", action_type[basis_enl], GSEPSolutionMethod, param.cbs, it.nfa, it.nfo, 
                    it.ntrials, it.MaxEnergyEval, 0.98, 2.0)
                
        elseif it.Type == opt_cycle
            s = @sprintf(" %s %s %i %i %i %i %i %i %i %.2f %.2f %i", action_type[it.Type], GSEPSolutionMethod, param.cbs, it.Kstart, it.Kstop, 
                    it.nfo, 1, it.ntrials, it.MaxEnergyEval, 0.98, 2.0, 1)
        end
        
        println(inout_f, s)
    end

    println(inout_f, line_sep)

    i = 1
    for it in history_list
        s = @sprintf(" %i %.16E %i %i %i", i, it.energy, it.ncycles , it.init, it.neval)
        println(inout_f, s)
        i += 1
    end

    println(inout_f, line_sep)
    
    close(inout_f)    # Close inout_file

    write_NonlinParam(cbs, npt, ZIndex, NonlinParam; NonlinParam_file=inout_file, ZIndex_used=ZIndex_used, verbose1=false, trace_f=trace_f, mode="a")
    
    return
end

#---------------------------------------------------------------
# Define reduced_mass() that constructs the reduced mass matrix
#---------------------------------------------------------------
function reduced_mass(Mass, n; verbose1=verbose1, trace_f=trace_f)
    
    MassMatrix = zeros(n,n)
    
    if abs(Mass[1]) > Mini

        for i in 1:n
            for j in 1:n
                MassMatrix[i,j] = 0.5/Mass[1]
            end
        end
        
        for i in 1:n
            for j in 1:n
                if i==j
                    MassMatrix[i,j] = MassMatrix[i,j] + 0.5/Mass[i+1]
                end
            end
        end
        
        if verbose1
            println("\nReduced Mass matrix")
            show(stdout, "text/plain", MassMatrix)
            println(" ")
            
            println(trace_f, "\nMassMatrix - Reduced Mass matrix")
            show(trace_f, "text/plain", MassMatrix)
            println(" ")
        end
    
    else
        if verbose1
            println("Error Mass[1]: {:e} is too small", Mass[1])
            println(trace_f, "Error Mass[1]: {:e} is too small", Mass[1])
        end
    end
    
    return MassMatrix
end

#------------------------------------------------------------------------------------------------------------------------------
# Define read_matrix_3D() that reads a 3D matrix from a file
#------------------------------------------------------------------------------------------------------------------------------
function read_matrix_3D(; file=file, matrix=matrix, verbose1=verbose1, trace_f=trace_f)

    nline = size(matrix,2)
    nmatrix = size(matrix,3)
    
    m = 1 # matrix number
    i = 1 # matrix line

    f = open(file)
    for x in eachline(f)
        j = 1
        z = split(x) # Julia split, https://docs.julialang.org/en/v1/base/strings/
        
        for s in z
            matrix[i, j, m] = parse(Float64, s) # Fill current matrix line
            j += 1                  # Increment matrix column
        end
    
        i += 1
    
        if i == nline+1  # If all lines in matrix have been read
            m += 1  # Increment matrix number
            i = 1   # Reset matrix line
        end
    
        if m == nmatrix + 1
            break
        end
    end
    
    # Julia: How to pretty print an array?
    # https://stackoverflow.com/questions/62868864/julia-how-to-pretty-print-an-array
    if verbose1
        matrix_name = first(file,length(file)-4)
        if nmatrix > 1
            for i in [1, nmatrix-1, nmatrix]
                println(trace_f, matrix_name, "[:, :, ", i, "]")
                show(trace_f, "text/plain", matrix[:, :, i])
                println(trace_f, "\n")
            end
        else
            println(trace_f, matrix_name, "[:, :, ", 1, "]")
            show(trace_f, "text/plain", matrix[:, :, 1])
            println(trace_f, "\n")
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------
# Define set_Transposit() that constructs a 4D matrix of all pair permutations
# The 𝑃1𝑖(𝑖 ≠ 1) transposition matrix has the following form, where indices start from 1 (Appendix A, Transposition matrices 𝑃𝑖𝑗):
#
#    𝑃𝑘,k = 1 if 𝑘 ≠ 𝑖−1 
#    𝑃𝑘,𝑖−1 = −1
#    𝑃𝑘,𝑙 = 0 if 𝑘 ≠ 𝑙 and 𝑘 ≠ 𝑖−1
#
# The 4D array Transposit contains all pair permutation matrices (transpositions) and has the following structure:
#
# Transposit(0:n,0:n,1,2) corresponds to P12
# Transposit(0:n,0:n,5,5) corresponds to P55
#------------------------------------------------------------------------------------------------------------------------------
function set_Transposit(n, npart)

    Transposit = zeros(Int64,n,n,npart,npart)

    # First set all of them to be unit matrices nxn
    for i in 1:npart
      for j in 1:npart
        for k in 1:n
          Transposit[k,k,i,j]=1 
        end
      end
    end
        
    for i in 2:npart
        for k in 1:n
            Transposit[k,i-1,1,i]=-1
        end
    end

    for i in 2:npart
        for j in i+1:npart
            Transposit[i-1,i-1,i,j]=0
            Transposit[j-1,j-1,i,j]=0
            Transposit[j-1,i-1,i,j]=1
            Transposit[i-1,j-1,i,j]=1
            Transposit[i-1,i-1,j,i]=0
            Transposit[j-1,j-1,j,i]=0
            Transposit[j-1,i-1,j,i]=1
            Transposit[i-1,j-1,j,i]=1
        end
    end
    
    return Transposit
end

#-----------------------------------------------------------------------------------------
# Define set_Young() that sets up Young operators, list and count their independent terms
#-----------------------------------------------------------------------------------------
function set_Young(YOperatorString; param=param, verbose1=verbose1, verbose2=verbose2)

#------------------------------------------------------------
# Split Young operator string into a list of smaller strings
#------------------------------------------------------------
    StrLen = length(YOperatorString)
    
    s = SubString(YOperatorString, 2, StrLen-1)  # Remove leading "(" and trailing ")" 
    dlm = ")("
    YOpStr = split(s, dlm)                       # Split Young operator using ")(" as delimiter
    YOpStr = ["+"*s for s in YOpStr]             # Concatenate "+" with each item in YOpStr

    NumFactY=length(YOpStr)

    if verbose1
        println("\nYoung operator, YOperatorString: ", YOperatorString)
        println("Young operator factors, YOpStr: ", YOpStr)
        println("Number of factors in the Young operator, NumFactY: ", NumFactY)
    end
    
#-------------------------------------------------------------------------------------------------
# Create an array that contains all the factors of the $Y^{\dagger}$ operator
# $Y^{\dagger}$ is the reversed $Y$ (i.e. the order of all factors is reversed as well 
# as permutation products if any in each factor come in reverse order.
#-------------------------------------------------------------------------------------------------
    YHOpStr = reverse(YOpStr)

    if verbose1
        println("YHOpStr = reverse(YOpStr): ", YHOpStr)
    end
    
    NumTermsInYOpFact = zeros(Int32, length(YOpStr))
    NumTermsInYHOpFact = zeros(Int32, length(YHOpStr))

#----------------------------------------------------------------------------
# Count the number of terms in the Young operator and in each of its factors
#----------------------------------------------------------------------------
    NumYTerms = 1
    i = 1
    for s in YOpStr
        NumTermsInYOpFact[i] = count("+", s) + count("-", s)
        NumYTerms = NumYTerms * NumTermsInYOpFact[i]
        i += 1
    end

    if verbose1
        println("Number of independent terms in the Young operator, NumYTerms: ", NumYTerms)
    end
    
    if verbose2
        println("Number of terms in each factor of the Young operator: ", NumTermsInYOpFact)
    end
    
#-----------------------------------------------------------------
# Count the number of terms in the the simplified Y^{+}Y operator 
#-----------------------------------------------------------------
    i = 1
    for s in YHOpStr
        NumTermsInYHOpFact[i] = count("+", s) + count("-", s)
        i += 1
    end
    
    # Number of terms in the non simplified Y^{+}Y operator: NumYTerms*NumYTerms
    
    NumYHYTerms = param.NumYHYTerms # Retrieve number of terms in the simplified Y^{+}Y operator from param structure

    if verbose2
        println("Number of terms in each factor of the Young dagger operator: ", NumTermsInYHOpFact)
        println("Number of independent terms in the nonsimplified Y^{+}Y operator: ", NumYTerms*NumYTerms )
    end

    if verbose1
        println("Number of terms in the simplified Y^{+}Y operator, NumYHYTerms: ", NumYHYTerms)
    end
    
    return NumYTerms, NumYHYTerms
end

#------------------------------------------------------------------------------------------------------------------------------
# Define data_init() that initializes data structures
# 3D arrays YMatr and YHYMatr contain all matrices for 𝑌 and 𝑌†𝑌 operators. The structure is as follows:
#
# - YHYMatr(1:n,1:n,5) is the matrix corresponding to the 5-th term of 𝑌†𝑌 operator
# - Arrays YCoeff and YHYCoeff contain all coefficients (coefficients of permutations) in the 𝑌 and 𝑌†𝑌 operators.
# - Variables NumYTerms and NumYHYTerms are the number of independent terms in the 𝑌 and 𝑌†𝑌 operator respectively.
#------------------------------------------------------------------------------------------------------------------------------
function data_init(; inout_file=inout_file, param_file=param_file, NonlinParam_db_file=NonlinParam_db_file, 
        verbose1=verbose1, verbose2=verbose2)

    ok = true
    ZIndex_used = false
    read_inout_done = false

    #---------------------------------------------------------------------
    # Create an instance of the Param data structure with init_ok = false
    #---------------------------------------------------------------------
    param = ((CGL0 || N_C_Pvec) ? Param{ComplexF64} : Param{Float64})()
    param.data_init_ok = false

    # Create an empty action_list
    action_list::Vector{action} = [] 
    
    #--------------------------------------------
    # If an inout file is provided, then read it
    #--------------------------------------------
    if isfile(inout_file)
        ok, param, action_list = read_inout(; inout_file=inout_file, trace_f=trace_f)
        if ok
            read_inout_done = true
        else
            if verbose1
                println("\nread_inout() failed")
                println(trace_f, "\nread_inout() failed")
            end
        end
    end

    #-------------------------------------------------------------------------
    # If ecg_config.json provided an "actions" array (ECG_Param.config_actions,
    # set by ECG_Config.apply!), use it as the command script instead of the
    # one parsed from inout.txt. The system + basis (param) still come from
    # inout.txt / NonlinParam; only the action_list is replaced.
    #-------------------------------------------------------------------------
    if config_actions !== nothing
        action_list = actions_from_config(config_actions)
        if verbose1
            println("\ndata_init - using $(length(action_list)) action(s) from ecg_config.json (overriding the inout.txt script)")
            println(trace_f, "\ndata_init - using $(length(action_list)) action(s) from ecg_config.json (overriding the inout.txt script)")
        end
    end

    #-----------------------------------------------------------------------
    # If not param0.nlp0, read non linear basis set parameters file NonlinParam_db
    #-----------------------------------------------------------------------
    if !param0.nlp0
        ok = read_NonlinParam_db(; param=param, NonlinParam_db_file=NonlinParam_db_file, verbose1=verbose1)
    end

    #------------------------------------------------
    # If do_setup_nlp flag is true, call setup_nlp()
    #------------------------------------------------
    if param.do_setup_nlp
        setup_nlp(param.cbs; param=param, verbose1=verbose1)
    end

    npart = param.npart
    n = param.n
    npt = param.npt
    cbs = param.cbs
    
    #--------------------------
    # Create matrices H, diagH
    #--------------------------
    if CGL0 || N_C_Pvec
        H = zeros(ComplexF64,cbs,cbs)
        diagH = zeros(ComplexF64,cbs)
    else
        H = zeros(Float64,cbs,cbs)
        diagH = zeros(Float64,cbs)
    end
    
    param.H = H
    param.diagH = diagH
    
    #--------------------------
    # Create matrices S, diagS
    #--------------------------
    if CGL0 || N_C_Pvec
        S = zeros(ComplexF64,cbs,cbs)
        diagS = zeros(ComplexF64,cbs)
    else
        S = zeros(Float64,cbs,cbs)
        diagS = zeros(Float64,cbs)
    end
    
    param.S = S
    param.diagS = diagS
    
    #-------------------------------------------------------------------------------------------------------
    # Create matrix D which contains the derivatives of the Hamiltonian H and the overlap matrix elements S
    # D(1:np,i,j) contains dHij/dvechLi
    # D(np+1:2*np,i,j) contains dSij/dvechLi
    #-------------------------------------------------------------------------------------------------------
    if CGL0 || N_C_Pvec
        D = zeros(ComplexF64,2*npt,cbs,cbs)
    else
        D = zeros(Float64,2*npt,cbs,cbs)
    end
    param.D = D
    
    #------------------------------
    # Read Mass from file Mass.txt
    #------------------------------
    if isfile(Mass_file)
        Mass = zeros(npart)
        m = readdlm(Mass_file)
        Mass[1:npart] = m[2:npart+1] # Skip leading string MASSES
        param.Mass = Mass
        println(trace_f, "\nMass")
        println(trace_f, Mass)
    end
    
    #-------------------------
    # Set reduced mass matrix
    #-------------------------
    Mass = param.Mass
    MassMatrix = reduced_mass(Mass, n)
    param.MassMatrix = MassMatrix
    
    #----------------------------------------------
    # Read PseudoCharge from file PseudoCharge.txt
    #----------------------------------------------
    if isfile(PseudoCharge_file)
        PseudoCharge = zeros(Float64, n)
        charge = readdlm("PseudoCharge.txt")
        
        PseudoCharge0 = charge[2] # Skip leading string CHARGES
        param.PseudoCharge0 = PseudoCharge0
        
        PseudoCharge[1:n] = charge[3:n+2]
        param.PseudoCharge = PseudoCharge
        
        if verbose1
            println(trace_f, "\nPseudoCharge0: ", PseudoCharge0)
            println(trace_f, "PseudoCharge: ", PseudoCharge)
            println(trace_f, " ")
        end
    end
    
    #--------------------------------------------------
    # Read non-linear basis set parameters from a file 
    #--------------------------------------------------
    if !read_inout_done && isfile(NonlinParam_file)
        ok, ZIndex, NonlinParam, ZIndex_used = read_NonlinParam(; param=param, NonlinParam_file=NonlinParam_file, verbose1=verbose1)
        if ok
            param.ZIndex = ZIndex
            param.NonlinParam = NonlinParam
            param.ZIndex_used = ZIndex_used
        end
    end
    
    #--------------------------------------------------------------------------
    # Y / Y†Y operator terms: read from the four reference text files when all
    # of them are present, otherwise compute them with module SymmetryOperators
    # (symmetry_operators.jl) from the Young operator string and n.
    #--------------------------------------------------------------------------
    Y_files = ("YCoeff.txt", "YMatr.txt", "YHYCoeff.txt", "YHYMatr.txt")
    YOperatorString = param.YOperatorString

    if all(isfile, Y_files)
        #----------------------------------
        # Read YCoeff from file YCoeff.txt
        #----------------------------------
        YCoeff = readdlm("YCoeff.txt",Int)
        param.YCoeff = YCoeff
        param.NumYTerms = size(YCoeff)[1]
        if verbose2
            println(trace_f, "YCoeff: ", YCoeff)
            println(trace_f, " ")
        end

        #--------------------------------------
        # Read YHYCoeff from file YHYCoeff.txt
        #--------------------------------------
        YHYCoeff = readdlm("YHYCoeff.txt", Int)
        param.YHYCoeff = YHYCoeff
        param.NumYHYTerms = size(YHYCoeff)[1]
        if verbose2
            println(trace_f, "YHYCoeff: ", YHYCoeff)
            println(trace_f, " ")
        end

        #------------------------------------
        # Compute NumYTerms, NumYHYTerms
        #------------------------------------
        NumYTerms, NumYHYTerms = set_Young(YOperatorString; param=param, verbose1=verbose1, verbose2=verbose2)

        #--------------------------------
        # Read YMatr from file YMatr.txt
        #--------------------------------
        YMatr = zeros(n,n,NumYTerms)
        param.YMatr = YMatr
        read_matrix_3D(file="YMatr.txt",matrix=YMatr)

        #------------------------------------
        # Read YHYMatr from file YHYMatr.txt
        #------------------------------------
        YHYMatr = zeros(n,n,NumYHYTerms)
        param.YHYMatr = YHYMatr
        read_matrix_3D(file="YHYMatr.txt",matrix=YHYMatr)
    else
        #--------------------------------------------------------------------
        # Compute Y / Y†Y operator terms with module SymmetryOperators
        # (verified against the ATOM-MOL-nonBO reference files for the HD+,
        # D3+ and Nitrogen atom cases)
        #--------------------------------------------------------------------
        if verbose1
            println("\ndata_init - Y operator files ", join([f for f in Y_files if !isfile(f)], ", "),
                    " not found - computing Y / Y†Y operator terms with module SymmetryOperators")
            println(trace_f, "\ndata_init - Y operator files ", join([f for f in Y_files if !isfile(f)], ", "),
                    " not found - computing Y / Y†Y operator terms with module SymmetryOperators")
        end

        sym = SymmetryOperators.compute_operators(strip(YOperatorString), n)

        NumYTerms   = sym.NumYTerms
        NumYHYTerms = sym.NumYHYTerms

        YCoeff = reshape(round.(Int, sym.YCoeff), :, 1)
        param.YCoeff = YCoeff

        YHYCoeff = reshape(round.(Int, sym.YHYCoeff), :, 1)
        param.YHYCoeff = YHYCoeff

        YMatr = zeros(n,n,NumYTerms)
        for k in 1:NumYTerms
            YMatr[:,:,k] = sym.YMatr[k]
        end
        param.YMatr = YMatr

        YHYMatr = zeros(n,n,NumYHYTerms)
        for k in 1:NumYHYTerms
            YHYMatr[:,:,k] = sym.YHYMatr[k]
        end
        param.YHYMatr = YHYMatr

        if verbose1
            println("data_init - computed NumYTerms: $NumYTerms, NumYHYTerms: $NumYHYTerms")
            println(trace_f, "data_init - computed NumYTerms: $NumYTerms, NumYHYTerms: $NumYHYTerms")
        end
        if verbose2
            println(trace_f, "YCoeff: ", YCoeff)
            println(trace_f, "YHYCoeff: ", YHYCoeff)
            println(trace_f, " ")
        end
    end

    param.NumYTerms = NumYTerms
    param.NumYHYTerms = NumYHYTerms
    
    # Set Transposit matrix
    Transposit = set_Transposit(n, npart)
    param.Transposit = Transposit
    
    #-------------------------------------------------------------------------------------------------------------
    # Set up the PP(2*n,NumYHYTerms) matrix which MatrixElements() uses to permute elements using the Pvec method 
    #-------------------------------------------------------------------------------------------------------------
    PP = zeros(Int64,2*n,NumYHYTerms)
    
    for k in 1:NumYHYTerms       # number of independent terms in the Young operator Y
        for i in 1:n             # Number of pseudoparticles
            PP[i,k] = i
            PP[i+n,k] = i
            for j in 1:n
                if YHYMatr[i,j,k] == 1
                    PP[i,k] = j
                end
                if YHYMatr[j,i,k] == 1
                    PP[i+n,k] = j
                end
            end
        end
    end
    
    param.PP = PP
    
    if verbose1
        println(trace_f, "\nPP[:,1]")
        println(trace_f, PP[:,1])
    end
    
    #-----------------------------------------------------
    # Set up the covec matrix which MatrixElements() uses
    #-----------------------------------------------------
    covec = zeros(Int64, 36)
    
    y = 1.0
    for i in 1:18
        covec[2*i-1] = y
        covec[2*i] = -y
        y=-y
    end
    
    param.covec = covec
    
    #---------------------------------------------------------
    # Set up the Indentity matrix which MatrixElements() uses
    #---------------------------------------------------------
    Indentity = zeros(Int64, 2*n)
    
    for i in 1:n
        Indentity[i] = i
        Indentity[i+n] = i
     end
    
    param.Indentity = Indentity

    if !ok
        println("\nData initialization failed")
        println(trace_f, "\nData initialization failed")
        return false, param, action_list
    end
    
    return ok, param, action_list
end

#----------------------------
# Initialize data structures
#----------------------------

#------------------------------------------------------------------------------------------
# Define set_parvec() that looks up the bra and ket triples via BFPI[ZIndex[k0],:] 
# and BFPI[ZIndex[l0],:], permutes the ket # triple through PP[:,j0], and assembles 
# the 36-row parvec matrix that ECG_Matelem_N_Pvec / N_C_Pvec sum (with the covec
# signs) to evaluate the prefactor-weighted overlap, kinetic and potential matrix elements
#------------------------------------------------------------------------------------------

# parvec source pattern (36×6): for each row, columns (1,3,5) permute the bra
# premultiplier triple (i_kk,j_kk,k_kk) = sources 1,2,3, and columns (2,4,6)
# permute the ket triple (pi_ll,pj_ll,pk_ll) = sources 4,5,6.
const PARVEC_SRC = [
    2 5 3 6 1 4;  2 6 3 5 1 4;  3 5 2 6 1 4;  3 6 2 5 1 4
    2 6 3 4 1 5;  2 4 3 6 1 5;  3 6 2 4 1 5;  3 4 2 6 1 5
    2 4 3 5 1 6;  2 5 3 4 1 6;  3 4 2 5 1 6;  3 5 2 4 1 6
    3 5 1 6 2 4;  3 6 1 5 2 4;  1 5 3 6 2 4;  1 6 3 5 2 4
    3 6 1 4 2 5;  3 4 1 6 2 5;  1 6 3 4 2 5;  1 4 3 6 2 5
    3 4 1 5 2 6;  3 5 1 4 2 6;  1 4 3 5 2 6;  1 5 3 4 2 6
    1 5 2 6 3 4;  1 6 2 5 3 4;  2 5 1 6 3 4;  2 6 1 5 3 4
    1 6 2 4 3 5;  1 4 2 6 3 5;  2 6 1 4 3 5;  2 4 1 6 3 5
    1 4 2 5 3 6;  1 5 2 4 3 6;  2 4 1 5 3 6;  2 5 1 4 3 6
]

function set_parvec(param::Param, k0, l0, j0)
    @unpack trace_f, n, PP, ZIndex = param

    i_k = max(ZIndex[k0], 1)
    i_l = max(ZIndex[l0], 1)

    # sources: (i_kk, j_kk, k_kk, pi_ll, pj_ll, pk_ll)
    src = (BFPI[i_k, 1], BFPI[i_k, 2], BFPI[i_k, 3],
           PP[BFPI[i_l, 1], j0], PP[BFPI[i_l, 2], j0], PP[BFPI[i_l, 3], j0])

    parvec = zeros(Int64, 36, n)
    @inbounds for c in 1:6, r in 1:36
        parvec[r, c] = src[PARVEC_SRC[r, c]]
    end

    if verbose3
        println(trace_f, "\nparvec[1:36,1:n]: ", parvec[1:36, 1:n])
    end
    return parvec
end

#------------------------------------------------------------------------------------------------------------------------------
# init!() : run-time initialisation -- opens the trace file, prints the run
# parameters, sets the model flags, reads the parameter file and loads the data
# structures. This is the code that used to run at module-include time; it was
# moved into a function so ECG_Init can be precompiled. Called from run_ecg
# after the configuration has been applied.
#------------------------------------------------------------------------------------------------------------------------------
function init!()
    global trace_f, seed, basis, N_Pvec, N_matr, N_C_Pvec, RGL0, RGL1, CGL0, verbose1, verbose2, verbose3, npart, n, npt, cbs, NumYHYTerms, ApproxEnergy, param, action_list

    trace_f = open(trace_file,"w") # Open trace file in write mode
    println("Trace file: ", trace_file, " Verbose: ", verbose)
    #------------------------------------------------------------------------------------------------------------------------------
    # Print parameters
    #------------------------------------------------------------------------------------------------------------------------------
    println("\ncompute H_S method: $compute_H_S_method,  Matrix Elements method: $MatElem_method")
    println(trace_f, "\ncompute H_S method: $compute_H_S_method,  Matrix Elements method: $MatElem_method")

    println("\nUse GSEPIIS: ", do_GSEPIIS)
    println(trace_f, "\nUse GSEPIIS: ", do_GSEPIIS)

    println("Run Fortran program: ", do_Fortran)
    println(trace_f, "Run Fortran program: ", do_Fortran)

    #------------------------------------
    # Reseed the random number generator 
    #------------------------------------
    seed = max(floor(Int, param0.seed), 0)
    println("seed: $seed")
    println(trace_f, "seed: $seed")
    Random.seed!(seed)

    println("Julia Machine epsilon, _EPS: $_EPS")
    println(trace_f, "Julia Machine epsilon, _EPS: $_EPS")

    println("Print eigenvectors: ", param0.print_eigenvectors)
    println(trace_f, "Print eigenvectors: ", param0.print_eigenvectors)

    if param0.overlap_threshold > 0
        println("\nMaximum allowed overlap between functions: ", param0.overlap_threshold)
        println(trace_f, "\nMaximum allowed overlap between functions: ", param0.overlap_threshold)
    else
        println("\nNo maximum threshold defined for overlap between functions")
         println(trace_f, "\nNo maximum threshold defined for overlap between functions")
    end

    if param0.nlp_threshold > 0
        println("Maximum allowed value of non-linear basis set parameters: ", param0.nlp_threshold)
        println(trace_f, "Maximum allowed value of non-linear basis set parameters: ", param0.nlp_threshold)
    else
        println("No maximum threshold defined for value of non-linear basis set parameters")
        println(trace_f, "No maximum threshold defined for value of non-linear basis set parameters")
    end

    if param0.dist_diagH_threshold > 0
        s = @sprintf("%.2E", param0.dist_diagH_threshold)
        println("Minimum allowed distance between diagonal elements of the Hamiltonian matrix H: ", s)
        println(trace_f, "Minimum allowed distance between diagonal elements of the Hamiltonian matrix H: ", s)
    else
        println("No minimum threshold defined for distance between diagonal elements of the Hamiltonian matrix H")
        println(trace_f, "No minimum threshold defined for distance between diagonal elements of the Hamiltonian matrix H")
    end

    if param0.dist_func_threshold > 0
        s = @sprintf("%.2E", param0.dist_func_threshold)
        println("Minimum allowed distance between functions: $s")
        println(trace_f, "Minimum allowed distance between functions: $s")
    else
        param0.dist_func_threshold = 1000*_EPS
        s = @sprintf("%.2E", param0.dist_func_threshold)
        println("Setting minimum allowed distance between functions to: $s")
        println(trace_f, "Setting minimum allowed distance between functions to: $s")
    end

    println("\nTarget energy: ", param0.TargetEnergy)
    println(trace_f, "\nTarget energy: ", param0.TargetEnergy)

    println("YOperatorStringLength: ", YOperatorStringLength)
    println(trace_f, "YOperatorStringLength: ", YOperatorStringLength)

    println("Gradient booleans: grad_k: ", grad_k, " grad_l: ", grad_l)
    println(trace_f, "Gradient booleans: grad_k: ", grad_k, " grad_l: ", grad_l)

    println("Boolean param0.nlp0: ", param0.nlp0)
    println(trace_f, "Boolean param0.nlp0: ", param0.nlp0)

    let status = param0.coeff_threshold > 0 ?
            " (optim_nlp guided by |YHYCoeff| >= threshold; accepted energies use the full operator)" :
            " (off - full operator throughout)"
        println("Symmetry-term truncation, coeff_threshold: ", param0.coeff_threshold, status)
        println(trace_f, "Symmetry-term truncation, coeff_threshold: ", param0.coeff_threshold, status)
    end

    #------------------------------------------------------------------------------------------------------------------------------
    # Define global variables
    #------------------------------------------------------------------------------------------------------------------------------
    basis = (compute_H_S_method == "basis terms")

    N_Pvec = (MatElem_method == "N_Pvec")
    N_matr = (MatElem_method == "N_Matrix_operations")
    N_C_Pvec = (MatElem_method == "N_C_Pvec")

    RGL0 = (MatElem_method == "RGL0")
    RGL1 = (MatElem_method == "RGL1")

    CGL0 = (MatElem_method == "CGL0")

    verbose1 = verbose >= 1
    verbose2 = verbose >= 2
    verbose3 = verbose >= 3
    npart, n, npt, cbs, NumYHYTerms, ApproxEnergy = read_Param(param_file=param_file, trace_f=trace_f)
    ok, num = read_YHYCoeff(; trace_f=trace_f)
    if ok
        NumYHYTerms = num
    end
    ok, param, action_list = data_init(; inout_file=inout_file, param_file=param_file, verbose1=verbose1, verbose2=verbose2)

    return nothing
end # function init!

# init!() is called at run time by run_ecg (after the configuration is applied),
# so this module precompiles cleanly without a configuration.

end # Module ECG_Init