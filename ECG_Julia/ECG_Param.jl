#------------------------------------------------------------------------------------------------------------------------------
# Author: Alain Chancé
# Date: June 4, 2023
# Version: 1.0
#
# Module ECG_Param
#------------------------------------------------------------------------------------------------------------------------------
module ECG_Param

export verbose, compute_H_S_method, MatElem_method, GSEPSolutionMethod, do_GSEPIIS, do_Fortran, outfile, inout_file, inout_F90_file  
export trace_file, seed, param_file, Mass_file, grad_k, grad_l, PseudoCharge_file, NonlinParam_file, NonlinParam_file_F90
export NonlinParam_db_file, _EPS, Mini, condition_max, param0, YOperatorStringLength, TargetEnergy, all_real, overlap_Skl, nlp0
export max_print_H, coeff_nlp, config_actions

# verbose level — default; overlaid from the "verbose" field of the ECG_Param
# section of ecg_config.json by ECG_Config.apply!. (Kept free of file I/O so the
# module precompiles cleanly as part of the ECG_Julia package.)
verbose = 1

compute_H_S_method = "basis terms"
#compute_H_S_method = "symmetry terms"

# Computation method used by matelem() function for Nitrogen only
MatElem_method = "N_Pvec"
#MatElem_method = "N_Matrix_operations"

# Computation method used by matelem() function for He and Li only
#MatElem_method = "RGL0"
#MatElem_method = "RGL0_Element_operations"

# Method used by StoreHS() function
GSEPSolutionMethod = "G"

# Maximum number of iterations in GSEPIIS
GSEPIIS_Max_iter = 200
do_GSEPIIS = false

# Boolean do_Fortran
do_Fortran = true

# Out file for shell scripts and Fortran program
outfile = "out.txt"
inout_file = "inout.txt"
inout_F90_file = "inout_F90.txt"
param_file = "Param_Julia.txt"
Mass_file = "Mass.txt"
PseudoCharge_file = "PseudoCharge.txt"
NonlinParam_file = "NonlinParam.txt"
trace_file = "Trace_Julia.txt"
NonlinParam_file_F90 = "NonlinParam_F90.txt"
NonlinParam_db_file = "NonlinParam_db.txt"

# Julia Machine epsilon
# https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Machine-epsilon
_EPS = eps(Float64) # _EPS = 2.22044e-16 Global variable used to chop small numbers to zero

# Minimum value for avoiding division by zero
Mini = 1e-90

# Boolean determines whether to read parameters from file param_file
read_param_file = false

#---------------------------------------------------------------------
# Define structure init_param that contains default parameters values
#---------------------------------------------------------------------
Base.@kwdef mutable struct Init_param
    name::String = ""      # Atom, Ion or molecule name
    npart::Int64 = 8
    cbs::Int64 = 6
    NumYHYTerms::Int = 5040
    ApproxEnergy::Float64 = -0.54589199E+02
    TargetEnergy::Float64 = 1.01*ApproxEnergy
    ntrials::Int64 = 1
    MaxEnergyEval::Int64 = 60
    print_eigenvectors::Bool = true
    do_solve_F90::Bool = false
    seed = 0
    #---------------------------------------------------------------------------------------
    # The following parameters control trial nonlinear parameters generation in stoch_nlp()
    coeff_nlp::Float64 = 0.15
    #
    # If gen_NonlinParam is true, stoch_nlp() generates each trial function with the Julia port of the
    # Fortran subroutine GenerateTrialParam (RGL0_1_F90/src/workproc.f90) as the FIRST method to try:
    # it perturbs a randomly selected existing (template) function by one of two stochastic methods.
    # Default false keeps the current x = coeff_nlp*uniform + template generation unchanged.
    gen_NonlinParam::Bool = false
    # Probability of using generation method 1 (independent per-parameter scaling); 1-RG_p1 selects method 2
    # (a single common scaling factor for all parameters of the selected function). [Fortran Glob_RG_p1]
    RG_p1::Float64 = 0.7
    # Method 1 relative standard deviation: new = (1 + RG_s1*randn())*template, per parameter. [Glob_RG_s1]
    RG_s1::Float64 = 1.0
    # Method 2 relative standard deviation: factor = 1 + RG_s2*randn(), redrawn until |factor| is outside
    # (0.8,1.2) to avoid near-linearly-dependent functions; new = factor*template. [Glob_RG_s2]
    RG_s2::Float64 = 3.0
    #
    nlp0::Bool = true
    # nlp0 = false # Fetch parameters from file "NonlinParam_db.txt"
    #
    # If nlp0 and shuffle_NonlinParam are true, then shuffle param.NonlinParam
    shuffle_NonlinParam = false
    #
    # The maximum threshold of the condition is defined as the ratio between the eigenvalue with largest magnitude
    # over the one with the smallest magnitude. stochnlp() discards functions that exceeds this threshold.
    #
    # Ref. Andrea Muolo, Explicitly Correlated Gaussians and the Quantum Few-Body Problem, DISS. ETH NO. 25680, 
    # December 2018, https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/352293/1/AMuolo.pdf
    # 2.4 The stochastic variational method, 3.8 Numerical stability of complex functions
    condition_max = 1e10
    #
    # If overlap_threshold > 0 then a generated function is rejected if the overlap between any accepted function exceeds this threshold
    overlap_threshold::Float64 = 0.0
    #
    # If nlp_threshold > 0 then a generated function is rejected if the maximum magnitude of the non-linear basis set parameters
    # exceeds this threshold
    nlp_threshold::Float64 = 0.25E+02
    #
    # If dist_diagH_threshold > 0 then a generated function is rejected if the distance with any accepted functions is below this threshold
    dist_diagH_threshold::Float64 = 0.0
    #
    # If dist_func_threshold > 0 then a generated function is rejected if the distance with any accepted functions is below this threshold
    dist_func_threshold::Float64 = 1000*_EPS
    #
    # If discard_degenerate is true then a function that yields doubly degenerate eigenvalues is discarded
    discard_degenerate::Bool = true
    #
    #---------------------------------------------------------------------------------------------------------------
    # The following parameters are used to set corresponding general options of the Optim package
    # https://julianlsolvers.github.io/Optim.jl/stable/user/config/
    #---------------------------------------------------------------------------------------------------------------
    # If optim is true then run the optim_nlp() function to perform a cyclic optimization of the non-linear basis set
    optim_iter::Int64 = 10   # iterations: How many iterations will run before the algorithm gives up?
    optim_time_limit::Float64 = 10.0 # A soft upper limit on the total run time.
    #
    # Symmetry-term truncation threshold used ONLY to guide nonlinear-parameter optimization (optim_nlp).
    # During the inner BFGS search, Y^{+}Y terms whose |YHYCoeff| < coeff_threshold are skipped, giving a cheaper
    # (partially symmetrized, non-variational) energy surface; every accepted/reported energy is recomputed with the
    # full operator. coeff_threshold = 0.0 disables truncation (full operator everywhere -- identical to before).
    # Example (Nitrogen, |coeff| in {96,192,960}): coeff_threshold = 192 drops the 96-class (~48% of 5040 terms).
    coeff_threshold::Float64 = 0.0
    #
    # Automatic threshold ramp: when > 0, optim_nlp starts at coeff_threshold (coarse) and lowers the threshold
    # to the next-finer coefficient tier (eventually 0 = full operator) whenever a cycle's total energy
    # improvement falls below coeff_ramp_tol. 0.0 keeps a fixed threshold (no ramp). Requires coeff_threshold > 0.
    coeff_ramp_tol::Float64 = 0.0
    #
    # If true, optim_nlp passes an analytic energy gradient (loss_grad!) to Optim instead of relying on Optim's
    # finite-difference gradient. Requires a matelem that returns correct Dk (e.g. ECG_Matelem_N_Pvec_AD.jl).
    # Default false keeps the finite-difference path unchanged.
    analytic_grad::Bool = false
    #
    #---------------------------------------------------------------------------------------------------------------
    # The following parameters are written into file Glob_Param.txt and read by Fortran program module globvars.f90
    #---------------------------------------------------------------------------------------------------------------
    # Maximum  fraction of Eigenvalue problem solution failures for random selection process
    Max_Frac_Of_Trial_Fails_Allowed = 0.15        # Glob_MaxFracOfTrialFailsAllowed
    # 
    # Maximum number of failures allowed in energy or gradient evaluation are allowed during optimization of nonlinear parameters
    # Check setting in https://github.com/LucasLang/ATOM-MOL-nonBO/commit/
    Max_Energy_Fails_Allowed = 10                 # Glob_MaxEnergyFailsAllowed
    #
    # Maximum number of times basis enlargement routine or the cyclic optimization routine are allowed to repeat
    # random trial and optimization process if the generated function/functions end up being linearly dependent with other
    # functions in the basis (overlap is close to 1.0) or any linear parameters by magnitude are greater then threshold.
    Bad_Overlap_Or_LinCoeffLim = 10               # Glob_BadOverlapOrLinCoeffLim
end

# Create param0 an instance of the Init_param structure
param0 = Init_param()

YOperatorStringLength = 100

max_print_H = 30

grad_k::Bool = false
grad_l::Bool = false

all_real = "F"

#----------------------------
# Overlap normalized boolean
#----------------------------
overlap_Skl = true

# overlap = true
# Overlap normalized
# Skl = 2^3n/2 (||Lk|| ||Ll||/|AKL|)^3/2

# overlap = false
# [Muolo] A.1.1 Overlap integral

# In ECG_Matelem_RGL0
# Skl = PI^(3.0*n/2.0)/det_tAkl*sqrt(det_tAkl)

# In ECG_Matelem_CGL0
# Skl = PI^(3.0*n/2.0)/det_tCkl*sqrt(det_tCkl)
#----------------------------------------------

#------------------------------------------------------------------------------
# config_actions — the "actions" array from ecg_config.json, if any.
# ECG_Config.apply! stores cfg["actions"] here (a Vector of Dicts); ECG_Init's
# data_init() turns it into the action_list (via actions_from_config), an
# alternate to the legacy inout.txt command script. nothing = use inout.txt.
#------------------------------------------------------------------------------
config_actions = nothing

end # ECG_Param