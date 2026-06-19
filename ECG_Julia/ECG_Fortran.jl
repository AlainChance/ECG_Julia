#------------------------------------------------------------------------------------------------------------------------------
# Module ECG_Fortran
#
# Fortran-interface functions split out of ECG.jl (June 2026). These drive and read back the external Fortran program
# (the `main` binary) through a TEXT-FILE interface only (no in-memory API): parameters and matrices are exchanged via
# *.txt files and `./main` is launched as a subprocess. They are only exercised when do_Fortran is true.
#
# The few numerical helpers they rely on — read_matrix, write_matrix, symm, solve_eigen, compute_and_solve — live in
# module ECG and are reached through a reference that ECG installs at load time via set_ecg_module!(ECG); this avoids a
# circular module dependency (ECG imports the functions below, while the functions below call back into ECG).
#
# Public functions: read_complex_matrix, compute_and_solve_F90, solve_F90, run_Fortran.
#------------------------------------------------------------------------------------------------------------------------------
module ECG_Fortran

import ...ECG_Param: verbose, MatElem_method, do_Fortran, outfile, inout_F90_file, NonlinParam_file, NonlinParam_file_F90
import ...ECG_Param: overlap_Skl, all_real, param0
import ...ECG_Init: trace_f, verbose1, verbose2, Param, param, CGL0, N_C_Pvec, read_inout, read_NonlinParam, write_NonlinParam, history

using Parameters, Printf

#------------------------------------------------------------------------------------------------------------------------------
# Fixed file / executable names exchanged with the Fortran program (text-file interface).
#------------------------------------------------------------------------------------------------------------------------------
const MAIN_EXE  = "main"               # Fortran executable launched as ./main
const GLOB_H    = "Glob_H"             # H matrix file produced by the "complete" Fortran build
const GLOB_S    = "Glob_S"             # S matrix file produced by the "complete" Fortran build
const H90_NAME  = "H_90"               # H matrix file produced by the "custom" Fortran build
const S90_NAME  = "S_90"               # S matrix file produced by the "custom" Fortran build
const GLOB_NLP  = "Glob_NonlinParam.txt"
const SWAPFILE  = "swapfile.dat"
const REALLOC   = "realloc.dat"

#------------------------------------------------------------------------------------------------------------------------------
# Numerical helpers that live in module ECG. To avoid a circular module dependency, they are reached through a reference to
# module ECG that ECG installs right after including this file (ECG_Fortran.set_ecg_module!(@__MODULE__)). The bindings are
# resolved at call time, so it does not matter that those helpers are defined later in ECG than this include.
#------------------------------------------------------------------------------------------------------------------------------
const _ECG = Ref{Module}()
set_ecg_module!(m::Module) = (_ECG[] = m; nothing)

_ecg() = isassigned(_ECG) ? _ECG[] :
    error("ECG_Fortran: module ECG not registered - ECG must call ECG_Fortran.set_ecg_module!(ECG) at load time")

read_matrix(args...; kw...)       = _ecg().read_matrix(args...; kw...)
write_matrix(args...; kw...)      = _ecg().write_matrix(args...; kw...)
symm(args...; kw...)              = _ecg().symm(args...; kw...)
solve_eigen(args...; kw...)       = _ecg().solve_eigen(args...; kw...)
compute_and_solve(args...; kw...) = _ecg().compute_and_solve(args...; kw...)

#------------------------------------------------------------------------------------------------------------------------------
# _snapshot / _restore! : capture and restore the param fields that a Fortran round-trip may overwrite, so a rejected
# Fortran result leaves param exactly as it was. (Replaces the long inline copy/restore blocks; ApproxEnergy is captured
# for reference but deliberately NOT restored, matching the original compute_and_solve_F90 behaviour.)
#------------------------------------------------------------------------------------------------------------------------------
function _snapshot(p::Param)
    return (ZIndex         = copy(p.ZIndex),
            NonlinParam    = copy(p.NonlinParam),
            ZIndex_used    = p.ZIndex_used,
            H              = copy(p.H),
            diagH          = copy(p.diagH),
            S              = copy(p.S),
            diagS          = copy(p.diagS),
            D              = copy(p.D),
            CurrEnergy     = p.CurrEnergy,
            InvitParameter = p.InvitParameter,
            EigvalTol      = p.EigvalTol,
            LastEigvector  = copy(p.LastEigvector),
            grad_k         = p.grad_k,
            grad_l         = p.grad_l,
            Evectors       = copy(p.Evectors),
            Evalue_eigen   = p.Evalue_eigen,
            Evalues        = copy(p.Evalues),
            condition      = p.condition)
end

function _restore!(p::Param, s)
    p.ZIndex         = s.ZIndex
    p.NonlinParam    = s.NonlinParam
    p.ZIndex_used    = s.ZIndex_used
    p.H              = s.H
    p.diagH          = s.diagH
    p.S              = s.S
    p.diagS          = s.diagS
    p.D              = s.D
    p.CurrEnergy     = s.CurrEnergy
    p.InvitParameter = s.InvitParameter
    p.EigvalTol      = s.EigvalTol
    p.LastEigvector  = s.LastEigvector
    p.grad_k         = s.grad_k
    p.grad_l         = s.grad_l
    p.Evectors       = s.Evectors
    p.Evalue_eigen   = s.Evalue_eigen
    p.Evalues        = s.Evalues
    p.condition      = s.condition
    return nothing
end

#------------------------------------------------------------------------------------------------------------------------------
# _accept_energy : shared accept/reject decision for an energy returned by the Fortran round-trip. `src` names the source of
# the candidate (for the log messages). Returns true to accept (energy is a valid improvement), false to reject. The caller
# performs the accept-side bookkeeping (writing files) and the reject-side restore.
#
# The three comparison phrases are per-call templates so each caller keeps its original wording: the defaults reproduce
# compute_and_solve_F90's messages ("< target energy", "> current energy", "<= current energy"), while solve_F90 passes its
# own ("is lower than target energy", "is greater than current energy", "is lower than or equal to current energy").
#------------------------------------------------------------------------------------------------------------------------------
function _accept_energy(p::Param, prev_energy, src::AbstractString, verbose1::Bool;
        below_target::AbstractString  = "< target energy",
        above_current::AbstractString = "> current energy",
        within::AbstractString        = "<= current energy")
    E   = p.CurrEnergy
    tol = p.EigvalTol
    tgt = param0.TargetEnergy

    if E < tgt - tol
        msg = "\nEnergy $E computed using $src $below_target $tgt"
        accept = false
    elseif E > prev_energy + tol
        msg = "\nEnergy $E computed using $src $above_current $prev_energy"
        accept = false
    elseif isnan(E)
        msg = "\nEnergy computed using $src is $E"
        accept = false
    else
        msg = "\nEnergy $E computed using $src $within $prev_energy"
        accept = true
    end

    if verbose1
        println(msg)
        println(trace_f, msg)
    end
    return accept
end

"""
    read_complex_matrix(; matrix_name="H_90", verbose1=verbose1) -> (ok, U)

Read a real matrix file written by the Fortran program as interleaved (REAL, AIMAG) column pairs and reassemble it into a
complex matrix `U`, with `U[i,j] = M[i,2j-1] + im*M[i,2j]`. Suited to a Fortran write of the form

    write(*,FMT) (REAL(Glob_H(i,j)), AIMAG(Glob_H(i,j)), j=1,cbs)

Returns `(false, nothing)` if the underlying matrix file could not be read.
"""
function read_complex_matrix(; matrix_name=H90_NAME, verbose1=verbose1)

    ok, M = read_matrix(; matrix_name=matrix_name, verbose1=verbose1)

    if ok
        l::Int64 = size(M)[1]
        U::Array{ComplexF64} = zeros(ComplexF64, l, l)

        for i in 1:l
            for j in 1:l
                U[i,j] = complex(M[i,2*j-1], M[i,2*j])   # (REAL, AIMAG) pair -> complex
            end
        end
        return true, U
    else
        return false, nothing
    end
end

"""
    compute_and_solve_F90(; param=param, inout_F90_file=inout_F90_file,
                            NonlinParam_file_F90=NonlinParam_file_F90, verbose1, verbose2, print=true) -> (ok, H_90, S_90)

Read the nonlinear parameters returned by the Fortran program (from `inout_F90` and/or `NonlinParam_F90.txt`), recompute
H and S and re-solve the secular equation, and accept the result only if its energy is a valid improvement. On rejection
the original `param` state is restored and `(false, nothing, nothing)` is returned.
"""
function compute_and_solve_F90(; param::Param=param, inout_F90_file=inout_F90_file, NonlinParam_file_F90=NonlinParam_file_F90,
        verbose1=verbose1, verbose2=verbose2, print=true)

    ok = true
    H_90 = nothing
    S_90 = nothing

    # Snapshot the param state, then select range Nmin, Nmax to be 1, cbs and disable gradient flags.
    snap = _snapshot(param)
    param.Nmin = 1
    param.Nmax = param.cbs
    param.grad_k = false
    param.grad_l = false

    if isfile(inout_F90_file)
        #-------------------------------------------------------------------------------------------------
        # Read inout_F90 file and print current energy returned by Fortran program in file inout_F90_file
        #-------------------------------------------------------------------------------------------------
        history_list_F90::Vector{history} = []  # History list (required by read_inout, unused here)
        ok, param_F90, action_list_F90 = read_inout(; inout_file=inout_F90_file, NonlinParam_file=NonlinParam_file_F90,
                trace_f=trace_f, history_list=history_list_F90, verbose1=false)

        param.ZIndex = param_F90.ZIndex
        param.NonlinParam = param_F90.NonlinParam
        param.ZIndex_used = param_F90.ZIndex_used

        if verbose1
            energy = param_F90.CurrEnergy
            println("\nFortran program returned current energy: $energy in file: $inout_F90_file")
            println(trace_f, "\nFortran program returned current energy: $energy in file: $inout_F90_file")
        end
    end

    if isfile(NonlinParam_file_F90)
        #------------------------------------------------------------------------------------------------------
        # Read non-linear parameters NonlinParam_F90 from file NonlinParam_F90.txt returned by Fortran program
        #------------------------------------------------------------------------------------------------------
        ok, param.ZIndex, param.NonlinParam, param.ZIndex_used =
            read_NonlinParam(; param=param, NonlinParam_file=NonlinParam_file_F90, verbose1=verbose1)

        # Check that the number of lines read from file NonlinParam_F90.txt is equal to the current basis size, cbs
        n_lines = size(param.NonlinParam)[1]
        if param.cbs != n_lines
            ok = false
            if verbose1
                println("\nNumber of lines ", n_lines, " in file ", NonlinParam_file_F90, " is not equal to current basis size ", param.cbs)
                println(trace_f, "\nNumber of lines ", n_lines, " in file ", NonlinParam_file_F90, " is not equal to current basis size ",
                    param.cbs)
            end
        end

    else
        ok = false
        if verbose1
            println("\nFile ", NonlinParam_file_F90, " containing non-linear basis set parameters not found")
            println(trace_f, "\nFile ", NonlinParam_file_F90, " containing non-linear basis set parameters not found")
        end
    end

    # Compute H and S matrices and solve secular equation
    if ok
        if verbose1
            println("\nComputing H, S matrices and solving secular equation using non linear parameters returned by Fortran program")
            println(trace_f, "\nComputing H, S matrices and solving secular equation using non linear parameters returned by Fortran program")
        end
        ok = compute_and_solve(; param=param, verbose1=false, verbose2=false, print=print, H_name=GLOB_H, S_name=GLOB_S)
    end

    # Compare with the previous current energy the one returned by compute_and_solve()
    if ok
        if _accept_energy(param, snap.CurrEnergy, "non linear parameters returned by Fortran program", verbose1)
            # Write non-linear basis set parameters and the H, S matrices, and point H_90/S_90 at the accepted matrices
            write_NonlinParam(param.cbs, param.npt, param.ZIndex, param.NonlinParam, NonlinParam_file=NonlinParam_file,
                ZIndex_used=param.ZIndex_used, verbose1=verbose1)
            write_matrix(param.H; matrix_name="H", verbose1=verbose1)
            write_matrix(param.S; matrix_name="S", verbose1=verbose1)
            H_90 = param.H
            S_90 = param.S
        else
            ok = false
        end
    end

    if !ok
        # Either compute_and_solve() failed, or the energy returned was not a valid improvement: restore param state.
        H_90 = nothing
        S_90 = nothing
        _restore!(param, snap)
    end

    return ok, H_90, S_90

end

"""
    solve_F90(; param=param, H_90_name="H_90", S_90_name="S_90", verbose1, verbose2) -> (ok, H_90, S_90)

Read the `H_90` and `S_90` matrix files produced by the Fortran program and use them to re-solve the secular equation,
accepting the result only if its energy is a valid improvement. On rejection the saved eigen-state is restored.
"""
function solve_F90(; param::Param=param, H_90_name=H90_NAME, S_90_name=S90_NAME, verbose1=verbose1, verbose2=verbose2)

    # Snapshot the param state so a rejected result can be rolled back
    snap = _snapshot(param)

    #------------------------------------------------------------------------------
    # If H_90 and S_90 matrices are there, then use them to solve secular equation
    #------------------------------------------------------------------------------
    # Read matrix H_90 from file H_90.txt or exit if not there
    ok, H_90 = read_matrix(; matrix_name=H_90_name)

    if H_90 === nothing
        return false, nothing, nothing
    else
        H_90 = symm(H_90)                              # Symmetrize H_90
        write_matrix(H_90; matrix_name=H_90_name)      # Write matrix back into file H_90.txt
    end

    # Read matrix S_90 from file S_90.txt or exit if not there
    if ok
        ok, S_90 = read_matrix(; matrix_name=S_90_name)
        if S_90 === nothing
            return false, nothing, nothing
        end
    end

    #--------------------------------------------------
    # Solve secular equation with solve_eigen function
    #--------------------------------------------------
    ok, Eval_eigen, Eval, cond = solve_eigen(; param=param, H=H_90, S=S_90, verbose1=verbose1, verbose2=verbose2)
    if !ok
        if verbose1
            println("\nsolve_eigen failed")
            println(trace_f, "\nsolve_F90 - solve_eigen failed")
        end
        return false, nothing, nothing
    end

    # Compare energy computed using H_90 and S_90 with the previous current energy (solve_F90's original wording)
    if _accept_energy(param, snap.CurrEnergy, "$H_90_name and $S_90_name", verbose1;
            below_target  = "is lower than target energy",
            above_current = "is greater than current energy",
            within        = "is lower than or equal to current energy")
        # Write non-linear basis set parameters to text file NonlinParam_file
        write_NonlinParam(param.cbs, param.npt, param.ZIndex, param.NonlinParam, NonlinParam_file=NonlinParam_file,
            ZIndex_used=param.ZIndex_used, verbose1=verbose1)
    else
        ok = false
        _restore!(param, snap)
    end

    return ok, H_90, S_90

end

"""
    run_Fortran(; param=param, verbose1, verbose2, print=true) -> (ok, H_90, S_90)

Launch the external Fortran program over the text-file interface:
  - copy NonlinParam.txt -> Glob_NonlinParam.txt and make `main` executable (native Julia, no shell),
  - run `./main` with stdout redirected to `outfile`, then echo its last lines,
  - if an inout_F90 / NonlinParam_F90 file is produced, finish via compute_and_solve_F90,
  - otherwise (real methods) finish via solve_F90 using the H_90 / S_90 matrix files.
Returns `(false, nothing, nothing)` when do_Fortran is false, `main` is missing, the method is unsupported, or the run fails.
"""
function run_Fortran(; param::Param=param, verbose1=verbose1, verbose2=verbose2, print=true)

    ok = true

    start_string = "\n----------------------------- Run_Fortran --------------------------------------------------"
    end_string = "\n----------------------------- Run_Fortran - End --------------------------------------------"

    #-----------------------------
    # Exit if do_Fortran is false
    #-----------------------------
    if !do_Fortran
        return false, nothing, nothing   # Exit if do_Fortran is false
    end

    if verbose1
        println(start_string)
        println(trace_f, start_string)
    end

    #-------------------------------------------------------------
    # Retrieve current basis size, n and npt from param structure
    #-------------------------------------------------------------
    cbs = param.cbs
    n = param.n
    np1 = Int(n*(n+1)/2)
    npt = param.npt

    #--------------------------------------------------------------------------------------
    # If CGL0 or N_C_Pvec method is used, then npt = 2*np1 where np1 = Int(n*(n+1)/2)
    # ECG_Init, Read_inout, Phase 1, Particles
    #--------------------------------------------------------------------------------------
    if (CGL0 || N_C_Pvec) && npt == np1
        if verbose1
            println("Number of nonlinear parameters per basis function, npt: ", npt, " not yet supported for CGL0 or N_C_Pvec method")
            println(end_string)

            println(trace_f, "Number of nonlinear parameters per basis function, npt: ", npt, " not yet supported for CGL0 or N_C_Pvec method")
            println(trace_f, end_string)
        end
        return false, nothing, nothing
    end

    #-----------------------------
    # Check that main file exists
    #------------------------------
    if !isfile(MAIN_EXE)
        if verbose1
            println("Fortran main program missing - Exiting")
            println(trace_f, "Fortran main program missing - Exiting")
        end
        return false, nothing, nothing   # Exit if main file is missing
    end

    #----------------------------------------------------------------------------------------------
    # Run Fortran main program
    # https://docs.julialang.org/en/v1/manual/running-external-programs/#Running-External-Programs
    #----------------------------------------------------------------------------------------------
    if verbose1
        println("\nFortran main output: ", outfile)
        println(trace_f, "\nFortran main output: ", outfile)
    end

    try
        # Copy file containing non-linear basis set parameters (native Julia, no shell)
        cp(NonlinParam_file, GLOB_NLP; force=true)

        # Ensure that we have rights to execute main (add the execute bits, like chmod +x)
        chmod(MAIN_EXE, filemode(MAIN_EXE) | 0o111)

        # Run Fortran program with stdout redirected to outfile
        run(pipeline(`./$MAIN_EXE`, stdout=outfile))

    catch e
        if verbose1
            println("\nFortran program main failed")
            println(end_string)

            println(trace_f, "\nFortran program main failed")
            println(trace_f, end_string)
        end
        return false, nothing, nothing
    end

    #------------------
    # Read output file
    #------------------
    if isfile(outfile)
        out_f = open(outfile)
        ea = eachline(out_f)
        local line

        line = ""
        line_1 = ""
        line_2 = ""
        line_3 = ""

        for l in ea
            line_3 = line_2
            line_2 = line_1
            line_1 = line
            line = l
        end

        z = split(line)

        if !isempty(z)
            if z[1] == "Error" || z[1] == "The" || z[1] == "in" || z[1] == "exceeded" || z[1] == "Ordering"
                if verbose1
                    println("\nFortran program main completed with error")
                    println(trace_f, "\nFortran program main completed with error")
                end
            end
        end

        if verbose1
            println("\nLast four lines in ", outfile)
            println(line_3)
            println(line_2)
            println(line_1)
            println(line)

            println(trace_f, "\nLast four lines in ", outfile)
            println(trace_f, line_3)
            println(trace_f, line_2)
            println(trace_f, line_1)
            println(trace_f, line)
        end
    end

    #--------------------------------------------------------------------------------------------------
    # If there is a file inout_F90_file or a file NonlinParam_file_F90 then call compute_and_solve_F90
    #--------------------------------------------------------------------------------------------------
    if isfile(inout_F90_file) || isfile(NonlinParam_file_F90)
        ok, H_90, S_90 = compute_and_solve_F90(; param=param, NonlinParam_file_F90=NonlinParam_file_F90, verbose1=verbose1,
            verbose2=verbose2, print=print)

        if verbose1
            println(end_string)
            println(trace_f, end_string)
        end
        return ok, H_90, S_90
    end

    #----------------------------------------------------
    # If CGL0 or N_C_Pvec method is used, then exit
    #----------------------------------------------------
    if CGL0 || N_C_Pvec
        if verbose1
            println(end_string)
            println(trace_f, end_string)
        end
        return false, nothing, nothing
    end

    #-----------------------------------------------------------------------------
    # Solve secular equation with H_90 and S_90 files returned by Fortran program
    #-----------------------------------------------------------------------------
    ok, H_90, S_90 = solve_F90(; param=param, H_90_name=H90_NAME, S_90_name=S90_NAME, verbose1=verbose1, verbose2=verbose2)

    if verbose1
        println(end_string)
        println(trace_f, end_string)
    end

    return ok, H_90, S_90

end # function run_Fortran

end # module ECG_Fortran
