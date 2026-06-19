"""
    ECG_Julia

Open, modular implementation of the explicitly correlated Gaussian (ECG) method
for non-relativistic, finite-nuclear-mass atomic and molecular bound-state
calculations (including fully non-Born–Oppenheimer). See the repository README
and `docs/ECG_Julia_paper.pdf`.

Entry point: [`run_ecg`](@ref).

Released under the MIT License. Copyright (c) 2026 Alain Chancé.
"""
module ECG_Julia

export run_ecg

# Package root (parent of src/) — the science modules live here as sibling files.
const _ROOT = normpath(joinpath(@__DIR__, ".."))

# Load the science modules at package (pre)compile time, in dependency order:
#   ECG_Param  (parameter defaults)
#   ECG_Config (configuration reader + apply!; reaches ECG_Param lazily at call time)
#   ECG_Init   (data structures + init!())   — import ..ECG_Param
#   ECG        (driver + init!())             — import ..ECG_Param, ..ECG_Init
# Each module keeps its configuration-dependent setup in a run-time init!()
# (called from run_ecg), so the package precompiles cleanly without a configuration.
include(joinpath(_ROOT, "ECG_Param.jl"))
include(joinpath(_ROOT, "ECG_Config.jl"))
include(joinpath(_ROOT, "ECG_Init.jl"))
include(joinpath(_ROOT, "ECG.jl"))

"""
    run_ecg(workdir="."; config="ecg_config.json")

Run the ECG calculation defined by `config` (a JSON file in `workdir`):

1. read the configuration and set up the working directory (`ECG_Config`),
2. overlay the configuration onto the parameter defaults (`ECG_Config.apply!`),
3. initialise the run (`ECG_Init.init!` opens the trace file and loads the data;
   `ECG.init!` selects the matrix-element variant from `MatElem_method` and
   builds the work instance),
4. execute `ECG.do_action()` and return its result.

Data files are read from `workdir` (the configuration's `setup` block copies
them there); the code modules are loaded from the precompiled package.

The science modules are precompiled with the package, so `run_ecg` performs no
`include` at run time and calls the compiled code directly.

Run one model per Julia session (one `run_ecg` call per fresh kernel), as the
example notebooks do. The science modules hold per-run state in module globals
that is established afresh only when the package is loaded, so calling `run_ecg`
more than once in the same session can carry state over from the previous run.
(Resetting that state for repeated in-session runs is a possible future
enhancement.)
"""
function run_ecg(workdir::AbstractString = "."; config::AbstractString = "ecg_config.json")
    return cd(workdir) do
        # 1. configuration: read JSON, copy/clean data files into the working dir
        cfg = ECG_Config.read_config(config)
        ECG_Config.setup_workdir!(cfg)
        # 2. overlay the configuration onto the ECG_Param defaults
        ECG_Config.apply!(cfg)
        # 3. run-time initialisation: data load, then matelem selection + work instance
        ECG_Init.init!()
        ECG.init!()
        # 4. run; always close the trace file opened by ECG_Init afterwards
        try
            return ECG.do_action()
        finally
            isdefined(ECG_Init, :trace_f) && ECG_Init.trace_f isa IOStream &&
                isopen(ECG_Init.trace_f) && close(ECG_Init.trace_f)
        end
    end
end

end # module ECG_Julia
