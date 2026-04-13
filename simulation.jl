using CUDA, ArgParse, HDF5
import OrdinaryDiffEq as ODE
import DiffEqCallbacks as CB, NonlinearSolve as NLS
import Logging: global_logger
import TerminalLoggers: TerminalLogger
import SciMLBase
global_logger(TerminalLogger())
include("specfunc.jl")
include("PeriodicPV.jl")

#Parse argument
aps = ArgParseSettings()
@add_arg_table! aps begin
    "--npoints", "-n"
    help = "Number of point vortices (even number, half positive halfnegative)."
    arg_type = Int
    default = 32
    "--href"
    help = "Reference value of the Hamiltonian."
    arg_type = Float64
    default = 0.0
    "--tspan"
    help = "Time span of the simulation"
    arg_type = Float64
    default = 100
    "--path", "-p"
    help = "The path of the output file."
    arg_type = String
    default = @__DIR__
    "--gpu"
	help = "Using GPU acceleration."
	action = :store_true
end
args = parse_args(aps)

npoints = args["npoints"]
tspan = args["tspan"]
href = args["href"]

circs = args["gpu"] ? [CUDA.ones(npoints ÷ 2); -CUDA.ones(npoints ÷ 2)] ./ npoints : [ones(npoints ÷ 2); -ones(npoints ÷ 2)] ./ npoints 

odefunc = PeriodicPV(circs)

function hamiltonian(u)
    x, y = mod.(u[:, 1], 1), mod.(u[:, 2], 1)
    xdiff, ydiff = x .- transpose(x), y .- transpose(y)
    H = circs .* green.(xdiff, ydiff) .* transpose(circs)

    H[1:npoints.==(1:npoints)'] .= 0.0
    return sum(H)
end

function set_u₀(h₀)
    sspfunc(du,u,p,t) = begin
        odefunc(du,u,p,t)
        du .= hcat(du[:,2], -du[:,1]) .* circs .* (hamiltonian(u) - h₀)
    end

    prob = SciMLBase.SteadyStateProblem(sspfunc, rand(npoints,2))
    
    sol = ODE.solve(prob)

    return sol.u

end

u₀ = set_u₀(href)

function isoHamiltonian_manifold(residual, u, p, t)
    residual .= href - hamiltonian(u)
    return nothing
end

mproj = CB.ManifoldProjection(isoHamiltonian_manifold, autodiff=NLS.AutoForwardDiff(), resid_prototype=zeros(1))

prob = ODE.ODEProblem(odefunc, u₀, (0, tspan))

sol = @time ODE.solve(prob, ODE.Vern7(); progress=true, callback=mproj)

@info sol.retcode

output_path = args["path"] * "/.output/N$(npoints)H$(href).h5"
isdir(args["path"] * "/.output/") || mkdir(args["path"] * "/.output/")
h5open(output_path, "w") do fid
    create_dataset(fid, "pv positions", Float64, (npoints, 2, Int(1e3)))
    create_dataset(fid, "hamiltonians", Float64, (Int(1e3),))
    fid["circulations"] = Array(circs)
    for (n, t) in enumerate(range(0, tspan, Int(1e3)))
        fid["pv positions"][:, :, n] = Array(sol(t))
        fid["hamiltonians"][n] = hamiltonian(sol(t))
    end
end

