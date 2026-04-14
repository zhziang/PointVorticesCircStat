using FourierFlows, CUDA, HDF5
include("utils/pv2field.jl")
include("utils/fieldanalysis.jl")


basepath = dirname(@__DIR__)

srcFiles = h5open(basepath * "/.output/N64H0.0.h5")
desFile = h5open(basepath * "/postprocess.h5", "w")

function estimate(func, file)
    circs = read(file["circulations"])
    samples = Iterators.map(1:1000) do n
        u = file["pv positions"][:, :, n]
        func(u, circs)
    end
    nsamples = length(samples)
    mean = sum(samples) ./ nsamples
    std = sqrt.(sum(x -> (x .- mean) .^ 2, samples) ./ nsamples)
    return (mean, std)
end

grid = TwoDGrid(CPU(); nx=256, Lx=1)

function rectCircMoments(u, circs, ns, width, height, order)
    return map(ns) do n
        ζh = grid.rfftplan * pv2field(u, circs, grid)
        Γ = rectCirculations(ζh, width * n, height * n, grid)
        sum(x -> x^order, Γ) / length(Γ)
    end
end

ns = 1:10
var10_10 = estimate((u,circs) -> rectCircMoments(u,circs,ns,10,10,2),srcFiles)
var5_20 = estimate((u,circs) -> rectCircMoments(u,circs,ns,5,20,2),srcFiles)
var4_16 = estimate((u,circs) -> rectCircMoments(u,circs,ns,4,16,2),srcFiles)


h5open("")