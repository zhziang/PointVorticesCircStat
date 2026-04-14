using FourierFlows, CUDA, HDF5

include("utils/pv2field.jl")
include("utils/fieldanalysis.jl")
include("utils/postprocess.jl")


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

function rectCircMoments(ns, width, height, order)
    res = map(ns) do n
        estimate(srcFiles) do u, circs
            ζh = grid.rfftplan * pv2field(u, circs, grid)
            Γ = rectCirculations(ζh, width * n, height * n, grid)
            sum(x -> x^order, Γ) / length(Γ)
        end
    end

    return getindex.(res, 1), getindex.(res, 2)
end

