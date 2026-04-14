using FourierFlows, CUDA, HDF5

include("utils/pv2field.jl")
include("utils/fieldanalysis.jl")

function readRawData(name)
    us = Iterators.map(1:1000) do n
        h5open(fid -> fid["pv positions"][:, :, n], name, "r")
    end

    circs = h5open(fid -> read(fid["circulations"]), name, "r")

    return circs, us
end

function equalAreaCircRatio(data)
    circs, us = data
    grid = FourierFlows.TwoDGrid(CPU(), nx=256, Lx=1)
    loopsizes = 4:4:128


    varRatio(l) = begin
        var_sq = sum(us) do u
            ζh = grid.rfftplan * pv2field(u, circs, grid)
            sum(abs2, rectCirculations(ζh, l, l, grid))
        end ./ length(us)

        var_rect = sum(us) do u
            ζh = grid.rfftplan * pv2field(u, circs, grid)
            sum(abs2, rectCirculations(ζh, 2l, l/2, grid))
        end ./ length(us)

        var_rect / var_sq
    end

    return loopsizes, [varRatio(l) for l in loopsizes]
end


readRawData(".output/N64H0.0.h5") |> equalAreaCircRatio