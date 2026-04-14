function pv2field(u::Array, circs::Array, grid)
    u .= mod.(u, grid.Lx)
    idx = floor.(Int, u ./ grid.dx) .+ 1
    ζ = zeros(grid.nx, grid.ny)
    for (n, circ) in zip(axes(idx, 1), circs)
        i, j = idx[n, 1], idx[n, 2]
        ζ[i, j] += circ / (grid.dx * grid.dy)
    end
    return ζ
end

function pv2field_kernel(ζ, u, circs, d, L)
    n = (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x
    i = floor(Int, mod(u[n, 1], L) / d) + 1
    j = floor(Int, mod(u[n, 2], L) / d) + 1
    circ = circs[n]
    CUDA.@atomic ζ[i, j] += circ / d^2
    return nothing
end

function pv2field(u::CuArray, circs::CuArray, grid)
    ζ = CUDA.zeros(grid.nx, grid.ny)
    npoints = length(circs)
    blocksize = 16
    nblocks = ceil(Int, npoints / blocksize)
    @cuda blocks = (nblocks) threads = (blocksize) pv2field_kernel(ζ, u, circs, grid.dx, grid.Lx)
    return ζ
end
