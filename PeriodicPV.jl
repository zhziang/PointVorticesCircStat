mutable struct PeriodicPV <: Function
    t::Float64
    circs::AbstractArray{Float64,1}

    u::AbstractArray{Float64,2}
    du::AbstractArray{Float64,2}

    function PeriodicPV(circs)
        if isa(circs, CuArray)
            u = CUDA.rand(length(circs), 2)
            du = CUDA.zeros(length(circs), 2)
        elseif isa(circs, Array)
            u = rand(length(circs), 2)
            du = zeros(length(circs), 2)
        else
            error("Circs is neither a Array nor a CuArray.")
        end

        return new(0.0, circs, u, du)
    end
end

function update_du_kernel(du, u, circs)
    i = (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - UInt32(1)) * blockDim().y + threadIdx().y

    if 1 ≤ i ≤ length(circs) && 1 ≤ j ≤ length(circs)
        x_rel = mod(u[j, 1], 1) - mod(u[i, 1], 1)
        y_rel = mod(u[j, 2], 1) - mod(u[i, 2], 1)
        circ = circs[j]

        du₁, du₂ = (i == j) ? (0.0, 0.0) : hvec(x_rel, y_rel) .* circ

        CUDA.@atomic du[i, 1] += du₁
        CUDA.@atomic du[i, 2] += du₂
    end
    return nothing
end
function update_du(du::CuArray, u::CuArray, circs::CuArray)
    npoints = size(u, 1)
    blocksize = 16
    nblocks = ceil(Int, npoints / blocksize)
    @cuda blocks = (nblocks, nblocks) threads = (blocksize, blocksize) update_du(du, u, circs)
    return nothing
end

function update_du(du::Array, u::Array, circs::Array)
    u = mod.(u, 1)
    x_rel = transpose(u[:, 1]) .- u[:, 1]
    y_rel = transpose(u[:, 2]) .- u[:, 2]
    vecs = hvec.(x_rel, y_rel) .* transpose(circs)
    fill!(du, 0.0)
    for i in 1:length(circs)
        for j in [1:(i-1); (i+1):length(circs)]
            du[i, :] .+= vecs[i, j]
        end
    end

    return nothing
end

function (f::PeriodicPV)(du, u, p, t)
    copyto!(f.u, u)
    fill!(f.du, 0.0)
    update_du(f.du, f.u, f.circs)
    copyto!(du, f.du)
end
