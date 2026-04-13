mutable struct PeriodicPV_GPU <: Function
	t::Float64
	circs::CuArray{Float64, 1}

	u::CuArray{Float64, 2}
	du::CuArray{Float64, 2}

	function PeriodicPV_GPU(circs)
		u = CUDA.rand(length(circs), 2)
		du = CUDA.zeros(length(circs), 2)
		return new(0.0, circs, u, du)
	end
end

function update_du(du, u, circs)
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

function (f::PeriodicPV_GPU)(du, u, p, t)
	copyto!(f.u, u)
	npoints = size(u, 1)
	blocksize = 16
	nblocks = ceil(Int, npoints / blocksize)
	fill!(f.du, 0.0)
	@cuda blocks = (nblocks, nblocks) threads = (blocksize, blocksize) update_du(f.du, f.u, f.circs)
	copyto!(du, f.du)
end


mutable struct PeriodicPV_CPU <: Function
	t::Float64
	circs::Array{Float64, 1}

	u::Array{Float64, 2}
	du::Array{Float64, 2}

	function PeriodicPV_CPU(circs)
		u = rand(length(circs), 2)
		du = zeros(length(circs), 2)
		return new(0.0, circs, u, du)
	end
end

function (f::PeriodicPV_CPU)(du, u, p, t)
	copyto!(f.u, u)
	fill!(f.du, 0.0)
	for i in axes(u, 1), j in axes(u, 1)
		if i ≠ j
			xi, xj = mod(u[i, 1], 1), mod(u[j, 1], 1)
			yi, yj = mod(u[i, 2], 1), mod(u[j, 2], 1)
			du₁, du₂ = hvec(xj-xi, yj-yi)
			f.du[i, 1] += du₁ * circs[j]
			f.du[i, 2] += du₂ * circs[j]
		end
	end
	copyto!(du, f.du)
end

