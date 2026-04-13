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

"""
	Compute the velocity circulation over a rectangular loop fixed by 'width' and 'height'
	using the convolution of the vorcity field 'ζ' and the loop Heaviside 'hs'.
	'ζh' and 'hsh' are their Fourier transforms respectively.
"""
function rectCirculations(ζh, width::Int, height::Int, grid)
	hsh = begin
        @devzeros typeof(grid.device) eltype(grid) (grid.nx, grid.ny) hs
		hs[1:width,1:height] .= 1
		grid.rfftplan * hs
	end

	Γh = device_array(grid)(ζh .* hsh) * (grid.dx * grid.dy)
	return grid.rfftplan \ Γh
end

" Compute the radial spectrum of enstrophy."
function enstrophySpectrum(ζh, grid)
	Ωh = device_array(grid)(abs2.(ζh) / (grid.nx * grid.ny)^2 / 2)
	return FourierFlows.radialspectrum(Ωh, grid; refinement = 1)
end

" Compute the radial spectrum of energy."
function energySpectrum(ζh, grid)
	Ωh = device_array(grid)(abs2.(ζh) .* grid.invKrsq / (grid.nx * grid.ny)^2 / 2)
	return FourierFlows.radialspectrum(Ωh, grid; refinement = 1)
end

" Compute the radial flux of enstrophy."
function radialEnstrophyFlux(ζh, grid)
	ζx = grid.rfftplan \ (@. im * grid.kr * ζh)
	ζy = grid.rfftplan \ (@. im * grid.l * ζh)
	u = grid.rfftplan \ (@. im * grid.l * grid.invKrsq * ζh)
	v = grid.rfftplan \ (@. -im * grid.kr * grid.invKrsq * ζh)
	Nh = grid.rfftplan * (@. u * ζx + v * ζy)
	fh = real.(Nh .* conj(ζh)) / (grid.nx * grid.ny)^2
	kr, fhr = FourierFlows.radialspectrum(fh, grid; refinement = 1)
	return (kr, cumsum(vec(fhr)) .* step(kr))
end

" Compute the radial flux of energy."
function radialEnergyFlux(ζh, grid)
	ζx = grid.rfftplan \ (@. im * grid.kr * ζh)
	ζy = grid.rfftplan \ (@. im * grid.l * ζh)
	u = grid.rfftplan \ (@. im * grid.l * grid.invKrsq * ζh)
	v = grid.rfftplan \ (@. -im * grid.kr * grid.invKrsq * ζh)
	Nh = grid.rfftplan * (@. u * ζx + v * ζy)
	fh = real.(Nh .* conj(ζh) .* grid.invKrsq) / (grid.nx * grid.ny)^2
	kr, fhr = FourierFlows.radialspectrum(fh, grid; refinement = 1)
	return (kr, cumsum(vec(fhr)) .* step(kr))
end

" Estimate the probability distribution function from a given array of samples (CPU ver.) "
function getPDF(samples::Array, bin)
	hist = zeros(length(bin)-1)

	for sp in samples
		idx = floor(Int, (sp - first(bin)) / step(bin)) + 1
		if 1 ≤ idx ≤ length(bin) - 1
			hist[idx] += 1
		end
	end

	return hist / (step(bin) * length(samples))
end

" Estimate the probability distribution function from a given array of samples (GPU ver.) "
function getPDF(samples::CuArray, bin)
	hist = CUDA.zeros(length(bin)-1)

	threads_per_block = 256

	blocks_per_grid = cld(length(samples), threads_per_block)

	@cuda threads = threads_per_block blocks = blocks_per_grid histogram_kernel!(
		hist, samples, first(bin), step(bin), length(bin)-1,
	)

	return hist / (step(bin) * length(samples))
end

function histogram_kernel!(hist, data, min_val, bin_width, nbins)
	idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	if idx <= length(data)
		value = data[idx]
		if value >= min_val && value <= min_val + bin_width * nbins
			bin_idx = Int(floor((value - min_val) / bin_width)) + 1
			bin_idx = max(1, min(bin_idx, nbins))
			CUDA.@atomic hist[bin_idx] += 1
		end
	end
	return nothing
end

" Struct defining the postprocess data. "
struct PostData
	" Source data (the vorticity fields). "
	ζhs::Any
	" The destination of the output data. "
	des::HDF5.Group
	" Method deriving the output data from the sourece data: method(ζhs,des)."
	method::Function
	function PostData(method, src::HDF5.File, des::HDF5.Group)
		ζhs = Iterators.map(src) do ds
			ζh = read(ds)
			device_array(dev)(ζh)
		end
		return new(ζhs, des, method)
	end
end

" Collect the output data. "
collect(pd::PostData) = pd.method(pd.ζhs, pd.des)

