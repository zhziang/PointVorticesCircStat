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

