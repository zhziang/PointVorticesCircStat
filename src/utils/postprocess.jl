# Data IO

function preOperation(file)
	circs = read(file["circulations"])
	ζhs = Iterators.map(1:1000) do n
		u = file["pv positions"][:, :, n]
		grid.rfftplan * pv2field(u, circs, grid)
	end
	return ζhs
end

function saveData(dict, root::HDF5.Group)
	for name in keys(dict)
		root[name] = dict[name]
	end
	return nothing
end

# Analysed quantities

function energySpectrum(ζhs, grid)
	kr, _ = energySpectrum(first(ζhs), grid)

	mean = sum(ζhs) do ζh
		_, Ehr = energySpectrum(ζh, grid)
		Ehr
	end ./ length(ζhs)

	var = sum(ζhs) do ζh
		_, Ehr = energySpectrum(ζh, grid)
		@. (Ehr - mean) ^ 2
	end ./ length(ζhs)

	return Dict([
		"var" => Array(kr)
		"val" => Array(mean)
		"std" => Array(sqrt.(var))
	])
end

function energyFlux(ζhs, grid)
	kr, _ = radialEnergyFlux(first(ζhs), grid)

	mean = sum(ζhs) do ζh
		_, Ehr = radialEnergyFlux(ζh, grid)
		Ehr
	end ./ length(ζhs)

	var = sum(ζhs) do ζh
		_, Ehr = radialEnergyFlux(ζh, grid)
		@. (Ehr - mean) ^ 2
	end ./ length(ζhs)

	return Dict([
		"var" => Array(kr)
		"val" => Array(mean)
		"std" => Array(sqrt.(var))
	])
end

function enstrophyFlux(ζhs, grid)
	kr, _ = radialEnstrophyFlux(first(ζhs), grid)

	mean = sum(ζhs) do ζh
		_, Ehr = radialEnstrophyFlux(ζh, grid)
		Ehr
	end ./ length(ζhs)

	var = sum(ζhs) do ζh
		_, Ehr = radialEnstrophyFlux(ζh, grid)
		@. (Ehr - mean) ^ 2
	end ./ length(ζhs)

	return Dict([
		"var" => Array(kr)
		"val" => Array(mean)
		"std" => Array(sqrt.(var))
	])
end

function rectSizeCircMoments(ζhs, grid; width = 10, height = 10, order = 2)
	ns = 1:floor(Int, min(grid.nx/width, grid.ny/height))

	moments = map(ns) do n
		mean = sum(ζhs) do ζh
			Γ = rectCirculations(ζh .* fh, n*width, n*height, grid)
			sum(x->abs(x)^order, Γ) / length(Γ)
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			Γ = rectCirculations(ζh .* fh, n*width, n*height, grid)
			res = sum(x->abs(x)^order, Γ) / length(Γ)
			@. (res - mean) ^ 2
		end ./ length(ζhs)

		(mean, var)
	end

	return Dict([
		"var" => Array(ns)
		"val" => Array(getindex.(moments, 1))
		"std" => Array(sqrt.(getindex.(moments, 1)))
	])
end

function aspectRatioCircMomentsArea(ζhs, grid; order = 2, loopsize = 10)
	ns = [m for m in 1:loopsize if (mod(loopsize^2, m) == 0) && loopsize^2 < m*grid.nx]

	moments = map(ns) do n
		mean = sum(ζhs) do ζh
			Γ = rectCirculations(ζh .* fh, n, loopsize^2 ÷ n, grid)
			sum(x->abs(x)^order, Γ) / length(Γ)
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			Γ = rectCirculations(ζh .* fh, n, loopsize^2 ÷ n, grid)
			res = sum(x->abs(x)^order, Γ) / length(Γ)
			@. (res - mean) ^ 2
		end ./ length(ζhs)

		(mean, var)
	end

	return Dict([
		"var" => Array(ns)
		"val" => Array(getindex.(moments, 1))
		"std" => Array(sqrt.(getindex.(moments, 1)))
	])
end

function aspectRatioCircMomentsPerimeter(ζhs, grid; order = 2, loopsize = 10)
	c = ceil(Int, loopsize/10)
	ns = c:c:loopsize

	moments = map(ns) do n
		mean = sum(ζhs) do ζh
			Γ = rectCirculations(ζh .* fh, n, 2loopsize - n, grid)
			sum(x->abs(x)^order, Γ) / length(Γ)
		end ./ length(ζhs)

		var = sum(ζhs) do ζh
			Γ = rectCirculations(ζh .* fh, n, 2loopsize - n, grid)
			res = sum(x->abs(x)^order, Γ) / length(Γ)
			@. (res - mean) ^ 2
		end ./ length(ζhs)

		(mean, var)
	end

	return Dict([
		"var" => Array(ns)
		"val" => Array(getindex.(moments, 1))
		"std" => Array(sqrt.(getindex.(moments, 1)))
	])
end




