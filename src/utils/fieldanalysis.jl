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
