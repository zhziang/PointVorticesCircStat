using FourierFlows, CUDA, HDF5

include("utils/pv2field.jl")
include("utils/fieldanalysis.jl")
include("utils/postprocess.jl")


basepath = dirname(@__DIR__)

srcFiles = Dict([
	:VortexPair => h5open(basepath * "/.output/N256H-0.001543.h5", "r");
	:MaxEntropy => h5open(basepath * "/.output/N256H-0.000917.h5", "r");
	:Condensate => h5open(basepath * "/.output/N256H0.0.h5", "r")
])
desFile = h5open(basepath * "/postprocess.h5", "w")

ngrid = 256

grid = TwoDGrid(GPU(); nx = ngrid, Lx = 1)


