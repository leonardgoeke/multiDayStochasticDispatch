using AnyMOD, Gurobi
using DataFrames, CSV, Flux, Random, Plots, Statistics, Dates, Interpolations, LinearAlgebra
using Clustering, Distances, MultivariateStats, Serialization, StatsBase, Suppressor, FFTW
import AnyMOD.getAncestors, AnyMOD.lookupTupleTree, AnyMOD.lookupString, AnyMOD.addLossesExc

include("code/objects.jl")
include("code/diffusionModel.jl")
include("code/energyModel.jl")
include("code/rollingHorizon.jl")
include("code/scenarioTree.jl")

if isempty(ARGS)
    id_int = 1
else
    id_int = parse(Int,ARGS[1])
end

# ! settings
par_df = CSV.read("settings_mainScript.csv", DataFrame)

# planning
global plan_str = string(par_df[id_int,:planStr])
global eeShare = string(par_df[id_int,:eeShare])
global reRun = par_df[id_int,:reRun]
global grid = "ntc"

# dispatch
global run_str = string(par_df[id_int,:name])
global yRun = par_df[id_int,:yRun]

global dCnt = 2
global setups_dic = Dict(:stoch => ((60,8),(60,7),(60,6),(60,5),(60,4),(60,3),), :det => (8,7,6,5,4,3))

global lvlMax_int = max(maximum(setups_dic[:det]), maximum(s[2] for s in setups_dic[:stoch]))
global pfStDir_str = "intermediate/storage_pf_" * plan_str

# forecast error
global T_int = par_df[id_int,:accuracy]
global p_fl  = 0.95

# performance
global dimSteps = 0.025
global numFrc = 5

# ! prepare dispatch
if reRun
    runPlanning()
end

preparations()

# one diffusion model per planning case, so runs that only differ in forecast accuracy share the same generator
if "diffusion_" * plan_str * ".mod" in readdir("intermediate/diffusion")
    global dif_mod = deserialize("intermediate/diffusion/diffusion_" * plan_str * ".mod")
else
    global dif_mod = trainDiffusion()
    serialize("intermediate/diffusion/diffusion_" * plan_str * ".mod", dif_mod)
end

# ! generate forecasts data

# history for starting, uses mean to prevent bias
hisMats_arr = [vec(data_mat[(i+1)*8760-contDim_int+1:(i+1)*8760, :]) for i in 0:length(y_arr)-1]
global his_arr = Float32.(mean(hcat(hisMats_arr...), dims=2)[:, 1])

# generate perfect foresight data and validate
global pfData_arr
pfData_arr, ~, ~ = compilePfForecast();
pfValidation(pfData_arr, run_str);

# build deterministic and stochastic forecasts for rolling horizonr
frcNodes_int = length(pfData_arr) - lvlMax_int

global detData_arr   = Vector{Dict{Int, nodeData}}(undef, frcNodes_int)
global stochData_dic = Dict(s => Vector{Dict{Int, nodeData}}(undef, frcNodes_int) for s in setups_dic[:stoch])
global stochTree_dic = Dict(s => Vector{Vector{Tuple}}(undef, frcNodes_int) for s in setups_dic[:stoch])
detErrNode_arr       = Vector{Vector{Tuple{Int,Int, Vector{Float32}, Vector{Float32}}}}(undef, frcNodes_int)

his0_arr = copy(his_arr)
blasThreads_int = BLAS.get_num_threads()
Threads.nthreads() > 1 && BLAS.set_num_threads(1)

Threads.@threads for i in 1:frcNodes_int
    cumDay_int = (i - 1) * daysPerNode_int + 1
    d = mod1(cumDay_int, 365)
    y = (cumDay_int - 1) ÷ 365 + 1
    println("node $(i) of $(frcNodes_int) $((d,y)) on thread $(Threads.threadid())")

    # create forecasts based on the already-realized data for this node, deterministic and all stochastic setups together
    det_nd, detErr_arr, stochRes_dic = compileForecasts(d, y, hisBeforeNode(his0_arr, i), pfData_arr[i], i)
    detData_arr[i]    = det_nd
    detErrNode_arr[i] = detErr_arr
    for s in setups_dic[:stoch]
        stochData_dic[s][i], stochTree_dic[s][i] = stochRes_dic[s]
    end
end

BLAS.set_num_threads(blasThreads_int)
trackDetErr_arr = reduce(vcat, detErrNode_arr; init = Tuple{Int,Int, Vector{Float32}, Vector{Float32}}[])

plotDetErr(trackDetErr_arr, run_str)

# ! deterministic simulation on pre-generated forecast data
for l in setups_dic[:det]
    println("det $(l)")
    name_str  = "det_" * string(l) * "_" * run_str
    det_mod, ~ = createDispatch(string("modelData/dispatch_", l, "lvl_", dCnt, "d"), string("det_",l, "lvl"), l, name_str)
    track_det  = trackItr(name_str)

    nameId_det_arr = [("l$(k)s1", k) for k in 1:l]
    for i in eachindex(detData_arr)
        d = track_det.d
        setResidualDemand!(det_mod, detData_arr[i], nameId_det_arr)
        lvlDet_df = runDispatch!(det_mod, track_det)
        updateDisp(det_mod, lvlDet_df, l, d)
    end

    CSV.write("results/trackStLvl_$(track_det.name).csv", track_det.stLvl)
    CSV.write("results/overallRes_$(track_det.name).csv", track_det.overallRes)
end

# ! stochastic simulation on pre-generated forecast data, create folders for unique scenario setups
foreach(y -> rm("modelData/scenarios/" * y, recursive = true), filter(x -> occursin("stoch_" * run_str, x), readdir("modelData/scenarios")))

for (n, l) in setups_dic[:stoch]

    println("stoch ($(n), $(l))")

    name_str = "stoch_" * string(n) * "_" * string(l) * "_" * run_str
    track_stoch = trackItr(name_str)

    # loop over scenario setups to create model and assignment dic
    scrCases_arr = unique(stochTree_dic[(n,l)])
    stochMod_dic = Dict{Int,anyModel}()
    nameId_dic = Dict{Int,Vector{Tuple{String,Int}}}()
    varToScrDic_dic = Dict{Int,Dict{Symbol, Dict{VariableRef, Int64}}}()

    for s in 1:length(scrCases_arr)

        scr_str = string(n) * "node" * string(l) * "lvl_" * string(s)
        # create folder with file
        stoch_str = "stoch_" * run_str * "_" * scr_str
        mkdir("modelData/scenarios/" * stoch_str)
        nameId_dic[s] = writeScenarioSet(scrCases_arr[s], l, "modelData/scenarios/" * stoch_str)

        # create model and assignment dic
        stoch_mod, ass_dic = createDispatch(string("modelData/dispatch_", l, "lvl_", dCnt, "d"), stoch_str, l, stoch_str)
        
        stochMod_dic[s] = stoch_mod
        varToScrDic_dic[s] = ass_dic
    end

    for i in eachindex(stochData_dic[(n,l)])
        println("stoch ($(n), $(l)) node $(i)")
        d = track_stoch.d
        # run on current stochastic model
        s_int = findfirst(==(stochTree_dic[(n,l)][i]), scrCases_arr)
        scrEntr_dic = setResidualDemand!(stochMod_dic[s_int], stochData_dic[n,l][i], nameId_dic[s_int])
        updateScrProb!(stochMod_dic[s_int], stochData_dic[n,l][i], varToScrDic_dic[s_int], scrEntr_dic)
        lvlStoch_df = runDispatch!(stochMod_dic[s_int], track_stoch)
        # update next stochastic model for next iteration
        if i < length(stochData_dic[(n,l)])
            sNext_int = findfirst(==(stochTree_dic[(n,l)][i+1]), scrCases_arr)
            updateDisp(stochMod_dic[sNext_int], lvlStoch_df, l, d)            
        end
    end

    CSV.write("results/trackStLvl_$(track_stoch.name).csv", track_stoch.stLvl)
    CSV.write("results/overallRes_$(track_stoch.name).csv", track_stoch.overallRes)
end

# ! perfect foresight simulation on pre-generated forecast data
pfHorizon_int = (yRun * 365) ÷ daysPerNode_int
runPf(yRun, pfData_arr[1:pfHorizon_int], String(run_str))

plotDetErr(trackDetErr_arr, run_str)
