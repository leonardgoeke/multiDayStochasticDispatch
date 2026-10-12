
function runPlanning()

    # make sure to delete files to be replaced
    if isdir("intermediate/capacity_" * plan_str) rm("intermediate/capacity_" * plan_str, recursive = true) end
    if isdir("intermediate/storage_" * plan_str) rm("intermediate/storage_" * plan_str, recursive = true) end
    for d in filter(x -> startswith(x, "storage_" * plan_str * "_"), readdir("intermediate"))
        rm("intermediate/" * d, recursive = true)
    end
    if isfile("intermediate/diffusion/diffusion_" * plan_str * ".mod") rm("intermediate/diffusion/diffusion_" * plan_str * ".mod", recursive = true) end
    if isdir(pfStDir_str) rm(pfStDir_str, recursive = true) end

    # relevant directories
    inDir_arr = ["modelData/basis", "modelData/eeShare/" * eeShare, "modelData/scenarios/20scr", "modelData/planning", "modelData/spatialScope/3Ctr", "modelData/" * grid]
  
    resultDir_str = "results/"

    # solve planning model
    anyM = anyModel(inDir_arr, resultDir_str, objName = "planning_" * plan_str, supTsLvl = 1, repTsLvl = 3, frsLvl = 2, shortExp = 5, emissionLoss = false, holdFixed = true);
    createOptModel!(anyM)
    if grid == "ptdf" addPtdf!(anyM, "modelData/ptdf_3ctr.csv") end
    setObjective!(:cost, anyM)

    set_optimizer(anyM.optModel, Gurobi.Optimizer)
    set_optimizer_attribute(anyM.optModel, "Crossover", 0);
    set_optimizer_attribute(anyM.optModel, "Method", 2);

    optimize!(anyM.optModel)
    
    sum_df = reportResults(:summary, anyM; rtnOpt = (:rawDf,:csv))
    reportResults(:cost, anyM)

    # write total quantities to compute dayahead error    
    parDef_dic = defineParameter(anyM.options, anyM.report)

    mkdir("intermediate/capacity_" * plan_str)

    # write relevant energy quantities for dayahead error computation
    enQant_df =copy(sum_df)
    enQant_df[!,:Te] = map(x -> anyM.sets[:Te].nodes[x].val, enQant_df[!,:Te])
    enQant_df[!,:R] = map(x -> anyM.sets[:R].nodes[x].val, enQant_df[!,:R_dis])

    enQant_df = filter(x -> x.Te in ("wind", "pv") || (x.variable in (:demand, :gen) && x.Te != "ccgt"), enQant_df)
    enQant_df = combine(x -> (value = abs(sum(x.value)),), groupby(enQant_df, [:Te, :R]))

    enQant_df[!,:Te] = map(x -> x == "none" ? "demand" : x, enQant_df[!,:Te])
    CSV.write("intermediate/capacity_" * plan_str * "/energyQuant.csv", rename(enQant_df, :Te => :technology, :R => :region))

    # write results as a capacity fix
    for z in ["capaConv", "capaStIn", "capaStOut", "capaStSize"]
        capa_df = rename(select(filter(x -> x.variable == Symbol(z), sum_df), [:R_dis,:Te,:value]), :R_dis => :R_exp)
        if !isempty(capa_df)
            writeParameterFile!(anyM, capa_df, Symbol(z,:Resi), parDef_dic[Symbol(z,:Resi)], "intermediate/capacity_" * plan_str * "/par_" * z)
        end
    end

    # write dual values and levels for storage
    allVar_df = DataFrame(Ts_expSup = Int[], Ts_disSup = Int[], Ts_dis = Int[], R_dis = Int[], C = Int[], Te = Int[], M = Int[], scr = Int[], id = Int[], value =Float64[])
    allCns_df = DataFrame(Ts_expSup = Int[], Ts_dis = Int[], R_dis = Int[], C = Int[], Te = Int[], M = Int[], scr = Int[], id = Int[], dual =Float64[])
    for tSym in keys(anyM.parts.tech)
        if :stBal in keys(anyM.parts.tech[tSym].cns)
            # write dual values
            cns_df = anyM.parts.tech[tSym].cns[:stBal]
            cns_df[!,:dual] = map(x -> abs(dual(x)) / anyM.options.scaFac.obj * getScaFac(x, anyM.options.scaFac.dispSt, "stLvl"), cns_df[!,:cns])
            append!(allCns_df, select(cns_df, Not([:cns])))
            # write levels
            var_df = anyM.parts.tech[tSym].var[:stLvl]
            var_df[!,:value] = map(x -> -1 * value(x), var_df[!,:var])
            append!(allVar_df, select(var_df, Not([:var])))
        end
    end

    mkdir("intermediate/storage_" * plan_str)
    mkdir(pfStDir_str)

    # aggregate duals across scenarios
    dualCns_df = combine(x -> (dual = -sum(x.dual)*1000,), groupby(allCns_df, [:Ts_dis, :R_dis, :Te]))
    dualCns_df = printObject(dualCns_df, anyM, fileName = "dualCns", rtnDf =(:csvDf,))
    dualCns_df[!,:timestep_dispatch] = map(x -> parse(Int, replace(split(x," < ")[end], " " => "", "h" => "")), dualCns_df[!,:timestep_dispatch])

    CSV.write("intermediate/storage_" * plan_str * "/dualLevels.csv", rename(dualCns_df, Dict(:timestep_dispatch => :timestep, :region_dispatch => :region, :technology => :technology)))

    # write starting levels for storage
    select!(allVar_df, [:Ts_dis, :R_dis, :Te, :value])
    minTs_int = minimum(allVar_df[!,:Ts_dis])
    allVar_df = combine(x -> (value = sum(x.value)/length(x.value),), groupby(filter(x -> x.Ts_dis == minTs_int, allVar_df), [:R_dis, :Te]))
    writeParameterFile!(anyM, allVar_df, :stLvlStart, parDef_dic[:stLvlStart], "intermediate/storage_" * plan_str * "/par_stLvlStart")
    writeParameterFile!(anyM, allVar_df, :stLvlStart, parDef_dic[:stLvlStart], pfStDir_str * "/par_stLvlStart")

end

function createDispatch(treeDir_str::String, scr_dir::String, l_int::Int, name_str::String)

    # the end-of-horizon storage value depends on how far out the horizon reaches, so each length needs its own folder
    lvlStorDir_str = "intermediate/storage_" * plan_str * "_" * string(l_int)

    if !isdir(lvlStorDir_str)
        mkpath(lvlStorDir_str)
        cp("intermediate/storage_" * plan_str * "/par_stLvlStart.csv", lvlStorDir_str * "/par_stLvlStart.csv")
        dual_df = CSV.read("intermediate/storage_" * plan_str * "/dualLevels.csv", DataFrame)
        dual_df = filter(x -> x.timestep == l_int * dataDim_int, dual_df)
        dual_df[!, :timestep] .= string("h",lpad(l_int * dataDim_int, 4, "0"))
        dual_df[!, :parameter] .= "costVarStLvl"
        CSV.write(lvlStorDir_str * "/par_dualLevels.csv", rename(dual_df, :timestep => :timestep_4, :region => :region_1, :technology => :technology_1, :dual => :value))
    end

    # relevant directories
    inDir_arr = ["modelData/basis", treeDir_str, "modelData/spatialScope/3Ctr", "intermediate/capacity_" * plan_str, lvlStorDir_str, "modelData/scenarios/" * scr_dir, "modelData/" * grid]

    resultDir_str = "results/"

    # create dispatch model
    anyM = anyModel(inDir_arr, resultDir_str, objName = "operation_" * name_str, numYears = l_int * daysPerNode_int  / 365, supTsLvl = 1, repTsLvl = 3, frsLvl = l_int == 1 ? 0 : 3, shortExp = 5, reportLvl = 0, errCheckLvl = 0, errWrtLvl = 0, emissionLoss = false, holdFixed = true);
    createOptModel!(anyM)
    if grid == "ptdf" addPtdf!(anyM, "modelData/ptdf_3ctr.csv") end
    setObjective!(:cost, anyM)

    # manipulations for rolling horizon dispatch
    addScaFac!(anyM)
    varToScr_dic = createVarScrMap(anyM)

    # one Gurobi environment reused per thread -- see the matching note in buildAbsPath
    env = get!(() -> Gurobi.Env(), task_local_storage(), :gurobiEnv)
    set_optimizer(anyM.optModel, () -> Gurobi.Optimizer(env))
    set_silent(anyM.optModel)
    set_optimizer_attribute(anyM.optModel, "Crossover", 0);
    set_optimizer_attribute(anyM.optModel, "Method", 2);

    return anyM, varToScr_dic
    
end

function runPf(yPfFrs_int::Int, pfData_arr::Vector{nodeData}, name_str::String)

    treeDir_str = "modelData/dispatch_" * string(yPfFrs_int) *"y_pf"

    # create file for dual on level
    dual_df = CSV.read("intermediate/storage_" * plan_str * "/dualLevels.csv", DataFrame)
    dual_df = filter(x -> x.timestep == 8760, dual_df)
    dual_df[!, :timestep_3] .= "d" * lpad(365*yPfFrs_int,4,"0")
    dual_df[!, :timestep_4] .= "h8760"
    dual_df[!, :parameter] .= "costVarStLvl"
    CSV.write(pfStDir_str * "/par_dualLevels.csv", rename(select(dual_df, Not([:timestep])), :region => :region_1, :technology => :technology_1, :dual => :value))

    # relevant directories
    inDir_arr = ["modelData/basis", treeDir_str, "modelData/spatialScope/3Ctr", "intermediate/capacity_" * plan_str, pfStDir_str, "modelData/scenarios/det_1lvl" , "modelData/" * grid]
    
    resultDir_str = "results/"

    # create dispatch model
    anyM = anyModel(inDir_arr, resultDir_str, objName = "operationPf_" * name_str, numYears = yPfFrs_int, supTsLvl = 1, repTsLvl = 3, frsLvl = 0, shortExp = 5, reportLvl = 0, errCheckLvl = 0, errWrtLvl = 0,emissionLoss = false, holdFixed = true);
    createOptModel!(anyM)
    if grid == "ptdf" addPtdf!(anyM, "modelData/ptdf_3ctr.csv") end
    setObjective!(:cost, anyM)

    # one Gurobi environment reused per thread -- see the matching note in buildAbsPath
    env = get!(() -> Gurobi.Env(), task_local_storage(), :gurobiEnv)
    set_optimizer(anyM.optModel, () -> Gurobi.Optimizer(env))
    set_silent(anyM.optModel)
    set_optimizer_attribute(anyM.optModel, "Crossover", 0);
    set_optimizer_attribute(anyM.optModel, "Method", 2);

    # ! set resdiual demand
    addScaFac!(anyM)

    # map time-step id to hour of day and to day
    tsLvl_int = anyM.sets[:Ts].height
    nodeToHour_dic = Dict{Int, Int}()
    nodeToDay_dic = Dict{Int, Int}()
    for (id, node) in anyM.sets[:Ts].nodes
        if node.lvl == tsLvl_int
            nodeToHour_dic[id] = parse(Int, node.val[2:end]) |> (h -> mod1(h, 24))
        elseif node.lvl == tsLvl_int - 1
            nodeToDay_dic[id] = parse(Int, node.val[2:end])
        end
    end

    # map region id to region symbol
    nodeToReg_dic = Dict{Int, Symbol}(id => Symbol(node.val) for (id, node) in anyM.sets[:R].nodes)
    # set RHS of each energy balance constraint to actual residual demand
    pfVec_arr = map(x -> x.data, pfData_arr)
    pf_dic = Dict(x => pfVec_arr[x] for x in 1:length(pfVec_arr))
    cns_df = anyM.parts.bal.cns[:enBalElectricity]
    cns_df[!, :day] = map(x -> nodeToDay_dic[getAncestors(x.Ts_dis, anyM.sets[:Ts], :int, tsLvl_int-1)[end]], eachrow(cns_df))

    resDm_fl = 0.0
    for s in groupby(cns_df, :day)
        data_obj = pf_dic[cld(s.day[1], daysPerNode_int)]
        # offset of this day within its node, taken from the absolute day so it stays in phase across year boundaries
        dayOff_int = (mod1(s.day[1], daysPerNode_int) - 1) * 24
        data_df = stack(denormalize(data_obj, ext_dic, reg_arr, dataDim_int, numReg_int), Symbol.(reg_arr))
        resDm_fl = resDm_fl + sum(data_df[!, :value]) / daysPerNode_int
        data_dic = Dict{Tuple{Int, Symbol}, Float64}((row.hour, Symbol(row.variable)) => row.value for row in eachrow(data_df))
        foreach(r -> set_normalized_rhs(r.cns, r.scaFac * data_dic[(dayOff_int + nodeToHour_dic[r.Ts_dis], nodeToReg_dic[r.R_dis])]), eachrow(s))
    end
    resDm_fl = resDm_fl / 1000 / yPfFrs_int

    optimize!(anyM.optModel)

    reportResults(:summary, anyM)
    reportResults(:cost, anyM)

    return anyM, resDm_fl

end

function addScaFac!(anyM::anyModel)
    cns_df = anyM.parts.bal.cns[:enBalElectricity]
    cns_df[!,:scaFac] = map(x -> getScaFac(x, anyM.options.scaFac.dispSt, "stExt"), cns_df[!,:cns])
end

function getScaFac(cns_ref::ConstraintRef, dispFac_fl::Float64, var_str::String)
    cns_dic = constraint_object(cns_ref).func.terms
    key_var = filter(x -> occursin(var_str, string(x)), collect(keys(cns_dic)))[1]
    return abs(cns_dic[key_var] / dispFac_fl)
end

# maps variable reference to scenario id for later manipulation with case-specfic scenario probabilities
function createVarScrMap(disTree_mod::anyModel)

    scrStr_arr = filter(x -> x != "none", map(x -> x.val, collect(values(disTree_mod.sets[:scr].nodes))))
    varToScr_dic = Dict{Symbol,Dict{VariableRef, Int}}()

    # loop-over cost constraints
    for c in keys(disTree_mod.parts.cost.cns)
        cVarToScr_dic = Dict{VariableRef, Int}()
        # map variable ref to AnyMOD scenario id
        var_arr = unique(vcat(map(x -> collect(keys(constraint_object(x).func.terms)), disTree_mod.parts.cost.cns[c][!,:cns])...))
        for x in var_arr
            matchStr_arr = filter(s -> occursin(s, string(x)), scrStr_arr)
            if !isempty(matchStr_arr) cVarToScr_dic[x] = lookupString(matchStr_arr[1], disTree_mod.sets[:scr]) end
        end

        # save dictionary if any data was written
        if !isempty(cVarToScr_dic)
            varToScr_dic[c] = cVarToScr_dic
        end
    end

    return varToScr_dic

end

# get storage levels at end of first day to set as starting levels for rolling horizon dispatch
function getNewStLvl(disTree_mod::anyModel)

    # get storage levels at end of first day
    allVar_df = DataFrame(Ts_expSup = Int[], Ts_disSup = Int[], Ts_dis = Int[], R_dis = Int[], C = Int[], Te = Int[], M = Int[], scr = Int[], id = Int[], value =Float64[])
    for tSym in keys(disTree_mod.parts.tech)
        if :stBal in keys(disTree_mod.parts.tech[tSym].cns)
            # write levels
            var_df = disTree_mod.parts.tech[tSym].var[:stLvl]
            # arbitrary small storage level instead of zero, since the correction below multiplies by it
            var_df[!,:value] = map(x -> max(1e-6, value(x)), var_df[!,:var])
            append!(allVar_df, select(var_df, Not([:var])))
        end
    end

    firstScr_df = filter(x -> x.scr == 1, allVar_df)
    maxTs_int = maximum(firstScr_df[!,:Ts_dis])
    firstScr_df = filter(x -> x.Ts_dis == maxTs_int, firstScr_df)

    return firstScr_df

end

# run dispatch, track iteration and return storage levels
function runDispatch!(dis_mod::anyModel,track_obj::trackItr)
    optimize!(dis_mod.optModel)
    if MOI.get(dis_mod.optModel, MOI.ConflictStatus()) == MOI.ConflictStatusCode(3)
        printIIS(dis_mod)
        error("Infeasible model")
    end
    trackIteration!(track_obj, dis_mod)
    return getNewStLvl(dis_mod)
end

function addPtdf!(anyM::anyModel, ptdfFile_str::String)

    # compute net-injections
    exc_df = anyM.parts.exc[:powerGrid].var[:exc]
    allNetInj_df = DataFrame(Ts_dis = Int[], scr = Int[], R = Int[], netVar = AffExpr[])
    for r in union(unique(exc_df[!,:R_from]), unique(exc_df[!,:R_to]))
        netExc_df = filter(x -> x.R_from == r || x.R_to == r, exc_df)
        netExc_df = combine(x -> (netVar = sum(x.var .* map(y -> y == r ? 1 : -1, x.R_from)),), groupby(netExc_df, [:Ts_dis, :scr]))
        netExc_df[!,:R] .= r
        append!(allNetInj_df, netExc_df)
    end

    # create ptdf-constraint
    ptdf_df = rename(select(CSV.read(ptdfFile_str, DataFrame),Not([:capacity_mw])), Dict(:bus0 => :R_from, :bus1 => :R_to))

    # convert for constraint generation
    ptdf_df[!,:R_from] = map(x -> lookupString(x,anyM.sets[:R]), ptdf_df[!,:R_from])
    ptdf_df[!,:R_to] = map(x -> lookupString(x,anyM.sets[:R]), ptdf_df[!,:R_to])
    ctr_col = filter(x -> !(x in ("R_from","R_to")), names(ptdf_df))
    ptdf_df[!,:ptdf] = map(x -> Dict(lookupString(c,anyM.sets[:R]) => x[c] for c in ctr_col), eachrow(ptdf_df))
    ptdf_dic = Dict((x.R_from, x.R_to) => x.ptdf for x in eachrow(ptdf_df))


    # get all relevant net-injection/lines combinations
    lines_arr =map(x -> (x.R_from ,x.R_to), eachrow(ptdf_df))
    allNetInj_df[!,:lines] .= [lines_arr]
    netVarLines_df = flatten(allNetInj_df, :lines)

    # compute product of flow and PTDF for each line
    netVarLines_df[!,:fac] .= map(x -> ptdf_dic[(x.lines[1], x.lines[2])][x.R], eachrow(netVarLines_df))
    netVarLines_df[!,:flowVar] .= map(x -> x.fac * x.netVar, eachrow(netVarLines_df))
    filter!(x -> x.flowVar != AffExpr(), netVarLines_df)

    # aggregate by line
    ptdf_df = combine(x -> (flowVar = sum(x.flowVar),), groupby(netVarLines_df, [:lines, :Ts_dis, :scr]))
    ptdf_df[!,:R_from] .= map(x -> x.lines[1], eachrow(ptdf_df))
    ptdf_df[!,:R_to] .= map(x -> x.lines[2], eachrow(ptdf_df))

    # join with line variable
    cns_df = innerjoin(ptdf_df, exc_df, on = [:R_from, :R_to, :Ts_dis, :scr])
    cns_df[!,:cns] = map(x -> @constraint(anyM.optModel, x.flowVar == x.var), eachrow(cns_df))

    anyM.parts.exc[:powerGrid].cns[:ptdf] = select(cns_df, [:Ts_dis, :scr, :R_from, :R_to, :C, :Exc, :cns])

end
