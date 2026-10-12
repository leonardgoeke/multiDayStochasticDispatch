
# use forecasts to set residual demand in energy balance constraints of dispatch model
function setResidualDemand!(dis_mod::anyModel, node_dic::Dict{Int, nodeData}, nameId_arr::Vector{Tuple{String,Int}})

    cns_df = dis_mod.parts.bal.cns[:enBalElectricity]

    # time-step mapping
    tsLvl_int = dis_mod.sets[:Ts].height
    nodeToHour_dic = Dict{Int, Int}()
    for (id, node) in dis_mod.sets[:Ts].nodes
        node.lvl != tsLvl_int && continue
        nodeToHour_dic[id] = parse(Int, node.val[2:end]) |> (h -> mod1(h, dataDim_int))
    end
    nodeToReg_dic = Dict{Int, Symbol}(id => Symbol(node.val) for (id, node) in dis_mod.sets[:R].nodes)

    # scenario id in model to node id in data mapping
    allScr_arr  = filter(x -> x.val != "none", collect(values(dis_mod.sets[:scr].nodes)))
    nodeNameId_dic = Dict(x.val => x.idx for x in allScr_arr)
    scrEntr_dic = Dict(nodeNameId_dic[x[1]] => x[2] for x in nameId_arr)

    for s in groupby(cns_df, :scr)
        data_obj = node_dic[scrEntr_dic[s.scr[1]]]
        data_df  = stack(denormalize(data_obj.data, ext_dic, reg_arr, dataDim_int, numReg_int), Symbol.(reg_arr))
        data_lkp = Dict{Tuple{Int,Symbol},Float64}((row.hour, Symbol(row.variable)) => row.value for row in eachrow(data_df))
        foreach(r -> set_normalized_rhs(r.cns, r.scaFac * data_lkp[(nodeToHour_dic[r.Ts_dis], nodeToReg_dic[r.R_dis])]), eachrow(s))
    end

    return scrEntr_dic

end

# update scenario probability weights in cost constraints based on marginal probabilities from the data tree
function updateScrProb!(dis_mod::anyModel, node_dic::Dict{Int, nodeData}, varToScr_dic::Dict{Symbol, Dict{VariableRef, Int64}}, scrEntr_dic::Dict{Int, Int})

    curProb_dic = Dict(x.scr => x.val for x in eachrow(unique(select(dis_mod.parts.obj.par[:scrProb].data, [:scr,:val]))))
    root_int    = filter(x -> x.lvl == 1, collect(values(node_dic)))[1].id
    marProb_dic = Dict(root_int => 1.0)
    for x in filter(x -> x != root_int, sort(collect(keys(node_dic))))
        marProb_dic[x] = marProb_dic[node_dic[x].ancestor] * node_dic[x].prob
    end

    corFacProb_dic = Dict(x => marProb_dic[scrEntr_dic[x]] / curProb_dic[x] for x in keys(curProb_dic))
    dis_mod.parts.obj.par[:scrProb].data[!,:val] = map(x -> marProb_dic[scrEntr_dic[x]], dis_mod.parts.obj.par[:scrProb].data[!,:scr])

    for c in keys(varToScr_dic)
        for cns in dis_mod.parts.cost.cns[c][!,:cns]
            cns_dic = constraint_object(cns).func.terms
            for v in keys(cns_dic)
                haskey(varToScr_dic[c], v) && set_normalized_coefficient(cns, v, cns_dic[v] * corFacProb_dic[varToScr_dic[c][v]])
            end
        end
    end

end

# update initial storage levels for the next iteration based on the previous iteration's realized end-of-day levels
function updateStLvlStart!(dis_mod::anyModel, firstScr_df::DataFrame)

    newLvl_dic = Dict((x.R_dis, x.Te) => -x.value for x in eachrow(firstScr_df))

    for tSym in keys(dis_mod.parts.tech)

        if !(:stBal in keys(dis_mod.parts.tech[tSym].cns)) || !(:stLvlStart in keys(dis_mod.parts.tech[tSym].par))
            continue
        end

        cns_df    = dis_mod.parts.tech[tSym].cns[:stBal]
        minTs_int = minimum(cns_df[!,:Ts_dis])
        start_df  = filter(x -> x.Ts_dis == minTs_int, cns_df)

        par_df      = dis_mod.parts.tech[tSym].par[:stLvlStart].data
        corFac_dic  = Dict((x.R_dis, x.Te) => newLvl_dic[(x.R_dis, x.Te)] / x.val for x in eachrow(par_df))
        foreach(r -> set_normalized_rhs(r.cns, normalized_rhs(r.cns) * corFac_dic[(r.R_dis, r.Te)]), eachrow(start_df))

        par_df[!,:val] = map(x -> newLvl_dic[(x.R_dis, x.Te)], eachrow(par_df))
    end

end

# update terminal value of storage levels at the end of the horizon based on the dual values from the original planning run
function updateCostVarStLvl!(dis_mod::anyModel, l_int::Int, d::Int)

    endDay_int = d + l_int * daysPerNode_int - 1
    hour_int   = mod1(endDay_int, 365) * 24

    dual_df      = filter(x -> x.timestep == hour_int, CSV.read("intermediate/storage_" * plan_str * "/dualLevels.csv", DataFrame))
    newCost_dic  = Dict((lookupString(x.region, dis_mod.sets[:R]), lookupString(x.technology, dis_mod.sets[:Te])) => x.dual for x in eachrow(dual_df))

    par_df     = dis_mod.parts.cost.par[:costVarStLvl].data
    corFac_dic = Dict((x.R_dis, x.Te) => newCost_dic[(x.R_dis, x.Te)] / x.val for x in eachrow(par_df))

    varToReg_dic = Dict{VariableRef, Tuple{Int,Int}}()
    for tSym in keys(dis_mod.parts.tech)
        if !(:stLvl in keys(dis_mod.parts.tech[tSym].var))
            continue
        end
        var_df    = dis_mod.parts.tech[tSym].var[:stLvl]
        maxTs_int = maximum(var_df[!,:Ts_dis])
        foreach(x -> varToReg_dic[collect(x.var.terms)[1][1]] = (x.R_dis, x.Te), eachrow(filter(x -> x.Ts_dis == maxTs_int, var_df)))
    end

    for cns in dis_mod.parts.cost.cns[:costVarStLvl][!,:cns]
        cns_dic = constraint_object(cns).func.terms
        for v in keys(cns_dic)
            if haskey(varToReg_dic, v)
                set_normalized_coefficient(cns, v, cns_dic[v] * corFac_dic[varToReg_dic[v]])
            end
        end
    end

    par_df[!,:val] = map(x -> newCost_dic[(x.R_dis, x.Te)], eachrow(par_df))

end

function analyseDispatch(dis_mod::anyModel)
    enBal_df   = filter(x -> x.scr == 1, dis_mod.parts.bal.cns[:enBalElectricity])
    resDem_fl  = sum(normalized_rhs.(enBal_df[!,:cns]))
    crt_fl     = sum(value.(enBal_df[!,:cns]) .- normalized_rhs.(enBal_df[!,:cns]))
    thermal_fl = sum(value.(filter(x -> x.scr == 1, dis_mod.parts.tech[:ccgt].var[:gen])[!,:var]))
    allExc_df = filter(x -> x.scr == 1, dis_mod.parts.exc[:powerGrid].var[:exc])
    lossExc_df =addLossesExc(allExc_df, dis_mod.parts.exc[:powerGrid], dis_mod.sets)
    excLoss_fl =sum(value.(allExc_df[!,:var])) - sum(value.(lossExc_df[!,:var]))
    lss_fl     = sum(value.(filter(x -> x.scr == 1, dis_mod.parts.bal.var[:lss])[!,:var]))
    phsIn_fl   = sum(value.(filter(x -> x.scr == 1, dis_mod.parts.tech[:phs].var[:stExtIn])[!,:var]))
    phsOut_fl  = sum(value.(filter(x -> x.scr == 1, dis_mod.parts.tech[:phs].var[:stExtOut])[!,:var]))
    liIn_fl    = sum(value.(filter(x -> x.scr == 1, dis_mod.parts.tech[:lithiumBattery].var[:stExtIn])[!,:var]))
    liOut_fl   = sum(value.(filter(x -> x.scr == 1, dis_mod.parts.tech[:lithiumBattery].var[:stExtOut])[!,:var]))
    return (crt = crt_fl, thermal = thermal_fl, lss = lss_fl, phsIn = phsIn_fl, phsOut = phsOut_fl, liIn = liIn_fl, liOut = liOut_fl, excLoss = excLoss_fl, resDem = resDem_fl)
end

# track results of iteration
function trackIteration!(track_obj::trackItr, dis_mod::anyModel)

    res_tup = analyseDispatch(dis_mod)
    track_obj.aggRes = (
        crt     = track_obj.aggRes.crt     + res_tup.crt     / 1000,
        thermal = track_obj.aggRes.thermal + res_tup.thermal / 1000,
        lss     = track_obj.aggRes.lss     + res_tup.lss     / 1000,
        phsIn   = track_obj.aggRes.phsIn   + res_tup.phsIn   / 1000,
        phsOut  = track_obj.aggRes.phsOut  + res_tup.phsOut  / 1000,
        liIn    = track_obj.aggRes.liIn    + res_tup.liIn    / 1000,
        liOut   = track_obj.aggRes.liOut   + res_tup.liOut   / 1000,
        excLoss = track_obj.aggRes.excLoss  + res_tup.excLoss  / 1000,
        resDem  = track_obj.aggRes.resDem  + res_tup.resDem  / 1000,
    )
    scaFac_fl = 365 / (track_obj.d + daysPerNode_int - 1 + 365 * (track_obj.y - 1))
    push!(track_obj.overallRes, (
        day     = track_obj.d,
        year    = track_obj.y,
        crt     = track_obj.aggRes.crt     * scaFac_fl,
        thermal = track_obj.aggRes.thermal * scaFac_fl,
        lss     = track_obj.aggRes.lss     * scaFac_fl,
        phsIn   = track_obj.aggRes.phsIn   * scaFac_fl,
        phsOut  = track_obj.aggRes.phsOut  * scaFac_fl,
        liIn    = track_obj.aggRes.liIn    * scaFac_fl,
        liOut   = track_obj.aggRes.liOut   * scaFac_fl,
        resDem  = track_obj.aggRes.resDem  * scaFac_fl,
        excLoss = track_obj.aggRes.excLoss * scaFac_fl,
        case    = track_obj.name,
    ))

    stLvl_arr = DataFrame[]
    for tSym in keys(dis_mod.parts.tech)
        if !(:stLvl in keys(dis_mod.parts.tech[tSym].var))
            continue
        end
        var_df = filter(x -> x.scr == 1, dis_mod.parts.tech[tSym].var[:stLvl])
        var_df[!,:value] = map(x -> value(x), var_df[!,:var])
        push!(stLvl_arr, select(var_df, [:Ts_dis, :R_dis, :Te, :value]))
    end
    stLvl_df = vcat(stLvl_arr...)
    stLvl_df[!,:d]    .= track_obj.d
    stLvl_df[!,:y]    .= track_obj.y
    stLvl_df[!,:Ts_dis] .= stLvl_df[!,:Ts_dis] .- 4
    stLvl_df[!,:R_dis]  .= map(x -> dis_mod.sets[:R].nodes[x].val,  stLvl_df[!,:R_dis])
    stLvl_df[!,:Te]     .= map(x -> dis_mod.sets[:Te].nodes[x].val, stLvl_df[!,:Te])
    stLvl_df[!,:case]  .= track_obj.name
    append!(track_obj.stLvl, rename(select(stLvl_df, [:d,:y,:Ts_dis,:R_dis,:Te,:value,:case]), :d => :day, :y => :year, :Ts_dis => :hour, :R_dis => :region, :Te => :technology, :value => :level))

    updateCounter!(track_obj)


    return track_obj
end

function updateDisp(dis_mod::anyModel, lvl_df::DataFrame, l_int::Int, d_int::Int)
    updateStLvlStart!(dis_mod, lvl_df)
    updateCostVarStLvl!(dis_mod, l_int, d_int)
end

function updateCounter!(track_obj::trackItr)
    newDay_int = track_obj.d + daysPerNode_int
    if newDay_int > 365
        track_obj.d = newDay_int - 365
        track_obj.y = track_obj.y + 1
    else
        track_obj.d = newDay_int
    end
end

function getSynContext(n::nodeData, synVal_int::Int, nodeData_arr::Vector{nodeData})
    synData_arr = Vector{Vector{Float32}}()
    push!(synData_arr, n.data)

    nPrev_int = n.ancestor
    while length(synData_arr) * dataDim_int < synVal_int
        nPrev_obj = filter(x -> x.id == nPrev_int, nodeData_arr)[1]
        push!(synData_arr, nPrev_obj.data)
        nPrev_int = nPrev_obj.ancestor
    end

    return vcat(reverse(synData_arr)...)[end - synVal_int * numReg_int + 1 : end]
end

function compilePfForecast()

    pfNode_arr = nodeData[]
    date_tup   = (1, 1)

    # kept per-region (not a flat vector) -- flattening would interleave incompatible region-major layouts
    hisPf_arr = [copy(his_arr[(i - 1) * contDim_int + 1 : i * contDim_int]) for i in 1:numReg_int]

    # generate 20 HSMM path candidates; pick the one with implied mean closest to the historical target
    lvlMax_int  = max(maximum(setups_dic[:det]), maximum(s[2] for s in setups_dic[:stoch]))
    pathLen_int = yRun * 365 + lvlMax_int * daysPerNode_int + 2
    candidates  = [generateForwardPath(startReg_int, startRem_int, 0, pathLen_int) for _ in 1:60]
    cand_means  = [pathMeanRD(c[1], yRun * 365) for c in candidates]
    best_idx    = argmin(abs.(cand_means .- hisMeanRD_fl))
    path_regs   = candidates[best_idx][1]
    path_rems   = candidates[best_idx][2]
    println("HSMM candidates (target: $(round(hisMeanRD_fl, digits=2)) TWh/yr): mean=$(round(mean(cand_means), digits=2))  std=$(round(std(cand_means), digits=2))  range=[$(round(minimum(cand_means), digits=2)), $(round(maximum(cand_means), digits=2))]  selected=$(round(cand_means[best_idx], digits=2))")
    simRegime_arr = [startReg_int; path_regs]
    simRemain_arr = [startRem_int; path_rems]

    # extra nodes for forecasting context
    mainHorizonNodes_int = (yRun * 365) ÷ daysPerNode_int

    while length(pfNode_arr) < mainHorizonNodes_int + lvlMax_int

        # chain daysPerNode_int single-day diffusion calls, each conditioned on the previous day's actually-realized output
        dayData_arr = Vector{Vector{Float32}}(undef, daysPerNode_int)

        for dOff in 0:(daysPerNode_int - 1)
            newDoy_int     = date_tup[1] + dOff
            yearsAhead_int = (newDoy_int - 1) ÷ 365
            dCor_int       = newDoy_int - yearsAhead_int * 365
            yCor_int       = date_tup[2] + yearsAhead_int

            sim_int = (yCor_int - 1) * 365 + dCor_int
            # single day's [prev, cur, next] regime sequence
            reg_vec = simRegime_arr[sim_int : sim_int + 2]

            # create forecast; level info reaches the model via regDiff_vec's middle block (current regime centroid)
            relHis_arr  = vcat([h[end - dayContDim_int + 1 : end] for h in hisPf_arr]...)
            regDiff_vec = regimeDiffVec(dCor_int, reg_vec)

            frc_mat = createForecasts(dCor_int, yCor_int, relHis_arr, dif_mod, 1, regDiff_vec, corrPath = true, dimSteps_fl = dimSteps)

            dayData_arr[dOff + 1] = frc_mat[:, 1]

            # update historic context based on the day just realized, so the next day's generation sees it
            for i in 1:numReg_int
                append!(hisPf_arr[i], frc_mat[(i - 1) * dayDataDim_int + 1 : i * dayDataDim_int, 1])
            end
        end

        # stitch the chained single-day outputs into the node's region-major layout
        nodeData_vec = vcat([vcat([dayData_arr[dd][(k - 1) * dayDataDim_int + 1 : k * dayDataDim_int] for dd in 1:daysPerNode_int]...) for k in 1:numReg_int]...)

        # single deterministic chain: id/lvl both track the running node count, ancestor is the previous node
        id_int  = length(pfNode_arr) + 1
        anc_int = id_int == 1 ? 0 : id_int - 1
        push!(pfNode_arr, nodeData(id_int, id_int, anc_int, nodeData_vec, 1.0))

        # advance counter by a full node (daysPerNode_int days)
        newDay_int = date_tup[1] + daysPerNode_int
        if newDay_int > 365
            date_tup = (newDay_int - 365, date_tup[2] + 1)
        else
            date_tup = (newDay_int, date_tup[2])
        end

    end

    # ex-post additive correction: shift node data so actual mean RD matches hisMeanRD_fl
    pre_mean_fl = let total = 0.0
        for nd in pfNode_arr[1:mainHorizonNodes_int], i in 1:numReg_int
            lim   = ext_dic[String(reg_arr[i])]
            rng   = (i - 1) * dataDim_int + 1 : i * dataDim_int
            total += sum((nd.data[rng] .+ 1f0) ./ 2f0 .* (lim[2] - lim[1]) .+ lim[1])
        end
        total / (mainHorizonNodes_int * dataDim_int) * 8760 / 1000
    end
    delta_norm_arr = Float32[2.0 * (hisMeanRD_fl - pre_mean_fl) * 1000 / 8760 / numReg_int /
                             (ext_dic[String(reg_arr[i])][2] - ext_dic[String(reg_arr[i])][1]) for i in 1:numReg_int]
    for nd in pfNode_arr, i in 1:numReg_int
        nd.data[(i - 1) * dataDim_int + 1 : i * dataDim_int] .+= delta_norm_arr[i]
    end
    println("Ex-post correction: before=$(round(pre_mean_fl, digits=2))  after≈$(round(hisMeanRD_fl, digits=2)) TWh/yr")

    return pfNode_arr, simRegime_arr, simRemain_arr
end

function generateErrorMat(d::Int, yearIdx_int::Int)
    # error-curve saturation value for the given (already drawn) year
    esatPool_arr = eSatAll_arr[mod1(d + T_int, 365)]
    satDraw_tup  = esatPool_arr[yearIdx_int]

    # pathDays_int spans every calendar day across all levels
    pathDays_int = maximum(setups_dic[:det]) * daysPerNode_int + 1

    # targeted signed level shift per day/region
    return [targetErrorValue(dOff + 1, x0_arr[k], satDraw_tup[k]) for dOff in 0:(pathDays_int - 1), k in 1:numReg_int]
end

# generate numFrc candidates per conditioning set (one column each) in one batched pass, keep per set whichever candidate's own level best matches that set's target level for the day
function bestMatchForecast(d, y, his_mat::Matrix{Float32}, regDiff_mat::Matrix{Float32}, refLevel_mat::Matrix{Float32})
    frc_mat  = createForecasts(d, y, his_mat, dif_mod, numFrc, regDiff_mat, corrPath = true, dimSteps_fl = dimSteps)
    levelMat = computeLevelMat(frc_mat)

    numSet_int    = size(his_mat, 2)
    bestData_mat  = Matrix{Float32}(undef, size(frc_mat, 1), numSet_int)
    bestLevel_mat = Matrix{Float32}(undef, numReg_int, numSet_int)
    for s_int in 1:numSet_int
        col_rng = (s_int - 1) * numFrc + 1 : s_int * numFrc
        # distance between two already-computed level vectors, not a forecast metric
        diffErr_arr = [errorVec(:rmse, levelMat[:, c], refLevel_mat[:, s_int]) for c in col_rng]
        bestIdx_int = col_rng[argmin(diffErr_arr)]
        bestData_mat[:, s_int]  = frc_mat[:, bestIdx_int]
        bestLevel_mat[:, s_int] = levelMat[:, bestIdx_int]
    end
    return bestData_mat, bestLevel_mat
end

# row range into data_mat for year-index yi at day d+dOff (day wraps at 365, yi itself is not incremented)
function dataMatRows(d::Int, dOff::Int, yi::Int)
    doy_int = mod1(d + dOff, 365)
    return (yi - 1) * 8760 + (doy_int - 1) * 24 + 1 : (yi - 1) * 8760 + doy_int * 24
end

# assemble a batch of multi-day nodes, each from daysPerNode_int chained single-day diffusion calls conditioned on its own previous day's actually-chosen output and stitched into the node's region-major .data layout; all nodes of the batch share one diffusion pass per day
function generateNodeProfile(d::Int, y::Int, his_arr::Vector{Vector{Float32}},
                              regDiffDay_arr::Vector{Vector{Vector{Float32}}}, refLevelDay_arr::Vector{Vector{Vector{Float32}}})

    numNode_int = length(his_arr)
    ndays_int   = length(regDiffDay_arr[1])

    # rolling 2-day context per node as one column, taken from the tail of each region's block
    ctx_mat = stack([begin
        hisLen_int = length(h) ÷ numReg_int
        vcat([h[(k-1)*hisLen_int+hisLen_int-dayContDim_int+1 : (k-1)*hisLen_int+hisLen_int] for k in 1:numReg_int]...)
    end for h in his_arr])

    dayData_arr  = Vector{Matrix{Float32}}(undef, ndays_int)
    dayLevel_arr = Vector{Matrix{Float32}}(undef, ndays_int)

    for dOff in 0:(ndays_int - 1)
        yearsAhead_int = (d + dOff - 1) ÷ 365
        dCor_int       = d + dOff - yearsAhead_int * 365
        yCor_int       = y + yearsAhead_int

        regDiff_mat  = stack([r[dOff + 1] for r in regDiffDay_arr])
        refLevel_mat = stack([r[dOff + 1] for r in refLevelDay_arr])

        bestData_mat, bestLevel_mat = bestMatchForecast(dCor_int, yCor_int, ctx_mat, regDiff_mat, refLevel_mat)
        dayData_arr[dOff + 1]  = bestData_mat
        dayLevel_arr[dOff + 1] = bestLevel_mat

        # slide every node's rolling 2-day context forward by the day just generated
        ctx_mat = vcat([vcat(ctx_mat[(k-1)*dayContDim_int+1 : k*dayContDim_int, :], bestData_mat[(k-1)*dayDataDim_int+1 : k*dayDataDim_int, :])[end-dayContDim_int+1:end, :] for k in 1:numReg_int]...)
    end

    nodeData_arr = [vcat([vcat([dayData_arr[dd][(k-1)*dayDataDim_int+1 : k*dayDataDim_int, n] for dd in 1:ndays_int]...) for k in 1:numReg_int]...) for n in 1:numNode_int]

    # per-day achieved levels per node, so the caller can take |error| per day before averaging
    nodeLevel_arr = [[dayLevel_arr[dd][:, n] for dd in 1:ndays_int] for n in 1:numNode_int]

    return nodeData_arr, nodeLevel_arr
end

# the path must sit exactly at its achievable bound each day; day 1 stays anchored, later days pick +bound/-bound (one shared binary per day) to best track yearVal_mat's own day-to-day movement
function buildAbsPath(dailyVal_mat::Matrix{Float64}, minVal_mat::Matrix{Float64}, maxVal_mat::Matrix{Float64}, rangeErr_mat::Matrix{Float64}, yearVal_mat::Matrix{Float64})

    pathDays_int, numReg_int = size(dailyVal_mat)
    path_mat = Matrix{Float64}(undef, pathDays_int, numReg_int)
    path_mat[1, :] = dailyVal_mat[1, :]

    # one Gurobi environment reused per thread (task-local, so no cross-thread sharing)
    env = get!(task_local_storage(), :gurobiEnv) do
        lock(() -> Gurobi.Env(), gurobiEnvLock)
    end
    mod = Model(() -> Gurobi.Optimizer(env))
    set_silent(mod)

    @variable(mod, bin[2:pathDays_int], Bin)
    @variable(mod, delta[1:(pathDays_int - 1), 1:numReg_int] >= 0)

    # steer away from whichever direction would clip at +-1, so the full target error is hit rather than truncated
    for r in 2:pathDays_int
        upSafe_bl   = all(dailyVal_mat[r, k] + rangeErr_mat[r, k] <= 1.0 for k in 1:numReg_int)
        downSafe_bl = all(dailyVal_mat[r, k] - rangeErr_mat[r, k] >= -1.0 for k in 1:numReg_int)
        upSafe_bl && !downSafe_bl && fix(bin[r], 0; force = true)
        downSafe_bl && !upSafe_bl && fix(bin[r], 1; force = true)
    end

    fPr = Matrix{Any}(undef, pathDays_int, numReg_int)
    for k in 1:numReg_int
        fPr[1, k] = dailyVal_mat[1, k]
        for r in 2:pathDays_int
            fPr[r, k] = maxVal_mat[r, k] * (1 - bin[r]) + minVal_mat[r, k] * bin[r]
        end
    end

    for r in 1:(pathDays_int - 1), k in 1:numReg_int
        deltaE_fl = yearVal_mat[r + 1, k] - yearVal_mat[r, k]
        @constraint(mod, delta[r, k] >= (fPr[r + 1, k] - fPr[r, k]) - deltaE_fl)
        @constraint(mod, delta[r, k] >= -((fPr[r + 1, k] - fPr[r, k]) - deltaE_fl))
    end

    @objective(mod, Min, sum(delta))
    optimize!(mod)

    for r in 2:pathDays_int, k in 1:numReg_int
        path_mat[r, k] = value(bin[r]) > 0.5 ? minVal_mat[r, k] : maxVal_mat[r, k]
    end

    return path_mat
end

# build the target path (in [-1,1] space) via the exact-boundary MILP
function buildTargetPath(dailyVal_mat::Matrix{Float64}, targetErr_mat::Matrix{Float64}, yearVal_mat::Matrix{Float64})

    pathDays_int, numReg_int = size(dailyVal_mat)

    # corridor half-width, so it must be non-negative -- the metric is signed now
    absErr_mat = abs.(targetErr_mat)
    minVal_mat = max.(-1.1, dailyVal_mat .- absErr_mat)
    maxVal_mat = min.(1.1, dailyVal_mat .+ absErr_mat)

    return buildAbsPath(dailyVal_mat, minVal_mat, maxVal_mat, absErr_mat, yearVal_mat)
end

# target path of the deterministic tree for day d, anchored on the already-realized pfData_arr nodes from index i on (no diffusion involved, node data is filled by compileForecasts)
function buildDetTarget(d::Int, i::Int)

    yearIdx_int   = rand(1:length(y_arr))
    targetErr_mat = generateErrorMat(d, yearIdx_int)
    pathDays_int  = maximum(setups_dic[:det]) * daysPerNode_int + 1

    # daily realized mean values for the next pathDays_int days (in [-1,1] space)
    dailyVal_mat = Matrix{Float64}(undef, pathDays_int, numReg_int)
    for dOff in 0:(pathDays_int - 1)
        pfNodeIdx_int = i + dOff ÷ daysPerNode_int
        dayInNode_int = dOff % daysPerNode_int
        pfData_vec = pfData_arr[pfNodeIdx_int].data
        for k in eachindex(reg_arr)
            rng = (k - 1) * dataDim_int + dayInNode_int * 24 + 1 : (k - 1) * dataDim_int + (dayInNode_int + 1) * 24
            dailyVal_mat[dOff + 1, k] = mean(pfData_vec[rng])
        end
    end

    # empirical path, used to shape the day-to-day movement of the constructed path
    yearVal_mat = Matrix{Float64}(undef, pathDays_int, numReg_int)
    for dOff in 0:(pathDays_int - 1)
        yearVal_mat[dOff + 1, :] = vec(mean(data_mat[dataMatRows(d, dOff, yearIdx_int), :], dims = 1))
    end

    # build the target path via the exact-boundary MILP
    path_mat = buildTargetPath(dailyVal_mat, targetErr_mat, yearVal_mat)

    # path_mat is already [-1,1] space -- convert to anomaly space
    pathAnom_mat = Matrix{Float32}(undef, pathDays_int, numReg_int)
    for dOff in 0:(pathDays_int - 1)
        doy_int = mod1(d + dOff, 365)
        pathAnom_mat[dOff + 1, :] = Float32.(path_mat[dOff + 1, :]) .- dailyNormMean_mat[doy_int, :]
    end

    return path_mat, pathAnom_mat, dailyVal_mat
end

# candidate paths and flexible-branching tree structure of one stochastic setup for day d (no diffusion involved, node data is filled by compileForecasts)
function buildStochStructure(d::Int, iBudget_int::Int, treeLvl_int::Int, i::Int)

    pathDays_int      = treeLvl_int * daysPerNode_int + 1
    numTrainYears_int = length(y_arr)

    esatPool_arr = eSatAll_arr[mod1(d + T_int, 365)]

    # already-realized/forecasted daily mean from pfData_arr (same across years, so computed once)
    dailyVal_mat = Matrix{Float64}(undef, pathDays_int, numReg_int)
    for dOff in 0:(pathDays_int - 1)
        pfNodeIdx_int = i + dOff ÷ daysPerNode_int
        dayInNode_int = dOff % daysPerNode_int
        pfData_vec = pfData_arr[pfNodeIdx_int].data
        for k in eachindex(reg_arr)
            rng = (k - 1) * dataDim_int + dayInNode_int * 24 + 1 : (k - 1) * dataDim_int + (dayInNode_int + 1) * 24
            dailyVal_mat[dOff + 1, k] = mean(pfData_vec[rng])
        end
    end

    # one candidate path per historical year, each scaled to that year's own E_sat-based target error along its own shape
    allPaths_dic = Dict{Int, Matrix{Float64}}()

    for yi in 1:numTrainYears_int

        targetErr_mat = [targetErrorValue(dOff + 1, x0_arr[k], esatPool_arr[yi][k]) for dOff in 0:(pathDays_int - 1), k in 1:numReg_int]

        yearVal_mat = Matrix{Float64}(undef, pathDays_int, numReg_int)
        for dOff in 0:(pathDays_int - 1)
            yearVal_mat[dOff + 1, :] = vec(mean(data_mat[dataMatRows(d, dOff, yi), :], dims = 1))
        end
        # build the target path via the exact-boundary MILP; its level feeds forward, not the shaping error
        allPaths_dic[yi] = buildTargetPath(dailyVal_mat, targetErr_mat, yearVal_mat)
    end

    # cluster paths into a flexible-branching tree, capped at maxStochStruct_int distinct structures
    nodeDic, nodeYear_dic = buildStochTree(treeLvl_int, iBudget_int, allPaths_dic, (iBudget_int, treeLvl_int))

    return nodeDic, nodeYear_dic, allPaths_dic
end

# history vector as the sequential loop had grown it before node i: the initial context followed by every earlier pfData_arr node; only a tail of at most two nodes is ever read, so older nodes are left out
function hisBeforeNode(his0_arr::Vector{Float32}, i::Int)
    return vcat(his0_arr, [pfData_arr[j].data for j in max(1, i - 2):(i - 1)]...)
end

# build the deterministic tree and every stochastic setup's tree for day d, rooted at the already-realized node i; nodes are filled level by level, so all nodes of a level across all trees share one batched diffusion pass per day
function compileForecasts(d::Int, y::Int, his_arr::Vector{Float32}, real_nd::nodeData, i::Int)

    detLvl_int = maximum(setups_dic[:det])

    # target paths and tree structures first, none of them depends on generated node data
    path_mat, pathAnom_mat, dailyVal_mat = buildDetTarget(d, i)
    det_dic = Dict{Int, nodeData}(1 => nodeData(1, 1, 0, real_nd.data, 1.0))

    stoch_dic = Dict{Tuple{Int,Int}, Tuple{Dict{Int, nodeData}, Dict{Int, Int}, Dict{Int, Matrix{Float64}}}}()
    for s in setups_dic[:stoch]
        nodeDic, nodeYear_dic, allPaths_dic = buildStochStructure(d, s[1], s[2], i)
        nodeDic[1].data = real_nd.data
        stoch_dic[s] = (nodeDic, nodeYear_dic, allPaths_dic)
    end

    errTrack_arr = Vector{Tuple{Int,Int, Vector{Float32}, Vector{Float32}}}()

    for l in 2:max(detLvl_int, maximum(s[2] for s in setups_dic[:stoch]))

        hisVal_int    = max(0, contDim_int - dataDim_int * (l - 1))
        synVal_int    = contDim_int - hisVal_int
        relHis_arr    = his_arr[end - hisVal_int * numReg_int + 1 : end]
        dayOffset_int = (l - 1) * daysPerNode_int

        yearsAhead_int = (d + dayOffset_int - 1) ÷ 365
        dCor_int       = d + dayOffset_int - yearsAhead_int * 365
        yCor_int       = y + yearsAhead_int

        # conditioning of every node on this level across all trees, and the node each result is written to
        ctx_arr      = Vector{Vector{Float32}}()
        regDiff_arr  = Vector{Vector{Vector{Float32}}}()
        refLevel_arr = Vector{Vector{Vector{Float32}}}()
        target_arr   = Vector{nodeData}()

        if l <= detLvl_int
            # per-day conditioning vectors built directly from the targeted path, one triplet/profile/target per calendar day in the node
            push!(regDiff_arr, [vcat(
                pathAnom_mat[dayOffset_int + dOff + 1, :] .- pathAnom_mat[dayOffset_int + dOff, :],
                pathAnom_mat[dayOffset_int + dOff + 1, :],
                pathAnom_mat[dayOffset_int + dOff + 2, :] .- pathAnom_mat[dayOffset_int + dOff + 1, :]
            ) for dOff in 0:(daysPerNode_int - 1)])

            # target level (from the MILP-built path itself, not the error that shaped it), per day
            push!(refLevel_arr, [Float32.(path_mat[dayOffset_int + dOff + 1, :]) for dOff in 0:(daysPerNode_int - 1)])

            push!(ctx_arr, vcat(relHis_arr, getSynContext(det_dic[l-1], synVal_int, collect(values(det_dic)))))
            det_dic[l] = nodeData(l, l, l-1, Float32[], 1.0)
            push!(target_arr, det_dic[l])
        end

        for (nodeDic, nodeYear_dic, allPaths_dic) in values(stoch_dic)
            for (id, nd) in nodeDic
                nd.lvl != l && continue

                yi = nodeYear_dic[id]
                pathAnomRow = r -> Float32.(allPaths_dic[yi][r, :]) .- dailyNormMean_mat[mod1(d + r - 1, 365), :]

                # per-day conditioning vectors, built one day beyond the node's own span (excluded from clustering, needed here for the exit diff)
                push!(regDiff_arr, [vcat(
                    pathAnomRow(dayOffset_int + dOff + 1) .- pathAnomRow(dayOffset_int + dOff),
                    pathAnomRow(dayOffset_int + dOff + 1),
                    pathAnomRow(dayOffset_int + dOff + 2) .- pathAnomRow(dayOffset_int + dOff + 1)
                ) for dOff in 0:(daysPerNode_int - 1)])

                # target level from this year's own MILP-built path, not the error that shaped it
                push!(refLevel_arr, [Float32.(allPaths_dic[yi][dayOffset_int + dOff + 1, :]) for dOff in 0:(daysPerNode_int - 1)])

                push!(ctx_arr, vcat(relHis_arr, getSynContext(nodeDic[nd.ancestor], synVal_int, collect(values(nodeDic)))))
                push!(target_arr, nd)
            end
        end

        isempty(target_arr) && continue

        # build forecasts with diffusion model for the whole level at once, keep per node whichever candidate's by-region level best matches its target, chained day by day
        nodeData_arr, nodeLevel_arr = generateNodeProfile(dCor_int, yCor_int, ctx_arr, regDiff_arr, refLevel_arr)
        for (n_int, nd) in enumerate(target_arr)
            nd.data = nodeData_arr[n_int]
        end

        # the det node sits first in the batch; track |forecast - realized| per day before averaging over the node, so over/undershoots can't cancel
        if l <= detLvl_int
            realLevelDay_arr = [Float32.(dailyVal_mat[dayOffset_int + dOff + 1, :]) for dOff in 0:(daysPerNode_int - 1)]
            tgtErrDay_arr    = [abs.(refLevel_arr[1][k] .- realLevelDay_arr[k]) for k in 1:daysPerNode_int]
            achErrDay_arr    = [abs.(nodeLevel_arr[1][k] .- realLevelDay_arr[k]) for k in 1:daysPerNode_int]
            push!(errTrack_arr, (d, l, mean(tgtErrDay_arr), mean(achErrDay_arr)))
        end
    end

    # compact tree structure per stochastic setup: (lvl, ancestor) per node id, enough to reconstruct the full tree later (e.g. treeStructMat) without carrying the full nodeData around
    stochRes_dic = Dict(s => (tree_tup[1], [(tree_tup[1][id].lvl, tree_tup[1][id].ancestor) for id in 1:length(tree_tup[1])]) for (s, tree_tup) in stoch_dic)

    return det_dic, errTrack_arr, stochRes_dic
end
