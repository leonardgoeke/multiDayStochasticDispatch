
function preparations()

    mkpath("results/preprocessing")
    mkpath("results/profiles/" * run_str)

    #region # * compute time-series for residual demand

    # read capacity values and relevant time-series files
    re_df = filter(x -> x.region_1 in ("CH","DE","FR"), CSV.read("intermediate/capacity_" * plan_str * "/par_capaConv.csv", DataFrame))
    #re_df = CSV.read("modelData/intermediate/par_capaConv.csv", DataFrame)
    re_df = unstack(filter(x -> x.technology_1 in ("wind", "pv"), re_df), :technology_1, :value)

    tsFile_arr = readdir("modelData/planning/timeSeries")
    global y_arr = sort(unique(map(x -> split(x[end],".")[1], split.(tsFile_arr,"_"))))
    global yAll_arr = string.(1982:2100)

    # loop over years
    global allRes_df = DataFrame(region_1 = String[], timestep_4 = Int[], year = Int[], value = Float64[])
    global peakLoad_dic = Dict{String, Float64}()
    for y in y_arr

        # read-in and merge time-series data
        res_df = rename(CSV.read("modelData/planning/timeSeries/par_demand_3ctr_" * y * ".csv", DataFrame), :value => :dem)
        pv_df = rename(CSV.read("modelData/planning/timeSeries/par_pv_3ctr_" * y * ".csv", DataFrame), :value => :pvCf)
        wind_df = rename(CSV.read("modelData/planning/timeSeries/par_wind_3ctr_" * y * ".csv", DataFrame), :value => :windCf)
        res_df = innerjoin(select(res_df,[:region_1,:timestep_4,:dem]), select(pv_df,[:region_1,:timestep_4,:pvCf]), on = [:region_1, :timestep_4])
        res_df = innerjoin(select(res_df,[:region_1,:timestep_4,:dem,:pvCf]), select(wind_df,[:region_1,:timestep_4,:windCf]), on = [:region_1, :timestep_4])

        # track peak load per region across years
        for g in groupby(res_df, :region_1)
            r_str = g.region_1[1]
            peakLoad_dic[r_str] = max(get(peakLoad_dic, r_str, -Inf), maximum(g.dem))
        end

        # add capacities
        re_df.pv = coalesce.(re_df.pv, 0.0)
        re_df.wind = coalesce.(re_df.wind, 0.0)
        res_df = innerjoin(res_df, select(re_df, Not([:parameter])), on = :region_1)
        res_df[!,:value] = res_df[!,:dem] .- (res_df[!,:pv] .* res_df[!,:pvCf]) .- (res_df[!,:wind] .* res_df[!,:windCf])

        # add to other data
        res_df[!,:timestep_4] = parse.(Int,replace.(res_df[!,:timestep_4], "h" => ""))
        res_df[!,:year] .= parse.(Int, y)
        append!(allRes_df, select(res_df, [:region_1, :timestep_4, :year, :value]))

    end

    global reg_arr = Symbol.(sort(unique(allRes_df[!,:region_1])))

    # sort and normalize data
    sort!(allRes_df,[:year, :timestep_4, :region_1])
    global ext_dic = Dict(x.region_1[1] => (minimum(x.value), maximum(x.value)) for x in groupby(allRes_df, [:region_1]))
    allRes_df[!,:value] = map(x -> Float32(2 * (x.value - ext_dic[x.region_1][1]) / (ext_dic[x.region_1][2] - ext_dic[x.region_1][1]) - 1.0), eachrow(allRes_df))
    allRes_df = unstack(allRes_df, :region_1, :value)

    global numReg_int = ncol(allRes_df) - 2

    #endregion

    #region # * setting of diffusion network

    global steps_int = 1000
    global beta_rng = range(1e-4, 0.02, length = steps_int)
    global alpha_rng = 1.0 .- beta_rng
    global alphaBar_rng = cumprod(alpha_rng)

    global embDim1_int = 256
    global embDim2_int = 512
    global batchSize_int = 128

    global data_mat = Matrix{Float32}(allRes_df[:, 3:end])

    global daysPerNode_int = dCnt # number of calendar days spanned by one forecast/dispatch node
    global dataDim_int = daysPerNode_int * 24 # number of hours per node (node-level, used by the forecast-tree layer)
    global timeDim_int = 1 # number of time features (so far, only time of day and weekday, needs adjustment, when not forecasting 24h)
    global contDim_int = 2 * dataDim_int # number of node-level context hours (two nodes' worth)

    global dayDataDim_int = 24 # the diffusion model's own atomic generation unit -- always exactly one day, regardless of daysPerNode_int
    global dayContDim_int = 2 * dayDataDim_int # fixed 2-day context window feeding a single-day generation

    # create diffusion model
    global dif_mod = ConditionalDenoiser()

    global calCont_dic = Dict{Tuple{Int, Int}, Vector{Float32}}()
    for y_int in 1982:2100
        for d_int in 1:365
            contTime_arr = Float32.([sin(d_int/365*2*pi), cos(d_int/365*2*pi)])
            wd_int = dayofweek(Date(y_int, 1, 1) + Day(d_int-1))
            contWk_arr = zeros(Float32, 7)
            contWk_arr[wd_int] = 1.0f0
            calCont_dic[(d_int, y_int)] = vcat(contTime_arr, contWk_arr)
        end
    end

    # harmonic regression seasonal mean: K sine/cosine pairs fitted to raw daily means
    numK_int = 1
    numTrainYears_int = length(y_arr)

    rawMean_mat = Matrix{Float32}(undef, 365, numReg_int)
    for d_int in 1:365
        hour_rows = vcat([collect((yi * 8760) + (d_int - 1) * 24 + 1 : (yi * 8760) + d_int * 24) for yi in 0:(numTrainYears_int - 1)]...)
        rawMean_mat[d_int, :] = vec(mean(data_mat[hour_rows, :], dims = 1))
    end

    # empirical error of climatological mean per day of year, numReg_int per tuple, signed: mean(target - reference)
    # leave-one-out: each year is compared against the mean of the OTHER years only, since including itself pulls the reference toward it and understates the error
    global eSatAll_arr = Vector{Vector{Tuple}}(undef, 365)
    for d_int in 1:365
        yearProfiles_arr = [data_mat[(yi * 8760) + (d_int - 1) * 24 + 1 : (yi * 8760) + d_int * 24, :] for yi in 0:(numTrainYears_int - 1)]
        sumProfile_mat   = sum(yearProfiles_arr)
        eSatAll_arr[d_int] = [Tuple(
            [errorVec(:abs, yp[:, i_int], (sumProfile_mat[:, i_int] .- yp[:, i_int]) ./ (numTrainYears_int - 1)) for i_int in eachindex(reg_arr)]
        ) for yp in yearProfiles_arr]
    end

    # region-specific day-ahead error factor: energy-share-weighted quadratic mean of the per-category factors of the planning run's generation mix
    dayShare_dic = Dict((x.region, x.category)   => x.value for x in eachrow(CSV.read("modelData/dayaheadShareEsat.csv", DataFrame)))
    enQuant_dic  = Dict((x.region, x.technology) => x.value for x in eachrow(CSV.read("intermediate/capacity_" * plan_str * "/energyQuant.csv", DataFrame)))

    # fit the logistic to return that factor on day 1 while still reaching p_fl at T_int; the logit is linear in t, so both anchors have a closed form
    lgtSat_fl = log(p_fl / (1.0 - p_fl))
    global x0_arr = Vector{Float64}(undef, numReg_int)
    for (i_int, r) in enumerate(reg_arr)
        quant_arr   = Float64[enQuant_dic[(String(r), c)]  for c in ("demand", "wind", "pv")]
        share_arr   = Float64[dayShare_dic[(String(r), c)] for c in ("demand", "wind", "pv")]
        dayAhead_fl = sqrt(sum((quant_arr ./ sum(quant_arr)) .* share_arr .^ 2))

        lgtDay_fl     = log(dayAhead_fl / (1.0 - dayAhead_fl))
        alpha_fl      = (lgtSat_fl - lgtDay_fl) / (T_int - 1)
        x0_arr[i_int] = 1.0 / (1.0 + exp(alpha_fl - lgtDay_fl))

        println("Day-ahead error $(r): share=$(round(dayAhead_fl, digits=4))  x0=$(round(x0_arr[i_int], digits=4))  alpha=$(round(alpha_fl, digits=4))")
    end

    d_arr = Float64.(1:365)
    X_mat = Matrix{Float64}(undef, 365, 2 * numK_int + 1)
    X_mat[:, 1] .= 1.0
    for k_int in 1:numK_int
        X_mat[:, 2k_int]     = cos.(2π * k_int * d_arr / 365)
        X_mat[:, 2k_int + 1] = sin.(2π * k_int * d_arr / 365)
    end
    global dailyNormMean_mat = Float32.(X_mat * (X_mat \ Float64.(rawMean_mat)))

    # HSMM weather regime calibration (month-specific)
    global numRegimes_int = 10
    nTrainDays_int = size(data_mat, 1) ÷ 24
    maxDwell_int   = 60
    global doyToMonth_arr = vcat(fill.(1:12, [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])...)

    dailyAnom_mat = Matrix{Float32}(undef, numReg_int, nTrainDays_int)
    for d_int in 1:nTrainDays_int
        doy_int  = mod1(d_int, 365)
        rows_arr = (d_int - 1) * 24 + 1 : d_int * 24
        for i_int in 1:numReg_int
            dailyAnom_mat[i_int, d_int] = mean(data_mat[rows_arr, i_int]) - dailyNormMean_mat[doy_int, i_int]
        end
    end

    doyAll_arr   = mod1.(1:nTrainDays_int, 365)
    monthAll_arr = doyToMonth_arr[doyAll_arr]

    global regCent_arr  = Vector{Matrix{Float32}}(undef, 12)
    global regDwell_arr = Vector{Matrix{Float32}}(undef, 12)
    global regTrans_arr = Vector{Matrix{Float32}}(undef, 12)
    global regInit_arr  = Vector{Vector{Float32}}(undef, 12)
    global dayRegime_arr = zeros(Int, nTrainDays_int)

    monthNames_arr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    minVals_m_vec  = Float32[ext_dic[String(r)][1] for r in reg_arr]
    maxVals_m_vec  = Float32[ext_dic[String(r)][2] for r in reg_arr]

    # elbow curves (WCSS vs k for each month)
    maxK_int = 15
    elbow_pl = plot(layout = (3, 4), size = (1200, 900),
                    bottom_margin = 6Plots.mm, left_margin = 6Plots.mm)
    for m_int in 1:12
        mask_elbow = findall(monthAll_arr .== m_int)
        anom_elbow = dailyAnom_mat[:, mask_elbow]
        wcss_vec   = Float64[kmeans(anom_elbow, k; maxiter = 100).totalcost for k in 1:maxK_int]
        plot!(elbow_pl, 1:maxK_int, wcss_vec; sp = m_int, title = monthNames_arr[m_int],
              xlabel = "k", ylabel = "WCSS", label = "WCSS", marker = :circle, lw = 2)
        vline!(elbow_pl, [numRegimes_int]; sp = m_int, color = :red, ls = :dash, label = "k=$numRegimes_int")
    end
    savefig(elbow_pl, "results/preprocessing/elbow_monthly_" * run_str * ".png")

    for m_int in 1:12
        mask_arr    = findall(monthAll_arr .== m_int)
        anom_m_mat  = dailyAnom_mat[:, mask_arr]
        km_m        = kmeans(anom_m_mat, numRegimes_int; maxiter = 300)
        state_m_arr = km_m.assignments

        regCent_arr[m_int] = Float32.(km_m.centers)

        # sort regimes by ascending aggregate centroid (index 1 = most severe drought)
        centSortAgg_vec    = Float32[sum(regCent_arr[m_int][i, k] * (maxVals_m_vec[i] - minVals_m_vec[i]) / 2.0f0 for i in 1:numReg_int) for k in 1:numRegimes_int]
        perm_arr           = sortperm(centSortAgg_vec)
        regCent_arr[m_int] = regCent_arr[m_int][:, perm_arr]
        state_m_arr        = invperm(perm_arr)[state_m_arr]
        dayRegime_arr[mask_arr] = state_m_arr

        dwellCnt_m_mat = zeros(Float32, numRegimes_int, maxDwell_int)
        transCnt_m_mat = zeros(Float32, numRegimes_int, numRegimes_int)
        d_idx = 1
        while d_idx <= length(state_m_arr)
            k_int   = state_m_arr[d_idx]
            run_int = 1
            while d_idx + run_int <= length(state_m_arr) && state_m_arr[d_idx + run_int] == k_int
                run_int += 1
            end
            dwellCnt_m_mat[k_int, min(run_int, maxDwell_int)] += 1
            if d_idx + run_int <= length(state_m_arr)
                transCnt_m_mat[k_int, state_m_arr[d_idx + run_int]] += 1
            end
            d_idx += run_int
        end

        dwell_m_mat = zeros(Float32, numRegimes_int, maxDwell_int)
        for k_int in 1:numRegimes_int
            total_k = sum(dwellCnt_m_mat[k_int, :])
            if total_k > 0
                dwell_m_mat[k_int, :] = dwellCnt_m_mat[k_int, :] ./ total_k
            else
                dwell_m_mat[k_int, 1] = 1.0f0
            end
        end
        trans_m_mat = copy(transCnt_m_mat)
        for k_int in 1:numRegimes_int
            s_fl = sum(trans_m_mat[k_int, :])
            s_fl > 0 && (trans_m_mat[k_int, :] ./= s_fl)
        end

        regDwell_arr[m_int] = dwell_m_mat
        regTrans_arr[m_int] = trans_m_mat
        regInit_arr[m_int]  = Float32[count(==(k_int), state_m_arr) / length(state_m_arr) for k_int in 1:numRegimes_int]

    end

    # recompute dwell and transitions from full time series so cross-month runs are captured correctly
    crossCnt_arr   = [zeros(Float32, numRegimes_int, maxDwell_int) for _ in 1:12]
    crossTrans_arr = [zeros(Float32, numRegimes_int, numRegimes_int) for _ in 1:12]
    d_idx = 1
    while d_idx <= length(dayRegime_arr)
        k_int   = dayRegime_arr[d_idx]
        run_int = 1
        while d_idx + run_int <= length(dayRegime_arr) && dayRegime_arr[d_idx + run_int] == k_int
            run_int += 1
        end
        crossCnt_arr[doyToMonth_arr[mod1(d_idx, 365)]][k_int, min(run_int, maxDwell_int)] += 1
        if d_idx + run_int <= length(dayRegime_arr)
            new_month = doyToMonth_arr[mod1(d_idx + run_int, 365)]
            crossTrans_arr[new_month][k_int, dayRegime_arr[d_idx + run_int]] += 1
        end
        d_idx += run_int
    end
    for m_int in 1:12
        for k_int in 1:numRegimes_int
            total_k = sum(crossCnt_arr[m_int][k_int, :])
            if total_k > 0
                regDwell_arr[m_int][k_int, :] = crossCnt_arr[m_int][k_int, :] ./ total_k
            else
                regDwell_arr[m_int][k_int, 1] = 1.0f0
            end
            total_t = sum(crossTrans_arr[m_int][k_int, :])
            if total_t > 0
                regTrans_arr[m_int][k_int, :] = crossTrans_arr[m_int][k_int, :] ./ total_t
            end
        end
    end

    # cap on distinct scenario-tree structures (bounds dispatch model count), reset for a clean rerun
    global maxStochStruct_int    = 150
    global stochStructCache_dic  = Dict{Tuple{Int,Int}, Vector{Vector{Tuple{Int,Int}}}}()
    global stochStructLock       = ReentrantLock()
    # Gurobi environments are started from the threaded forecast loop, but starting them is not thread-safe
    global gurobiEnvLock         = ReentrantLock()

    global startReg_int = sample(1:numRegimes_int, Weights(regInit_arr[1]))
    global startRem_int = _sampleHsmmDwell(startReg_int, 1)

    global hisMeanRD_fl = let total = 0.0
        for d_int in 1:length(dayRegime_arr)
            doy_int   = mod1(d_int, 365)
            month_int = doyToMonth_arr[doy_int]
            reg_int   = dayRegime_arr[d_int]
            for i_int in 1:numReg_int
                lim_tup = ext_dic[String(reg_arr[i_int])]
                norm_fl = dailyNormMean_mat[doy_int, i_int] + regCent_arr[month_int][i_int, reg_int]
                total  += (norm_fl + 1.0) / 2.0 * (lim_tup[2] - lim_tup[1]) + lim_tup[1]
            end
        end
        total / length(dayRegime_arr) * 8760 / 1000
    end

    #endregion

end

function _sampleHsmmDwell(k_int::Int, m_int::Int)
    return sample(1:size(regDwell_arr[m_int], 2), Weights(regDwell_arr[m_int][k_int, :]))
end

function _nextHsmmRegime(k_int::Int, m_int::Int)
    return sample(1:numRegimes_int, Weights(regTrans_arr[m_int][k_int, :]))
end

# calendar conditioning vector for a single day (y is an index into yAll_arr)
function blockCalCont(d::Int, y::Int)
    return calCont_dic[(d, parse(Int, yAll_arr[y]))]
end

function generateForwardPath(regime_int::Int, remain_int::Int, startDoy_int::Int, n_int::Int)
    regs_arr = Vector{Int}(undef, n_int)
    rems_arr = Vector{Int}(undef, n_int)
    curReg_int = regime_int
    curRem_int = remain_int
    for step_int in 1:n_int
        month_int = doyToMonth_arr[mod1(startDoy_int + step_int, 365)]
        if curRem_int <= 0
            curReg_int = _nextHsmmRegime(curReg_int, month_int)
            curRem_int = _sampleHsmmDwell(curReg_int, month_int) - 1
        else
            curRem_int -= 1
        end
        regs_arr[step_int] = curReg_int
        rems_arr[step_int] = curRem_int
    end
    return regs_arr, rems_arr
end

function pathMeanRD(regs_arr::Vector{Int}, nDays_int::Int)
    total_fl = 0.0
    for d_int in 1:nDays_int
        doy_int   = mod1(d_int, 365)
        month_int = doyToMonth_arr[doy_int]
        reg_int   = regs_arr[d_int]
        for i_int in 1:numReg_int
            lim_tup = ext_dic[String(reg_arr[i_int])]
            norm_fl = dailyNormMean_mat[doy_int, i_int] + regCent_arr[month_int][i_int, reg_int]
            total_fl  += (norm_fl + 1.0) / 2.0 * (lim_tup[2] - lim_tup[1]) + lim_tup[1]
        end
    end
    return total_fl / nDays_int * 8760 / 1000
end

# inverse of the [−1, 1] normalization applied in preparations()
function denormalize(data_vec, ext_dic, reg_arr, dataDim_int, numReg_int)
    norm_mat = reshape(Float64.(data_vec), dataDim_int, numReg_int)
    result_df = DataFrame(hour = 1:dataDim_int)
    for (i, r) in enumerate(reg_arr)
        min_fl, max_fl = ext_dic[String(r)]
        result_df[!, r] = (norm_mat[:, i] .+ 1.0) ./ 2.0 .* (max_fl - min_fl) .+ min_fl
    end
    return result_df
end

# ! training

# compute forward pass (Shortcut for x_t)
forwardPass(x0_mat, ϵ_arr, alphaBar_arr) = sqrt.(alphaBar_arr) .* x0_mat .+ sqrt.(1.0f0 .- alphaBar_arr) .* ϵ_arr

# training step
function trainStep!(dif_mod, x0_mat, his_mat, cal_mat, regDiff_mat, lastCtx_mat, opt_state)

    # random noise step t
    t_arr = rand(1:steps_int, size(x0_mat,2))
    ϵ_arr = randn(Float32, size(x0_mat))

    # noise variation pass
    alphaBar_arr = reshape(Float32.(alphaBar_rng[t_arr]), 1, size(x0_mat,2))
    xt_arr = forwardPass(x0_mat, ϵ_arr, alphaBar_arr)
    t_mat  = reshape(Float32.(t_arr) ./ steps_int, 1, size(x0_mat,2))

    # context noise injection to prevent overfitting
    hisNoise_mat = copy(his_mat)
    hisNoise_mat .+= randn(Float32, size(his_mat)) .* 0.05f0
    hisNoise3d_mat = reshape(hisNoise_mat, dayContDim_int, numReg_int, :)

    loss_fl, grads_obj = Flux.withgradient(dif_mod) do m
        ϵPred_arr = m(xt_arr, t_mat, hisNoise3d_mat, cal_mat, regDiff_mat, lastCtx_mat)
        Flux.mse(ϵPred_arr, ϵ_arr)
    end

    Flux.update!(opt_state, dif_mod, grads_obj[1])
    return loss_fl
end

# ! model definition

# learnable per-stream scalar gain, initialized to 1.0 (a no-op) -- rescales a stream without forcing a target scale like LayerNorm
struct StreamGain
    scale::Vector{Float32}
end
Flux.@layer StreamGain
StreamGain() = StreamGain(Float32[1.0])
(g::StreamGain)(x) = g.scale[1] .* x

# network architecture
struct ConditionalDenoiser
    time_nn
    his_nn
    cal_nn
    regDiff_nn
    lastCtx_nn
    main_nn
end

Flux.@layer ConditionalDenoiser

function ConditionalDenoiser()

    # network for diffusion step
    time_nn = Chain(Dense(1, embDim1_int, relu), Dense(embDim1_int, embDim1_int), StreamGain())

    # network for context
    his_nn = Chain(
        Conv((5,), numReg_int => 32, relu, pad=SamePad()),
        Conv((3,), 32 => 64, relu, pad=SamePad()),
        Flux.flatten,
        Dense(dayContDim_int * 64, embDim1_int, relu),
        Dense(embDim1_int, embDim1_int),
        StreamGain()
    )

    cal_nn  = Chain(
        Dense(timeDim_int * 2 + 7, embDim1_int, relu),
        Dense(embDim1_int, embDim1_int, relu),
        Dense(embDim1_int, embDim1_int),
        StreamGain()
    )

    regDiff_nn = Chain(Dense(3 * numReg_int, embDim1_int, relu), Dense(embDim1_int, embDim1_int), StreamGain())

    # explicit last-context-hour conditioning: model sees boundary value directly
    lastCtx_nn = Chain(Dense(numReg_int, embDim1_int, relu), Dense(embDim1_int, embDim1_int), StreamGain())

    # main network
    main_nn = Chain(
        Dense(dayDataDim_int * numReg_int + embDim1_int * 5, embDim2_int, relu),
        Dense(embDim2_int, embDim2_int, relu),
        Dense(embDim2_int, embDim2_int, relu),
        Dense(embDim2_int, dayDataDim_int * numReg_int)
    )

    return ConditionalDenoiser(time_nn, his_nn, cal_nn, regDiff_nn, lastCtx_nn, main_nn)
end

function (nn::ConditionalDenoiser)(x_t, t_mat, his3d_mat, cal_mat, regDiff_mat, lastCtx_mat)
    t_emb   = nn.time_nn(t_mat)
    his_emb = nn.his_nn(his3d_mat)
    c_emb   = nn.cal_nn(cal_mat)
    rd_emb  = nn.regDiff_nn(regDiff_mat)
    lc_emb  = nn.lastCtx_nn(lastCtx_mat)

    in_arr = vcat(x_t, t_emb, his_emb, c_emb, rd_emb, lc_emb)

    return nn.main_nn(in_arr)
end

function historicIndex(srt_int::Int, totalLength_int::Int)
    if srt_int - dayContDim_int  > 0
        return (srt_int - dayContDim_int):(srt_int - 1)
    else
        cont1_arr = 1:(srt_int - 1)
        cont2_arr = (totalLength_int - (dayContDim_int - length(cont1_arr)) + 1):(totalLength_int)
        return vcat(collect(cont2_arr), collect(cont1_arr))
    end

end

# ! interference
# conditioning inputs hold one column per conditioning set (a vector is a single set); batch_size samples are drawn per set, stacked set-major along the output columns
function createDiff(dif_mod, his_mat, cal_mat, regDiff_mat::AbstractVecOrMat{Float32}, lastCtx_mat::AbstractVecOrMat{Float32},
                     batch_size::Int = 1;
                     dimSteps_fl::Float64 = 0.1, eta_fl::Float32 = 1.0f0)

    # pass each set's context once, then broadcast across its samples -- samples of one set share the conditioning, only the noise differs
    his_emb        = repeat(dif_mod.his_nn(reshape(his_mat, dayContDim_int, numReg_int, :)), inner = (1, batch_size))
    c_emb          = repeat(dif_mod.cal_nn(cal_mat), inner = (1, batch_size))
    regDiff_emb    = repeat(dif_mod.regDiff_nn(reshape(regDiff_mat, 3 * numReg_int, :)), inner = (1, batch_size))
    lastCtx_emb    = repeat(dif_mod.lastCtx_nn(reshape(lastCtx_mat, numReg_int, :)), inner = (1, batch_size))
    numSmp_int     = size(his_emb, 2)

    # generate initial noise
    x_mat = randn(Float32, dayDataDim_int * numReg_int, numSmp_int)

    # time sub-sequence for DDIM acceleration
    tau_arr = round.(Int, range(1, steps_int, length = Int(floor(dimSteps_fl * 1000))))

    # steps through the sub-sequence
    for idx in length(tau_arr):-1:1
        t_int = tau_arr[idx]
        tNext_int = idx > 1 ? tau_arr[idx-1] : 0

        # broadcast across batch
        t_mat = fill(Float32(t_int)/steps_int, (1, numSmp_int))
        t_emb = dif_mod.time_nn(t_mat)

        # concatenate and pass through the main dense network
        in_arr = vcat(x_mat, t_emb, his_emb, c_emb, regDiff_emb, lastCtx_emb)
        ϵPred_mat = dif_mod.main_nn(in_arr)

        # DDIM reverse step calculations
        alphaBarT_fl = Float32(alphaBar_rng[t_int])
        alphaBarNext_fl = tNext_int > 0 ? Float32(alphaBar_rng[tNext_int]) : 1.0f0
        
        # calculate stochastic noise component (sigma) for DDIM
        sigma_fl = 0.0f0
        if tNext_int > 0 && eta_fl > 0.0f0
            # DDIM generalized variance formula
            sigma_fl = eta_fl * sqrt((1.0f0 - alphaBarNext_fl) / (1.0f0 - alphaBarT_fl) * (1.0f0 - 1.0f0 * alphaBarT_fl / alphaBarNext_fl))
        end
        
        # predict x_0 from the current noisy state and predicted noise
        x0_pred = (x_mat .- sqrt(1.0f0 - alphaBarT_fl) .* ϵPred_mat) ./ sqrt(alphaBarT_fl)
        
        # compute x at the next step pointing towards the predicted x_0, plus stochastic noise
        z_mat = randn(Float32, size(x_mat))
        coef_eps = sqrt(max(0.0f0, 1.0f0 - alphaBarNext_fl - sigma_fl^2))
        x_mat = sqrt(alphaBarNext_fl) .* x0_pred .+ coef_eps .* ϵPred_mat .+ sigma_fl .* z_mat
    end

    return x_mat
end

# ! analysis
dailyAverage(hVal_arr) = mean(reshape(hVal_arr, 24, :), dims=2)

function yearlyProfile(hVal_arr::Union{Vector{Union{Missing, Float32}},Array{Float32,1}})
    startTime_tm  = DateTime(2020, 1, 1, 0)
    # cycle hour-of-year via mod so this works for any length (partial year, whole years, or whole years plus a remainder)
    times_vec     = [startTime_tm + Hour(mod(h, 8760)) for h in 0:(length(hVal_arr) - 1)]
    data_df = DataFrame(time = times_vec, value = hVal_arr)
    data_df.week = week.(data_df.time)
    return combine(groupby(data_df, [:week]), :value => mean => :avgValue)[!, :avgValue]
end

function analyseDiffResults(allRes_df::DataFrame, val_dic::Dict{Symbol, Array{Float32}}; label_str::AbstractString = "", res_dir::AbstractString = "")
    
    durErr_fl = 0.0
    dErr_fl = 0.0
    yErr_fl  = 0.0

    plDur_arr = []
    plDay_arr = []
    plYear_arr = []

    drghtThrs_arr = Float32.(0.1:0.1:0.7)
    maxDk_int     = 21
    drght_pl      = plot(layout = (length(drghtThrs_arr), numReg_int), size = (400 * numReg_int, 400 * length(drghtThrs_arr)))

    for (ri, r) in enumerate(reg_arr)

        valHis_arr = allRes_df[!, r]
        valSyn_arr = val_dic[r]

        # duration curve
        xHis_arr = collect(0:1/(length(valHis_arr)-1):1)
        yHis_arr = collect(0:1/(size(valSyn_arr,1)-1):1)
        sortHis_arr = sort(valHis_arr)
        sortSyn_arr = sort(valSyn_arr)

        interHis_pol = LinearInterpolation(xHis_arr, sortHis_arr)
        interSyn_pol = LinearInterpolation(yHis_arr, sortSyn_arr)

        durErr_fl = durErr_fl + sum(abs.(interHis_pol.(range(0, 1, length=1000)) .- interSyn_pol.(range(0, 1, length=1000))))/1000/2

        dur_pl = plot(collect(0:1/(length(valHis_arr)-1):1), sort(valHis_arr), title = string(r), label = "data")
        plot!(collect(0:1/(size(valSyn_arr,1)-1):1), sort(valSyn_arr), title = string(r), label = "diffusion model")
        push!(plDur_arr, dur_pl)

        # daily profile
        dHis_arr= dailyAverage(valHis_arr)
        dSyn_arr= dailyAverage(valSyn_arr)
        dErr_fl = sum(abs.(dHis_arr .- dSyn_arr))/24/2

        daily_pl = plot(dHis_arr, title =string(r), label = "data")
        plot!(dSyn_arr, title = string(r), label = "diffusion model")
        push!(plDay_arr, daily_pl)

        # yearly pofile
        yHis_arr= yearlyProfile(valHis_arr)
        ySyn_arr= yearlyProfile(valSyn_arr)
        # compare over the overlapping length only -- ySyn_arr may span less than a full year
        nYearComp_int = min(length(yHis_arr), length(ySyn_arr))
        yErr_fl = sum(abs.(yHis_arr[1:nYearComp_int] .- ySyn_arr[1:nYearComp_int])) / nYearComp_int / 2

        year_pl = plot(yHis_arr, title = string(r), label = "data")
        plot!(ySyn_arr, title = string(r), label = "diffusion model")
        push!(plYear_arr, year_pl)

        # dunkelflaute frequency-duration curves (VMBT, Kittel & Schill 2026)
        nTrainDays_int = length(valHis_arr) ÷ 24
        nSimDays_int   = length(valSyn_arr) ÷ 24
        trainDaily_arr = Float32[mean(valHis_arr[(d_int-1)*24+1 : d_int*24]) for d_int in 1:nTrainDays_int]
        synthDaily_arr = Float32[mean(valSyn_arr[(d_int-1)*24+1 : d_int*24]) for d_int in 1:nSimDays_int]
        nTrainYears_fl = nTrainDays_int / 365.0
        nSimYears_fl   = nSimDays_int   / 365.0
        τ_arr          = Float32[quantile(trainDaily_arr, 1.0 - Float64(iv)) for iv in drghtThrs_arr]

        for (ti_int, iVal_fl) in enumerate(drghtThrs_arr)
            sp_int     = (ti_int - 1) * numReg_int + ri
            freqTr_arr = vmbt(trainDaily_arr, τ_arr[ti_int], maxDk_int, nTrainYears_fl)
            freqSy_arr = vmbt(synthDaily_arr, τ_arr[ti_int], maxDk_int, nSimYears_fl)
            plot!(drght_pl, 2:maxDk_int, freqTr_arr[2:end]; sp = sp_int, title = "$(r)  i = $(iVal_fl)", label = "data")
            plot!(drght_pl, 2:maxDk_int, freqSy_arr[2:end]; sp = sp_int, label = "diffusion model")
        end

    end

    num_countries = length(reg_arr)
    grid_layout = (1, num_countries)

    comDur_pl = plot(plDur_arr..., layout = grid_layout, size = (400 * num_countries, 400))
    comDay_pl = plot(plDay_arr..., layout = grid_layout, size = (400 * num_countries, 400))
    comYear_pl = plot(plYear_arr..., layout = grid_layout, size = (400 * num_countries, 400))

    savefig(comDay_pl, res_dir * "dailyProfile_$(label_str).png")
    savefig(comYear_pl, res_dir * "yearlyProfile_$(label_str).png")
    savefig(comDur_pl, res_dir * "durationCurve_$(label_str).png")
    savefig(drght_pl,    res_dir * "droughts_$(label_str).png")
        savefig(comDay_pl, res_dir * "dailyProfile_$(label_str).svg")
    savefig(comYear_pl, res_dir * "yearlyProfile_$(label_str).svg")
    savefig(comDur_pl, res_dir * "durationCurve_$(label_str).svg")
    savefig(drght_pl,    res_dir * "droughts_$(label_str).svg")


    return durErr_fl,dErr_fl, yErr_fl

end

# compute the VMBT frequency-duration curve (Kittel & Schill 2026)
function vmbt(daily_arr::Vector{Float32}, thresh_fl::Float32, maxDur_int::Int, nYears_fl::Float64)
    n_int     = length(daily_arr)
    avail_vec = trues(n_int)
    count_arr = zeros(Int, maxDur_int)

    for dur_int in maxDur_int:-1:1
        t_int = dur_int
        while t_int <= n_int
            start_int = t_int - dur_int + 1
            if all(@view avail_vec[start_int:t_int]) && mean(@view daily_arr[start_int:t_int]) > thresh_fl
                count_arr[dur_int] += 1
                avail_vec[start_int:t_int] .= false
                t_int += dur_int
            else
                t_int += 1
            end
        end
    end

    return [sum(count_arr[d_int:end]) / nYears_fl for d_int in 1:maxDur_int]
end

function pfValidation(pfData_arr::Vector{nodeData}, name_str::AbstractString = "")

    pfVec_arr = map(x -> x.data, pfData_arr)

    val_dic = Dict{Symbol, Array{Float32}}()
    foreach(x -> val_dic[reg_arr[x]] = Float32[], 1:numReg_int)

    for day_vec in pfVec_arr
        forecast_mat = reshape(day_vec, dataDim_int, numReg_int)
        foreach(x -> append!(val_dic[reg_arr[x]], forecast_mat[:, x]), 1:numReg_int)
    end

    durErr_fl, dErr_fl, yErr_fl = analyseDiffResults(allRes_df, val_dic, label_str = name_str, res_dir = "results/")

    # denormalized sum across all regions
    hisSum_arr = zeros(Float32, size(data_mat, 1))
    for r in reg_arr
        min_fl, max_fl = ext_dic[String(r)]
        hisSum_arr .+= (allRes_df[!, r] .+ 1.0f0) ./ 2.0f0 .* Float32(max_fl - min_fl) .+ min_fl
    end

    synSum_arr = zeros(Float32, length(pfVec_arr) * dataDim_int)
    for (d_int, day_vec) in enumerate(pfVec_arr)
        day_df  = denormalize(day_vec, ext_dic, reg_arr, dataDim_int, numReg_int)
        rng_arr = (d_int - 1) * dataDim_int + 1 : d_int * dataDim_int
        for r in reg_arr
            synSum_arr[rng_arr] .+= Float32.(day_df[!, r])
        end
    end

    nHisDays_int  = length(hisSum_arr) ÷ 24
    nSynDays_int  = length(synSum_arr) ÷ 24
    nHisYears_fl  = nHisDays_int / 365.0
    nSynYears_fl  = nSynDays_int / 365.0

    # duration curve
    dur_pl = plot(collect(0:1/(length(hisSum_arr)-1):1), sort(hisSum_arr);
                  label = "data", title = "total RD", xlabel = "quantile", ylabel = "RD (GW)")
    plot!(collect(0:1/(length(synSum_arr)-1):1), sort(synSum_arr); label = "diffusion model")
    savefig(dur_pl, "results/durationCurve_sum_$(name_str).png")
     savefig(dur_pl, "results/durationCurve_sum_$(name_str).svg")


    # daily profile
    dHis_arr = vec(mean(reshape(hisSum_arr, 24, :), dims = 2))
    dSyn_arr = vec(mean(reshape(synSum_arr, 24, :), dims = 2))
    daily_pl = plot(dHis_arr; label = "data", title = "total RD", xlabel = "hour", ylabel = "RD (GW)")
    plot!(dSyn_arr; label = "diffusion model")
    savefig(daily_pl, "results/dailyProfile_sum_$(name_str).png")
    savefig(daily_pl, "results/dailyProfile_sum_$(name_str).svg")

    # yearly profile (weekly averages)
    year_pl = plot(yearlyProfile(hisSum_arr); label = "data", title = "total RD", xlabel = "week", ylabel = "RD (GW)")
    plot!(yearlyProfile(synSum_arr); label = "diffusion model")
    savefig(year_pl, "results/yearlyProfile_sum_$(name_str).png")
    savefig(year_pl, "results/yearlyProfile_sum_$(name_str).svg")

    # dunkelflaute on daily sums (VMBT)
    hisDaily_arr  = Float32[mean(hisSum_arr[(d_int-1)*24+1 : d_int*24]) for d_int in 1:nHisDays_int]
    synDaily_arr  = Float32[mean(synSum_arr[(d_int-1)*24+1 : d_int*24]) for d_int in 1:nSynDays_int]
    drghtThrs_arr = Float32.(0.1:0.1:0.7)
    maxDk_int     = 21
    dk_pl = plot(layout = (1, length(drghtThrs_arr)), size = (400 * length(drghtThrs_arr), 450), bottom_margin = 8Plots.mm)
    τSum_arr = Float32[quantile(hisDaily_arr, 1.0 - Float64(iv)) for iv in drghtThrs_arr]
    for (ti_int, iVal_fl) in enumerate(drghtThrs_arr)
        freqHis_arr = vmbt(hisDaily_arr, τSum_arr[ti_int], maxDk_int, nHisYears_fl)
        freqSyn_arr = vmbt(synDaily_arr, τSum_arr[ti_int], maxDk_int, nSynYears_fl)
        plot!(dk_pl, 2:maxDk_int, freqHis_arr[2:end]; sp = ti_int, title = "i = $(iVal_fl)", label = "data", xlabel = "duration (days)", ylabel = "events/year")
        plot!(dk_pl, 2:maxDk_int, freqSyn_arr[2:end]; sp = ti_int, label = "diffusion model")
    end
    savefig(dk_pl, "results/droughts_sum_$(name_str).png")
    savefig(dk_pl, "results/droughts_sum_$(name_str).svg")

    # monthly std of daily total RD anomaly: data vs diffusion model
    minVals_vec    = Float32[ext_dic[String(r)][1] for r in reg_arr]
    maxVals_vec    = Float32[ext_dic[String(r)][2] for r in reg_arr]
    seasonMean_arr = Float32[
        sum((dailyNormMean_mat[doy_int, i] + 1.0f0) / 2.0f0 *
            (maxVals_vec[i] - minVals_vec[i]) + minVals_vec[i] for i in 1:numReg_int)
        for doy_int in 1:365
    ]
    hisDoy_arr   = mod1.(1:nHisDays_int, 365)
    synDoy_arr   = mod1.(1:nSynDays_int, 365)
    hisMonth_arr = doyToMonth_arr[hisDoy_arr]
    synMonth_arr = doyToMonth_arr[synDoy_arr]
    hisAnom_arr  = hisDaily_arr .- seasonMean_arr[hisDoy_arr]
    synAnom_arr  = synDaily_arr .- seasonMean_arr[synDoy_arr]
    hisStd_vec   = Float32[std(hisAnom_arr[hisMonth_arr .== m]) for m in 1:12]
    synStd_vec   = Float32[std(synAnom_arr[synMonth_arr .== m]) for m in 1:12]
    mNames_arr   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mstd_pl = plot(1:12, hisStd_vec;
                   xticks = (1:12, mNames_arr), label = "data",
                   marker = :circle, lw = 2, ylims = (0, :auto),
                   xlabel = "month", ylabel = "std of daily total RD anomaly (GW)",
                   title  = "Monthly RD anomaly std")
    plot!(1:12, synStd_vec; label = "diffusion model", marker = :diamond, lw = 2)
    savefig(mstd_pl, "results/monthlyStd_$(name_str).png")
    savefig(mstd_pl, "results/monthlyStd_$(name_str).svg")

    return hisSum_arr, synSum_arr

end

# ! main function for training and validation

function trainDiffusion()

    # opt_state is correctly initialized here for Flux v0.15+
    opt_state = Flux.setup(Flux.Adam(1e-3), dif_mod)

    # iteration settings
    global iter_int = 0
    maxIter_int = 17500

    smoLss_fl   = Inf
    smoFac_fl   = 0.01

    trackTm_tm = now()

    while iter_int < maxIter_int
        
        iter_int += 1

        x0_mat         = fill(Float32(0.0), (dayDataDim_int * numReg_int, batchSize_int))
        his_mat        = fill(Float32(0.0), (dayContDim_int * numReg_int, batchSize_int))
        cal_mat        = fill(Float32(0.0), (timeDim_int * 2 + 7, batchSize_int))
        regDiff_mat    = fill(Float32(0.0), (3 * numReg_int, batchSize_int))

        for i in 1:batchSize_int

            # get random day and year -- the diffusion model always trains on a single calendar day
            d_int    = 1 + Int(floor(365*rand()))
            y_int    = parse(Int, rand(y_arr))
            yCnt_int = findfirst(==(y_int), parse.(Int, y_arr))-1

            # get output
            srt_int = 8760 * yCnt_int + (d_int - 1) * 24 + 1
            x0_mat[:,i] = vec(@view data_mat[srt_int : srt_int + dayDataDim_int - 1, :])

            # get historic and time context
            cont_arr = historicIndex(srt_int, size(data_mat, 1))
            his_mat[:, i] = vec(@view data_mat[cont_arr, :])
            cal_mat[:, i] = blockCalCont(d_int, yCnt_int + 1)

            # regime context for this single day: [diff(prev->day), abs(day), diff(day->next)]
            d_cnt       = yCnt_int * 365 + d_int
            prevReg_int = dayRegime_arr[max(1, d_cnt - 1)]
            curReg_int  = dayRegime_arr[d_cnt]
            nextReg_int = dayRegime_arr[min(d_cnt + 1, length(dayRegime_arr))]
            month_int   = doyToMonth_arr[d_int]
            regDiff_mat[:, i] = vcat(
                regCent_arr[month_int][:, curReg_int] .- regCent_arr[month_int][:, prevReg_int],
                regCent_arr[month_int][:, curReg_int],
                regCent_arr[month_int][:, nextReg_int] .- regCent_arr[month_int][:, curReg_int]
            )
        end

        # last context hour per region: explicit boundary conditioning
        lastCtx_mat = his_mat[dayContDim_int:dayContDim_int:dayContDim_int*numReg_int, :]

        # run training step
        curLss_fl = trainStep!(dif_mod, x0_mat, his_mat, cal_mat, regDiff_mat, lastCtx_mat, opt_state)
        smoLss_fl = smoLss_fl == Inf ? curLss_fl : smoFac_fl * curLss_fl + (1 - smoFac_fl) * smoLss_fl

        # check progress every 500 iterations
        if iter_int % 500 == 0
            println("Iteration: $iter_int | Time: $(round((now() - trackTm_tm).value / 1e3, digits=2)) | Smoothed Loss: $(round(smoLss_fl, digits=5))")
            trackTm_tm = now()
        end

    end

    return dif_mod
end

# logistic error-growth curve scaled by a given E_sat value
function targetErrorValue(leadDay_int::Int, x0_fl::Float64, eSat_fl::Float32)
    alpha_fl = -(1.0 / T_int) * log((x0_fl * (1.0 - p_fl))/(p_fl * (1.0 - x0_fl)))
    x_fl     = x0_fl / (x0_fl + (1.0 - x0_fl) * exp(-alpha_fl * leadDay_int))
    return x_fl * eSat_fl
end

# write residual demand data to csv
function printNodeData(node_dic::Dict{Int, nodeData}, pfNode_arr::Vector{nodeData}, i::Int, name_str::String = "")

    outDir_str = "results/profiles/" * run_str
    mkpath(outDir_str)

    resiData_arr = DataFrame[]
    for x in keys(node_dic)
        data_df = denormalize(node_dic[x].data, ext_dic, reg_arr, dataDim_int, numReg_int)
        data_df[!,:lvl]      .= node_dic[x].lvl
        data_df[!,:id]       .= node_dic[x].id
        data_df[!,:ancestor] .= node_dic[x].ancestor
        push!(resiData_arr, data_df)
    end
    allData_df = vcat(resiData_arr...)
    allData_df[!,:case] .= name_str
    CSV.write(outDir_str * "/nodeData_" * name_str * ".csv", allData_df)

    # plot all leaf-to-root branches
    max_lvl = maximum(x.lvl for x in values(node_dic))
    min_lvl = minimum(x.lvl for x in values(node_dic))
    leaves  = filter(x -> x.lvl == max_lvl, collect(values(node_dic)))
    n_days  = max_lvl - min_lvl + 1
    day_boundaries = collect(dataDim_int : dataDim_int : (n_days - 1) * dataDim_int)

    # collect all leaf-to-root chains once, reuse for plotting and jump metric
    chains = Vector{Vector{nodeData}}()
    for leaf in leaves
        chain = nodeData[]
        curr_id::Int = leaf.id
        while true
            pushfirst!(chain, node_dic[curr_id])
            nd = node_dic[curr_id]
            (nd.ancestor == curr_id || !haskey(node_dic, nd.ancestor)) && break
            curr_id = nd.ancestor
        end
        push!(chains, chain)
    end

    pf_dfs = [denormalize(pfNode_arr[j].data, ext_dic, reg_arr, dataDim_int, numReg_int) for j in i:min(i + n_days - 1, length(pfNode_arr))]
    pf_df  = vcat(pf_dfs...)

    xTicks_arr = 0:24:(n_days * dataDim_int)

    plts = []
    for r in reg_arr
        p = plot(title = String(r), xlabel = "Hour", ylabel = "Residual Demand (MW)", legend = false, xticks = xTicks_arr)
        for chain in chains
            chain_dfs = [denormalize(nd.data, ext_dic, reg_arr, dataDim_int, numReg_int) for nd in chain]
            branch_df = vcat(chain_dfs...)
            plot!(p, 1:nrow(branch_df), branch_df[!, r], alpha = 0.6)
        end
        plot!(p, 1:nrow(pf_df), pf_df[!, r], color = :black, linestyle = :dash, linewidth = 1.5)
        vline!(p, day_boundaries, color = :grey, alpha = 0.4, linewidth = 1)
        push!(plts, p)
    end
    p_tot = plot(title = "Total", xlabel = "Hour", ylabel = "Residual Demand (MW)", legend = false, xticks = xTicks_arr)
    for chain in chains
        chain_dfs = [denormalize(nd.data, ext_dic, reg_arr, dataDim_int, numReg_int) for nd in chain]
        branch_df = vcat(chain_dfs...)
        plot!(p_tot, 1:nrow(branch_df), sum(branch_df[!, r] for r in reg_arr), alpha = 0.6)
    end
    plot!(p_tot, 1:nrow(pf_df), sum(pf_df[!, r] for r in reg_arr), color = :black, linestyle = :dash, linewidth = 1.5)
    vline!(p_tot, day_boundaries, color = :grey, alpha = 0.4, linewidth = 1)
    push!(plts, p_tot)
    n_panels = length(reg_arr) + 1
    savefig(plot(plts..., layout = (n_panels, 1), size = (900, 300 * n_panels)),
            outDir_str * "/nodeData_" * name_str * ".png")
    savefig(plot(plts..., layout = (n_panels, 1), size = (900, 300 * n_panels)),
            outDir_str * "/nodeData_" * name_str * ".svg")

    return allData_df

end

# build [diff(prev->day), abs(day), diff(day->next)] conditioning vector from HSMM centroids, reg_vec = [prev, day, next]
function regimeDiffVec(d::Int, reg_vec::Vector{Int})
    month_int = doyToMonth_arr[mod1(d, 365)]
    vcat(
        regCent_arr[month_int][:, reg_vec[2]] .- regCent_arr[month_int][:, reg_vec[1]],
        regCent_arr[month_int][:, reg_vec[2]],
        regCent_arr[month_int][:, reg_vec[3]] .- regCent_arr[month_int][:, reg_vec[2]]
    )
end

# generate forecast profiles; his_mat/regDiff_mat hold one column per conditioning set (a vector is a single set), the output holds frs_int samples per set, set-major
function createForecasts(d, y, his_mat, dif_mod, frs_int::Int,
                          regDiff_mat::AbstractVecOrMat{Float32};
                          eta_fl::Float32 = 1.0f0, dimSteps_fl::Float64 = 0.1, corrPath::Bool = false)

    his_mat     = reshape(his_mat, dayContDim_int * numReg_int, :)
    regDiff_mat = reshape(regDiff_mat, 3 * numReg_int, :)
    numSet_int  = size(his_mat, 2)

    # all sets share the same calendar day
    cal_mat = repeat(reshape(blockCalCont(d, y), :, 1), 1, numSet_int)

    lastCtx_mat = his_mat[dayContDim_int:dayContDim_int:dayContDim_int*numReg_int, :]

    # generate all samples of all sets in a single batched pass instead of one-at-a-time
    frc_mat = createDiff(dif_mod, his_mat, cal_mat, regDiff_mat, lastCtx_mat, frs_int, dimSteps_fl = dimSteps_fl, eta_fl = eta_fl)

    # post-scale to targeted mean using atanh to prevent clipping
    if corrPath
        for j_int in 1:size(frc_mat, 2)
            set_int = cld(j_int, frs_int)
            for (i_int, _) in enumerate(reg_arr)
                dayRng_arr = (i_int - 1) * dayDataDim_int + 1 : i_int * dayDataDim_int
                doy_int    = mod1(d, 365)
                absVal_fl  = regDiff_mat[numReg_int + i_int, set_int]
                exp_mean   = dailyNormMean_mat[doy_int, i_int] + absVal_fl
                day_arr  = clamp.(frc_mat[dayRng_arr, j_int], -0.9999f0, 0.9999f0)
                shift_fl = atanh(clamp(exp_mean, -0.9999f0, 0.9999f0)) - atanh(clamp(mean(day_arr), -0.9999f0, 0.9999f0))
                frc_mat[dayRng_arr, j_int] = tanh.(atanh.(day_arr) .+ shift_fl)
            end
        end
    end

    return frc_mat
end

# mean level per region per candidate, matched against an absolute target level, not a delta vs a reference
function computeLevelMat(frc_mat::Matrix{Float32})
    return [mean(frc_mat[(k-1)*dayDataDim_int+1 : k*dayDataDim_int, j]) for k in eachindex(reg_arr), j in 1:size(frc_mat, 2)]
end

errorVec(errorSym::Symbol,vec1::Union{SubArray,Vector{Float32}}, vec2::Union{SubArray,Vector{Float32}}) = errorVec(Val{errorSym}(),vec1::Union{SubArray,Vector{Float32}}, vec2::Union{SubArray,Vector{Float32}})

# :abs is the signed forecast-error metric, mean(target - reference)
errorVec(objGrp::Val{:abs},vec1::Union{SubArray,Vector{Float32}}, vec2::Union{SubArray,Vector{Float32}}) = sum(vec1 .- vec2)/length(vec1)
# :rmse is a distance kernel comparing two already-computed error vectors, not a forecast metric itself
errorVec(objGrp::Val{:rmse},vec1::Union{SubArray,Vector{Float32}}, vec2::Union{SubArray,Vector{Float32}}) = sqrt(mean((vec1 .- vec2) .^ 2))

# plot mean +- 80% CI of targeted vs. achieved |forecast - realized| error over lead days, per region + total
function plotDetErr(trackDetErr_arr::Vector{Tuple{Int,Int,Vector{Float32},Vector{Float32}}}, run_str::AbstractString = run_str)

    lvl_arr = sort(unique(getindex.(trackDetErr_arr, 2)))
    leadDays_arr = (lvl_arr .- 1) .* daysPerNode_int

    function regionPanel(colIdx_int::Int, title_str::String)
        targetMean_arr = Float64[]; targetCi_arr = Float64[]; realMean_arr = Float64[]; realCi_arr = Float64[]
        for l in lvl_arr
            entries_arr = filter(x -> x[2] == l, trackDetErr_arr)
            target_arr  = [x[3][colIdx_int] for x in entries_arr]
            real_arr    = [x[4][colIdx_int] for x in entries_arr]
            push!(targetMean_arr, mean(target_arr))
            push!(targetCi_arr, 1.28 * std(target_arr) / sqrt(length(target_arr)))
            push!(realMean_arr, mean(real_arr))
            push!(realCi_arr, 1.28 * std(real_arr) / sqrt(length(real_arr)))
        end
        p = plot(leadDays_arr, targetMean_arr, ribbon = targetCi_arr, label = "targeted", title = title_str,
                  xlabel = "lead days", ylabel = "|error| vs realized", fillalpha = 0.2, xticks = leadDays_arr)
        plot!(p, leadDays_arr, realMean_arr, ribbon = realCi_arr, label = "realized", fillalpha = 0.2)
        return p
    end

    function totalPanel(blockRng::UnitRange{Int}, title_str::String)
        targetMean_arr = Float64[]; targetCi_arr = Float64[]; realMean_arr = Float64[]; realCi_arr = Float64[]
        for l in lvl_arr
            entries_arr   = filter(x -> x[2] == l, trackDetErr_arr)
            targetTot_arr = [mean(x[3][blockRng]) for x in entries_arr]
            realTot_arr   = [mean(x[4][blockRng]) for x in entries_arr]
            push!(targetMean_arr, mean(targetTot_arr))
            push!(targetCi_arr, 1.28 * std(targetTot_arr) / sqrt(length(targetTot_arr)))
            push!(realMean_arr, mean(realTot_arr))
            push!(realCi_arr, 1.28 * std(realTot_arr) / sqrt(length(realTot_arr)))
        end
        p = plot(leadDays_arr, targetMean_arr, ribbon = targetCi_arr, label = "targeted", title = title_str,
                  xlabel = "lead days", ylabel = "|error| vs realized", fillalpha = 0.2, xticks = leadDays_arr)
        plot!(p, leadDays_arr, realMean_arr, ribbon = realCi_arr, label = "realized", fillalpha = 0.2)
        return p
    end

    errPlts = [regionPanel(k, String(r)) for (k, r) in enumerate(reg_arr)]
    push!(errPlts, totalPanel(1:numReg_int, "total"))

    plt = plot(errPlts..., layout = (1, length(reg_arr) + 1), size = (400 * (length(reg_arr) + 1), 400))
    savefig(plt, "results/detErr_$(run_str).png")
    savefig(plt, "results/detErr_$(run_str).svg")

    err_df = DataFrame(
        run       = fill(run_str, length(trackDetErr_arr)),
        day       = getindex.(trackDetErr_arr, 1),
        lvl       = getindex.(trackDetErr_arr, 2),
        lead_days = (getindex.(trackDetErr_arr, 2) .- 1) .* daysPerNode_int,
    )
    for (k, r) in enumerate(reg_arr)
        err_df[!, String(r) * "_targeted_abserr"] = Float32[x[3][k] for x in trackDetErr_arr]
        err_df[!, String(r) * "_realized_abserr"] = Float32[x[4][k] for x in trackDetErr_arr]
    end
    CSV.write("results/detErr_$(run_str).csv", err_df)

    return plt
end
