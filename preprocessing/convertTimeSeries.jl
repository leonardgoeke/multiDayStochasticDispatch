using CSV, DataFrames

tsDir_str = "C:/Users/lgoeke/OneDrive - ETH Zurich/TheModel - Documentation/input_daten/timeSeries/database/pecd_v3.1/"

relCtr_str = ["CH","DE","FR"]
relYear_arr = string.(collect(1982:2016))
allFile_arr = filter(x -> any(occursin.(relYear_arr, x)), readdir(tsDir_str))


# convert demand time-series
for fDem in filter(x -> occursin("demand", x) && occursin("country", x), allFile_arr)
    fDem_df = rename(CSV.read(joinpath(tsDir_str, fDem), DataFrame), :country => :region_1, :hour => :timestep_4)
    
    fDem_df[!, :parameter] .= "dem"
    fDem_df[!, :carrier_1] .= "electricity"
    fDem_df[!, :timestep_4] .= map(x -> string("h",lpad(x, 4, "0")), fDem_df[!, :timestep_4])

    filter!(x -> any(occursin.(relCtr_str, x.region_1)), fDem_df)

    y_str = split(split(fDem, "_")[end], ".")[1]
    fDem_df[!, :scenario] .= "scr" * y_str

    CSV.write("modelData/planning/timeSeries/par_demand_3ctr_" * y_str * ".csv", select(fDem_df, [:region_1, :carrier_1, :timestep_4, :scenario, :parameter, :value]))
end

# convert renewable time-series
for fRe in filter(x -> any(occursin.(["solarAva","windOnshore"], x)) && occursin("cluster", x), allFile_arr)
    println(fRe)

    reCase_str = occursin("solarAva", fRe) ? "pv" : "wind"

    fRe_df = rename(CSV.read(joinpath(tsDir_str, fRe), DataFrame), :cluster => :region_1, :hour => :timestep_4)
    fRe_df[!,:region_1] .= map(x -> x[3:end], fRe_df[!, :region_1])

    filter!(x -> any(occursin.(relCtr_str, x.region_1)), fRe_df)
    fRe_df = combine(x-> (timestep_4 = x.timestep_4[1], value = sum(x.value)/length(x.value)), groupby(fRe_df, [:region_1, :timestep_4]))

    fRe_df[!, :parameter] .= "avaConv"
    fRe_df[!, :technology_1] .= reCase_str
    fRe_df[!, :timestep_4] .= map(x -> string("h",lpad(x, 4, "0")), fRe_df[!, :timestep_4])

    y_str = split(split(fRe, "_")[end], ".")[1]
    fRe_df[!, :scenario] .= "scr" * y_str

    CSV.write("modelData/planning/timeSeries/par_" * reCase_str * "_3ctr_" * y_str * ".csv", select(fRe_df, [:region_1, :technology_1, :timestep_4, :scenario, :parameter, :value]))
end