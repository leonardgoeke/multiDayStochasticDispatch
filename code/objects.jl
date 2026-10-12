mutable struct trackItr
    name::String
    d::Int
    y::Int
    aggRes::NamedTuple{(:crt, :thermal, :lss, :phsIn, :phsOut, :liIn, :liOut, :excLoss, :resDem), NTuple{9, Float64}}
    overallRes::DataFrame
    stLvl::DataFrame
    trackItr(name_str::String) = new(name_str, 1, 1,
        (crt = 0.0, thermal = 0.0, lss = 0.0, phsIn = 0.0, phsOut = 0.0, liIn = 0.0, liOut = 0.0, excLoss = 0.0, resDem = 0.0),
        DataFrame(day = Int[], year = Int[], crt = Float64[], thermal = Float64[], lss = Float64[], phsIn = Float64[], phsOut = Float64[], liIn = Float64[], liOut = Float64[], resDem = Float64[], excLoss = Float64[], case = String[]),
        DataFrame(day = Int[], year = Int[], hour = Int[], region = String[], technology = String[], level = Float64[], case = String[]))
end

mutable struct nodeData
    id::Int
    lvl::Int
    ancestor::Int
    data::Vector{Float32}
    prob::Float64
end