# clustering for scenario tree construction
function clusterMedoids(pathList_arr::Vector{<:AbstractMatrix}, k_int::Int)
    feat_mat    = Float32.(hcat(vec.(pathList_arr)...))
    pca_mod     = fit(PCA, feat_mat; maxoutdim = 16)
    featRed_mat = predict(pca_mod, feat_mat)
    pcaDist_mat = pairwise(Cityblock(), featRed_mat, dims = 2)
    rawDist_mat = pairwise(Cityblock(), feat_mat, dims = 2)

    assign_arr = collect(kmedoids(pcaDist_mat, k_int).assignments)

    medoidIdx_arr = Int[]
    for g in 1:k_int
        members_arr   = findall(==(g), assign_arr)
        totalDist_arr = [sum(rawDist_mat[m, members_arr]) for m in members_arr]
        push!(medoidIdx_arr, members_arr[argmin(totalDist_arr)])
    end

    return assign_arr, medoidIdx_arr, rawDist_mat
end

# total distance of every path in a group to the medoid of the group
function groupScatter(pathList_arr::Vector{<:AbstractMatrix})
    length(pathList_arr) == 1 && return 0.0, 1
    ~, medoidIdx_arr, dist_mat = clusterMedoids(pathList_arr, 1)
    return sum(dist_mat[:, medoidIdx_arr[1]]), medoidIdx_arr[1]
end

# total distance of every path to its own cluster's medoid, given a k-way clustering result
withinClusterDist(assign_arr::Vector{Int}, medoidIdx_arr::Vector{Int}, dist_mat::AbstractMatrix) =
    sum(dist_mat[i, medoidIdx_arr[assign_arr[i]]] for i in eachindex(assign_arr))

# greedy algorithm to build budget tree
function buildBudgetTree(lvlMax_int::Int, iBudget_int::Int, allPaths_dic::Dict{Int, Matrix{Float64}})

    years_arr = collect(keys(allPaths_dic))
    n_int     = length(years_arr)

    groupAssign_dic = Dict(l => ones(Int, n_int) for l in 1:lvlMax_int)
    nextGroupId_int = 2
    remainBudget_int = iBudget_int - lvlMax_int # one node per level for the initial, fully-unsplit single chain

    while remainBudget_int > 0
        bestGain_fl         = 0.0
        bestLevel_int       = 0
        bestNewMembers_arr  = Int[]

        for l in 2:lvlMax_int
            splitCost_int = lvlMax_int - l + 1
            splitCost_int > remainBudget_int && continue

            for g in unique(groupAssign_dic[l])
                members_arr = findall(==(g), groupAssign_dic[l])
                length(members_arr) <= 1 && continue

                subPaths_arr = [allPaths_dic[years_arr[m]][((l - 1) * daysPerNode_int + 1):(lvlMax_int * daysPerNode_int), :] for m in members_arr]
                before_fl, ~ = groupScatter(subPaths_arr)
                assign2_arr, medoid2_arr, dist2_mat = clusterMedoids(subPaths_arr, 2)
                gain_fl = before_fl - withinClusterDist(assign2_arr, medoid2_arr, dist2_mat)

                if gain_fl > bestGain_fl
                    bestGain_fl        = gain_fl
                    bestLevel_int      = l
                    bestNewMembers_arr = members_arr[findall(==(2), assign2_arr)]
                end
            end
        end

        bestLevel_int == 0 && break # no beneficial, affordable split left anywhere

        for lPropagate in bestLevel_int:lvlMax_int, m in bestNewMembers_arr
            groupAssign_dic[lPropagate][m] = nextGroupId_int
        end
        nextGroupId_int  += 1
        remainBudget_int -= (lvlMax_int - bestLevel_int + 1)
    end

    return groupAssign_dic, years_arr
end

# turn the level-by-level assignment into a nodeData tree, with each non-root node's conditional probability
function assembleTree(groupAssign_dic::Dict{Int, Vector{Int}}, years_arr::Vector{Int}, lvlMax_int::Int, allPaths_dic::Dict{Int, Matrix{Float64}})

    nodeDic      = Dict{Int, nodeData}()
    nodeYear_dic = Dict{Int, Int}()
    idOf_dic     = Dict{Tuple{Int, Int}, Int}()

    nextId_int = 1
    for l in 1:lvlMax_int, g in sort(unique(groupAssign_dic[l]))
        idOf_dic[(l, g)] = nextId_int
        nextId_int += 1
    end

    for l in 1:lvlMax_int
        for g in sort(unique(groupAssign_dic[l]))
            id_int      = idOf_dic[(l, g)]
            members_arr = findall(==(g), groupAssign_dic[l])

            if l == 1
                anc_int = 0
                prob_fl = 1.0
            else
                parentGroup_int   = groupAssign_dic[l - 1][members_arr[1]]
                anc_int           = idOf_dic[(l - 1, parentGroup_int)]
                parentMembers_arr = findall(==(parentGroup_int), groupAssign_dic[l - 1])
                prob_fl           = length(members_arr) / length(parentMembers_arr)
            end

            nodeDic[id_int] = nodeData(id_int, l, anc_int, Float32[], prob_fl)

            if l > 1
                subPaths_arr = [allPaths_dic[years_arr[m]] for m in members_arr]
                ~, medoidRel_int = groupScatter(subPaths_arr)
                nodeYear_dic[id_int] = years_arr[members_arr[medoidRel_int]]
            end
        end
    end

    return nodeDic, nodeYear_dic
end

# pairwise distances for every level's tail slice, precomputed once over all years so scoring candidate structures only indexes submatrices
function levelDistances(lvlMax_int::Int, allPaths_dic::Dict{Int, Matrix{Float64}}, years_arr::Vector{Int})

    pcaDist_dic = Dict{Int, Matrix{Float64}}()
    rawDist_dic = Dict{Int, Matrix{Float64}}()

    for l in 2:lvlMax_int
        feat_mat = Float32.(hcat([vec(allPaths_dic[y][((l - 1) * daysPerNode_int + 1):(lvlMax_int * daysPerNode_int), :]) for y in years_arr]...))
        pca_mod = fit(PCA, feat_mat; maxoutdim = 16)
        pcaDist_dic[l] = pairwise(Cityblock(), predict(pca_mod, feat_mat), dims = 2)
        rawDist_dic[l] = pairwise(Cityblock(), feat_mat, dims = 2)
    end

    # full paths, used for the node's representative year exactly as assembleTree does
    fullMat_mat  = Float32.(hcat([vec(allPaths_dic[y]) for y in years_arr]...))
    fullDist_mat = pairwise(Cityblock(), fullMat_mat, dims = 2)

    return (pcaDist = pcaDist_dic, rawDist = rawDist_dic, fullDist = fullDist_mat)
end

# medoid of a member set under a precomputed distance matrix, returned as a position within members_arr
function medoidOf(members_arr::Vector{Int}, dist_mat::AbstractMatrix)
    length(members_arr) == 1 && return 1
    return argmin([sum(dist_mat[m, members_arr]) for m in members_arr])
end

# total within-group distance-to-medoid for one group, the same quantity groupScatter returns
groupScatterDist(members_arr::Vector{Int}, dist_mat::AbstractMatrix) =
    length(members_arr) <= 1 ? 0.0 : sum(dist_mat[members_arr, members_arr[medoidOf(members_arr, dist_mat)]])

# fit an already-known tree shape to the current paths, re-assigning years to branches; returns (nodeDic, nodeYear_dic, score), score = Inf if infeasible
function fitTreeToStruct(treeStruct_arr::Vector{<:Tuple}, lvlMax_int::Int, years_arr::Vector{Int}, dist_ntup)

    nNodes_int   = length(treeStruct_arr)
    children_dic = childrenOf(treeStruct_arr, nNodes_int)
    root_arr     = [id for id in 1:nNodes_int if treeStruct_arr[id][2] == 0]
    length(root_arr) == 1 || return Dict{Int, nodeData}(), Dict{Int, Int}(), Inf

    members_dic = Dict{Int, Vector{Int}}(root_arr[1] => collect(1:length(years_arr)))
    score_fl    = 0.0

    for l in 1:(lvlMax_int - 1)
        for id_int in [id for id in 1:nNodes_int if treeStruct_arr[id][1] == l]
            kids_arr = sort(children_dic[id_int])
            isempty(kids_arr) && continue
            own_arr = members_dic[id_int]

            if length(kids_arr) == 1
                members_dic[kids_arr[1]] = own_arr
                continue
            end
            # the shape demands more branches than there are years left to fill them
            length(own_arr) < length(kids_arr) && return Dict{Int, nodeData}(), Dict{Int, Int}(), Inf

            childLvl_int = l + 1
            assign_arr   = collect(kmedoids(dist_ntup.pcaDist[childLvl_int][own_arr, own_arr], length(kids_arr)).assignments)

            # order clusters deterministically so the same data always lands on the same child
            cluster_arr = [own_arr[findall(==(g), assign_arr)] for g in 1:length(kids_arr)]
            filter!(!isempty, cluster_arr)
            length(cluster_arr) < length(kids_arr) && return Dict{Int, nodeData}(), Dict{Int, Int}(), Inf
            sort!(cluster_arr, by = c -> (-length(c), c[1]))

            for (n_int, kid_int) in enumerate(kids_arr)
                members_dic[kid_int] = cluster_arr[n_int]
            end
        end
    end

    nodeDic      = Dict{Int, nodeData}()
    nodeYear_dic = Dict{Int, Int}()
    for id_int in 1:nNodes_int
        lvl_int = treeStruct_arr[id_int][1]
        anc_int = treeStruct_arr[id_int][2]
        own_arr = members_dic[id_int]

        prob_fl = anc_int == 0 ? 1.0 : length(own_arr) / length(members_dic[anc_int])
        nodeDic[id_int] = nodeData(id_int, lvl_int, anc_int, Float32[], prob_fl)

        if lvl_int > 1
            nodeYear_dic[id_int] = years_arr[own_arr[medoidOf(own_arr, dist_ntup.fullDist)]]
            score_fl += groupScatterDist(own_arr, dist_ntup.rawDist[lvl_int])
        end
    end

    return nodeDic, nodeYear_dic, score_fl
end

# children of every node, indexed by node id (root's ancestor is 0 and is not a key)
function childrenOf(treeStruct_arr::Vector{<:Tuple}, nNodes_int::Int)
    children_dic = Dict{Int, Vector{Int}}(id => Int[] for id in 1:nNodes_int)
    for id in 1:nNodes_int
        anc_int = treeStruct_arr[id][2]
        anc_int > 0 && push!(children_dic[anc_int], id)
    end
    return children_dic
end

# shape-only signature of the subtree below a node, so identical shapes get the same string regardless of ids
function subtreeSig(id_int::Int, children_dic::Dict{Int, Vector{Int}})
    isempty(children_dic[id_int]) && return "()"
    return "(" * join(sort([subtreeSig(c, children_dic) for c in children_dic[id_int]])) * ")"
end

# renumber nodes so a given tree shape always yields the same ids, level-major, siblings ordered by subtree signature
function canonicalTree(nodeDic::Dict{Int, nodeData}, nodeYear_dic::Dict{Int, Int}, lvlMax_int::Int)

    nNodes_int   = length(nodeDic)
    struct_arr   = [(nodeDic[id].lvl, nodeDic[id].ancestor) for id in 1:nNodes_int]
    children_dic = childrenOf(struct_arr, nNodes_int)
    sig_dic      = Dict(id => subtreeSig(id, children_dic) for id in 1:nNodes_int)

    # walk level by level, ordering each parent's children by signature (parents already canonical by then)
    newId_dic  = Dict{Int, Int}()
    order_arr  = [id for id in 1:nNodes_int if struct_arr[id][1] == 1]
    nextId_int = 1
    for id in order_arr
        newId_dic[id] = nextId_int
        nextId_int += 1
    end

    for l in 2:lvlMax_int
        nextLevel_arr = Int[]
        for parent_int in order_arr
            append!(nextLevel_arr, sort(children_dic[parent_int], by = c -> (sig_dic[c], c)))
        end
        for id in nextLevel_arr
            newId_dic[id] = nextId_int
            nextId_int += 1
        end
        order_arr = nextLevel_arr
    end

    newNode_dic = Dict{Int, nodeData}()
    newYear_dic = Dict{Int, Int}()
    for (oldId_int, nd) in nodeDic
        id_int  = newId_dic[oldId_int]
        anc_int = nd.ancestor == 0 ? 0 : newId_dic[nd.ancestor]
        newNode_dic[id_int] = nodeData(id_int, nd.lvl, anc_int, nd.data, nd.prob)
        haskey(nodeYear_dic, oldId_int) && (newYear_dic[id_int] = nodeYear_dic[oldId_int])
    end

    return newNode_dic, newYear_dic
end

# build the scenario tree for one call; below maxStochStruct_int the greedy runs, above it a cached shape is refitted instead
function buildStochTree(lvlMax_int::Int, iBudget_int::Int, allPaths_dic::Dict{Int, Matrix{Float64}}, setup_tup::Tuple{Int,Int})

    # the cache is shared by parallel forecast nodes, so it is only touched under the lock; the snapshot keeps clustering and refitting below lock-free
    cache_arr = lock(stochStructLock) do
        copy(get!(stochStructCache_dic, setup_tup, Vector{Vector{Tuple{Int,Int}}}()))
    end

    if length(cache_arr) < maxStochStruct_int
        groupAssign_dic, years_arr = buildBudgetTree(lvlMax_int, iBudget_int, allPaths_dic)
        nodeDic, nodeYear_dic      = assembleTree(groupAssign_dic, years_arr, lvlMax_int, allPaths_dic)
        nodeDic, nodeYear_dic      = canonicalTree(nodeDic, nodeYear_dic, lvlMax_int)

        struct_arr = [(nodeDic[id].lvl, nodeDic[id].ancestor) for id in 1:length(nodeDic)]
        # cap re-checked under the lock, since other nodes may have filled the cache in the meantime
        lock(stochStructLock) do
            shared_arr = stochStructCache_dic[setup_tup]
            length(shared_arr) < maxStochStruct_int && !(struct_arr in shared_arr) && push!(shared_arr, struct_arr)
        end
        return nodeDic, nodeYear_dic
    end

    years_arr = collect(keys(allPaths_dic))
    dist_ntup = levelDistances(lvlMax_int, allPaths_dic, years_arr)

    bestScore_fl = Inf
    bestNode_dic = Dict{Int, nodeData}()
    bestYear_dic = Dict{Int, Int}()
    for struct_arr in cache_arr
        nodeDic, nodeYear_dic, score_fl = fitTreeToStruct(struct_arr, lvlMax_int, years_arr, dist_ntup)
        if score_fl < bestScore_fl
            bestScore_fl = score_fl
            bestNode_dic = nodeDic
            bestYear_dic = nodeYear_dic
        end
    end

    # every cached shape was infeasible -- fall back to the greedy, but don't cache the result
    if isinf(bestScore_fl)
        groupAssign_dic, years2_arr = buildBudgetTree(lvlMax_int, iBudget_int, allPaths_dic)
        nodeDic, nodeYear_dic       = assembleTree(groupAssign_dic, years2_arr, lvlMax_int, allPaths_dic)
        return canonicalTree(nodeDic, nodeYear_dic, lvlMax_int)
    end

    return bestNode_dic, bestYear_dic
end

# each node's rank among its level's siblings, grouped by parent (AnyMOD's "l<lvl>s<n>" scenario-set convention)
function levelPositions(treeStruct_arr::Vector{Tuple}, lvlMax_int::Int)
    pos_dic = Dict{Int, Int}(1 => 1)

    for l in 2:lvlMax_int
        levelIds_arr = sort(filter(id -> treeStruct_arr[id][1] == l, eachindex(treeStruct_arr)), by = id -> (pos_dic[treeStruct_arr[id][2]], id))
        for (n, id) in enumerate(levelIds_arr)
            pos_dic[id] = n
        end
    end

    return pos_dic
end

# write tree structure to matrix
function treeStructMat(treeStruct_arr::Vector{Tuple}, lvlMax_int::Int)
    pos_dic = levelPositions(treeStruct_arr, lvlMax_int)
    leafIds_arr = sort(filter(id -> treeStruct_arr[id][1] == lvlMax_int, eachindex(treeStruct_arr)), by = id -> pos_dic[id])
    nameId_arr = Vector{Tuple{String,Int}}()
    struct_mat = Matrix{String}(undef, length(leafIds_arr), lvlMax_int)
    for (r, leafId) in enumerate(leafIds_arr)
        id = leafId
        for l in lvlMax_int:-1:1
            name_str = "l" * string(l) * "s" * string(pos_dic[id])
            struct_mat[r, l] = name_str
            push!(nameId_arr, (name_str, id))
            id = treeStruct_arr[id][2]
        end
    end

    return struct_mat, nameId_arr
end

# write a tree's structure to set_scenario.csv, in the layout AnyMOD's createDispatch reads
function writeScenarioSet(treeStruct_arr::Vector{Tuple}, lvlMax_int::Int, dir_str::String)
    struct_mat, nameId_arr = treeStructMat(treeStruct_arr, lvlMax_int)
    struct_df  = DataFrame(struct_mat, ["scenario_" * string(l) for l in 1:lvlMax_int])
    CSV.write(dir_str * "/set_scenario.csv", struct_df)
    return nameId_arr
end