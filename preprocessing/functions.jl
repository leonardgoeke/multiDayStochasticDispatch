# ! training

# compute forward pass (Shortcut for x_t)
forwardPass(x0_arr, ϵ_arr, alphaBar_fl) = sqrt(alphaBar_fl) .* x0_arr .+ sqrt(1.0 - alphaBar_fl) .* ϵ_arr

# training step
function trainStep!(dif_mod, x0_mat, cont_mat, opt_state)

    # random noise step t 
    t_int = rand(1:steps_int)
    ϵ_arr = randn(Float32, size(x0_mat))
    
    # noise variation
    xt_arr = forwardPass(x0_mat, ϵ_arr, alphaBar_rng[t_int])
    t_mat = fill(Float32(t_int)/steps_int,  (1, batchSize_int))

    grads_obj = Flux.gradient(dif_mod) do m
        ϵPred_arr = m(xt_arr, t_mat, cont_mat)
        Flux.mse(ϵPred_arr, ϵ_arr)
    end
    
    # update model
    Flux.update!(opt_state, dif_mod, grads_obj[1])
    
    # return loss
    return Flux.mse(dif_mod(xt_arr, t_mat, cont_mat), ϵ_arr)
end

# ! model definition

# network architecture
struct ConditionalDenoiser
    time_nn # sub-network for noise-step 't'
    con_nn # sub-network for context
    main_nn # main network
end

# Use @layer instead of @functor for modern Flux
Flux.@layer ConditionalDenoiser

function ConditionalDenoiser()

    # network for diffusion step
    time_nn = Chain(Dense(1, embDim_int, relu), Dense(embDim_int, embDim_int))
    
    # network for context
    con_nn = Chain(Dense(contDim_int + timeDim_int * 2, embDim_int, relu), Dense(embDim_int, embDim_int))
    
    # main network
    main_nn = Chain(
        Dense(dataDim_int + embDim_int + embDim_int, 128, relu), 
        Dense(128, 128, relu), 
        Dense(128, dataDim_int)
    ) 
    
    return ConditionalDenoiser(time_nn, con_nn, main_nn)
end

function (nn::ConditionalDenoiser)(x_t, t_int, con_arr)
    t_emb = nn.time_nn(t_int)
    c_emb = nn.con_nn(con_arr)
    
    # concatenate current noisy state, time embedding, and context embedding
    in_arr = vcat(x_t, t_emb, c_emb)
    return nn.main_nn(in_arr)
end

# ! interference

function generateForecast(dif_mod, con_mat)
    
    # start with pure noise
    x_mat = randn(Float32, dataDim_int, 1)

    for t in steps_int:-1:1
   
        # predict noise
        t_mat = fill(Float32(t)/steps_int, (1, 1))
        ϵPred_mat = dif_mod(x_mat, t_mat, con_mat)

        # add noise for t > 1
        σ_fl = t > 1 ? sqrt(Float32(beta_rng[t])) : 0.0
        z_mat = randn(Float32, dataDim_int, 1)        
        
        # reverse step components
        alpha_fl = Float32(alpha_rng[t])
        alphaBar_fl = Float32(alphaBar_rng[t])
        beta_fl = Float32(beta_rng[t])
        
        # update forecast
        coeff_fl = beta_fl / sqrt(1 - alphaBar_fl)
        x_mat = (1 / sqrt(alpha_fl)) .* (x_mat .- coeff_fl .* ϵPred_mat) .+ σ_fl .* z_mat
    end

    return clamp.(vec(x_mat), 0.0, 1.0)
end