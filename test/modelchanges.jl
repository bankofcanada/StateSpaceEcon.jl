# This set of tests reruns the existing test, but uses example models which have been constructed 
# from the existing sample models.

function constructE1!(model::Model)
    model.flags.linear = true
    model.options.substitutions = false

    @parameters model begin
        α = 0.5
        β = 0.5
    end
    
    @variables model y
    
    @shocks model y_shk
    
    @autoexogenize model y = y_shk
    
    @equations model begin
        y[t] = α * y[t - 1] + β * y[t + 1] + y_shk[t]
    end
    
    @reinitialize model

    return model
end

function constructE2!(model::Model)
    model.flags.linear = true
    model.options.substitutions = false

    # add parameters
    @parameters model begin
        cp = [0.5, 0.02]
        cr = [0.75, 1.5, 0.5]
        cy = [0.5, -0.02]
    end
    
    # add variables: a list of symbols
    @variables model begin
        pinf
        rate
        ygap
    end
    
    # add shocks: a list of symbols
    @shocks model begin
        pinf_shk
        rate_shk
        ygap_shk
    end
    
    # autoexogenize: define a mapping of variables to shocks
    @autoexogenize model begin
        pinf = pinf_shk
        rate = rate_shk
        ygap = ygap_shk
    end
    
    # add equations: a sequence of expressions, such that
    # use y[t+1] for expectations/leads
    # use y[t] for contemporaneous
    # use y[t-1] for lags
    # each expression must have exactly one "="
    @equations model begin
        pinf[t]=cp[1]*pinf[t-1]+(.98-cp[1])*pinf[t+1]+cp[2]*ygap[t]+pinf_shk[t]
        rate[t]=cr[1]*rate[t-1]+(1-cr[1])*(cr[2]*pinf[t]+cr[3]*ygap[t])+rate_shk[t]
        ygap[t]=cy[1]*ygap[t-1]+(.98-cy[1])*ygap[t+1]+cy[2]*(rate[t]-pinf[t+1])+ygap_shk[t]
    end
    
    # call initialize! to construct internal structures
    @reinitialize model
    return model
end

function constructE3!(model::Model)
    model.flags.linear = true
    model.options.substitutions = false

    # add parameters
    @parameters model begin
        cp = [0.5, 0.02]
        cr = [0.75, 1.5, 0.5]
        cy = [0.5, -0.02]
    end
    
    # add variables: a list of symbols
    @variables model begin
        pinf
        rate
        ygap
    end
    
    # add shocks: a list of symbols
    @shocks model begin
        pinf_shk
        rate_shk
        ygap_shk
    end
    
    # autoexogenize: define a mapping of variables to shocks
    @autoexogenize model begin
        pinf = pinf_shk
        rate = rate_shk
        ygap = ygap_shk
    end
    
    # add equations: a sequence of expressions, such that
    # use y[t+1] for expectations/leads
    # use y[t] for contemporaneous
    # use y[t-1] for lags
    # each expression must have exactly one "="
    @equations model begin
        pinf[t]=cp[1]*pinf[t-1]+0.3*pinf[t+1]+0.05*pinf[t+2]+0.05*pinf[t+3]+cp[2]*ygap[t]+pinf_shk[t]
        rate[t]=cr[1]*rate[t-1]+(1-cr[1])*(cr[2]*pinf[t]+cr[3]*ygap[t])+rate_shk[t]
        ygap[t]=cy[1]/2*ygap[t-2]+cy[1]/2*ygap[t-1]+(.98-cy[1])*ygap[t+1]+cy[2]*(rate[t]-pinf[t+1])+ygap_shk[t]
    end
    
    # call initialize! to construct internal structures
    @reinitialize model
    return model
end

function constructE3nl!(model::Model)
    model.flags.linear = false
    model.options.maxiter = 200
    model.options.substitutions = false

    @parameters model begin
        cp = [0.5, 0.02]
        cr = [0.75, 1.5, 0.5]
        cy = [0.5, -0.02]
    end

    @variables model begin
        pinf
        rate
        ygap
    end

    @shocks model begin
        pinf_shk
        rate_shk
        ygap_shk
    end

    @autoexogenize model begin
        pinf = pinf_shk
        rate = rate_shk
        ygap = ygap_shk
    end

    @equations model begin
        exp(pinf[t])=exp(cp[1]*pinf[t-1]+0.3*pinf[t+1]+0.05*pinf[t+2]+0.05*pinf[t+3]+cp[2]*ygap[t]+pinf_shk[t])
        rate[t]=cr[1]*rate[t-1]+(1-cr[1])*(cr[2]*pinf[t]+cr[3]*ygap[t])+rate_shk[t]
        ygap[t]=cy[1]/2*ygap[t-2]+cy[1]/2*ygap[t-1]+(.98-cy[1])*ygap[t+1]+cy[2]*(rate[t]-pinf[t+1])+ygap_shk[t]
    end

    @reinitialize model
    return model
end

function constructE6!(model::Model)
    model.flags.linear = true
    model.options.substitutions = false

    @parameters model begin
        p_dlp = 0.0050000000000000 
        p_dly = 0.0045000000000000 
    end

    @variables model begin
        dlp; dly; dlyn; lp; ly; lyn
    end

    @shocks model begin
        dlp_shk; dly_shk
    end

    @autoexogenize model begin
        ly = dly_shk
        lp = dlp_shk
    end

    @equations model begin
        dly[t]=(1-0.2-0.2)*p_dly+0.2*dly[t-1]+0.2*dly[t+1]+dly_shk[t]
        dlp[t]=(1-0.5)*p_dlp+0.1*dlp[t-2]+0.1*dlp[t-1]+0.1*dlp[t+1]+0.1*dlp[t+2]+0.1*dlp[t+3]+dlp_shk[t]
        dlyn[t]=dly[t]+dlp[t]
        ly[t]=ly[t-1]+dly[t]
        lp[t]=lp[t-1]+dlp[t]
        lyn[t]=lyn[t-1]+dlyn[t]
    end

    @reinitialize model
    return model
end

function constructE7!(model::Model)
    model.flags.linear = false
    model.options.substitutions = true

    @parameters model begin
        delta = 0.1000000000000000 
        p_dlc_ss = 0.0040000000000000 
        p_dlinv_ss = 0.0040000000000000 
        p_growth = 0.0040000000000000 
    end

    @variables model begin
        dlc; dlinv; dly; lc; linv;
        lk; ly;
    end

    @shocks model begin
        dlc_shk; dlinv_shk;
    end

    @autoexogenize model begin
        lc = dlc_shk
        linv = dlinv_shk
    end

    @equations model begin
    dlc[t]=(1-0.2-0.2)*p_dlc_ss+0.2*dlc[t-1]+0.2*dlc[t+1]+dlc_shk[t]
    dlinv[t]=(1-0.5)*p_dlinv_ss+0.1*dlinv[t-2]+0.1*dlinv[t-1]+0.1*dlinv[t+1]+0.1*dlinv[t+2]+0.1*dlinv[t+3]+dlinv_shk[t]
    lc[t]=lc[t-1]+dlc[t]
    linv[t]=linv[t-1]+dlinv[t]
    ly[t]=log(exp(lc[t])+exp(linv[t]))
    dly[t]=ly[t]-ly[t-1]
    lk[t]=log((1-delta)*exp(lk[t-1])+exp(linv[t]))
    end

    @reinitialize model
    return model
end

function constructE7A!(model::Model)
    model.flags.linear = false
    model.options.substitutions = false

    @parameters model begin
        delta = 0.1000000000000000 
        p_dlc_ss = 0.0040000000000000 
        p_dlinv_ss = 0.0040000000000000 
        p_growth = 0.0040000000000000 
    end

    @variables model begin
        dlc; dlinv; dly; lc; linv;
        lk; ly;
    end

    @shocks model begin
        dlc_shk; dlinv_shk;
    end

    @autoexogenize model begin
        lc = dlc_shk
        linv = dlinv_shk
    end

    @equations model begin
        dlc[t] = (1 - 0.2 - 0.2) * p_dlc_ss + 0.2 * dlc[t - 1] + 0.2 * dlc[t + 1] + dlc_shk[t]
        dlinv[t] = (1 - 0.5) * p_dlinv_ss + 0.1 * dlinv[t - 2] + 0.1 * dlinv[t - 1] + 0.1 * dlinv[t + 1] + 0.1 * dlinv[t + 2] + 0.1 * dlinv[t + 3] + dlinv_shk[t]
        lc[t] = lc[t - 1] + dlc[t]
        linv[t] = linv[t - 1] + dlinv[t]
        ly[t] = log(exp(lc[t]) + exp(linv[t]))
        dly[t] = ly[t] - ly[t - 1]
        lk[t] = log((1 - delta) * exp(lk[t - 1]) + exp(linv[t]))
    end

    @reinitialize model

    @steadystate model linv = lc - 7;
    @steadystate model lc = 14;

    return model
end

function constructS1!(model::Model)
    model.flags.linear = true
    model.options.substitutions = false

    @variables model a b c

    @shocks model b_shk c_shk

    @parameters model begin
        a_ss = 1.2 
        α = 0.5 
        β = 0.8 
        q = 2
    end

    @equations model begin
        a[t] = b[t] + c[t]
        b[t] = @sstate(b) * (1 - α) + α * b[t-1] + b_shk[t]
        c[t] = q * @sstate(b) * (1 - β) + β * c[t-1] + c_shk[t]
    end

    @reinitialize model

    @steadystate model a = a_ss

    return model
end

function constructS2!(model::Model)
    model.flags.linear = false
    model.options.substitutions = false

    @parameters model begin
        α = 0.5
        x_ss = 3.1
    end

    @variables model begin
        y
        @log x
        # @shock x_shk
    end

    @shocks model begin
        x_shk
    end

    @autoexogenize model begin
        x = x_shk
    end

    @equations model begin
        y[t] = (1 - α) * 2 * @sstate(x) + (α) * @movav(y[t-1], 4)
        log(x[t]) = (1 - α) * log(x_ss) + (α) * @movav(log(x[t-1]), 2) + x_shk[t]
    end

    @reinitialize model

    return model
end

function deconstructE1!(model::Model)
    @variables model @delete y
    
    @shocks model @delete y_shk
    
    @autoexogenize model begin
        @delete y => y_shk
    end
    
    @equations model begin
        @delete _EQ1
    end
end

function deconstructE2!(model::Model)
    @variables model @delete pinf rate ygap
    
    @shocks model @delete pinf_shk rate_shk ygap_shk

    @autoexogenize model begin
        @delete pinf = pinf_shk
        @delete rate = rate_shk
        @delete ygap = ygap_shk
    end
    
    @equations model begin
        @delete _EQ1 _EQ2 _EQ3
    end
end

function deconstructE3!(model::Model)
    @variables model @delete pinf rate ygap
    
    @shocks model @delete pinf_shk rate_shk ygap_shk

    @autoexogenize model begin
        @delete pinf = pinf_shk
        @delete rate = rate_shk
        @delete ygap = ygap_shk
    end
    
    @equations model begin
        @delete _EQ1 _EQ2 _EQ3
    end
end

function deconstructE3nl!(model::Model)
    @variables model @delete pinf rate ygap
    
    @shocks model @delete pinf_shk rate_shk ygap_shk

    @autoexogenize model begin
        @delete pinf = pinf_shk
        @delete rate = rate_shk
        @delete ygap = ygap_shk
    end
    
    @equations model begin
        @delete _EQ1 _EQ2 _EQ3
    end
end

function deconstructE6!(model::Model)
    @variables model @delete dlp dly dlyn lp ly lyn
    
    @shocks model @delete dlp_shk dly_shk
    
    @autoexogenize model begin
        @delete ly => dly_shk
        @delete lp => dlp_shk
    end

    @equations model begin
        @delete _EQ1 _EQ2 _EQ3 _EQ4 _EQ5 _EQ6
    end
end

function deconstructE7!(model::Model)
    @variables model @delete dlc dlinv dly lc linv lk ly

    @shocks model @delete dlc_shk dlinv_shk

    @autoexogenize model begin
        @delete lc => dlc_shk
        @delete linv => dlinv_shk
    end

    @equations model begin
        @delete _EQ1 _EQ2 _EQ3 _EQ4 _EQ5 _EQ6 _EQ7
    end
end

function deconstructE7A!(model::Model)
    @variables model @delete dlc dlinv dly lc linv lk ly
    
    @shocks model @delete dlc_shk dlinv_shk

    @autoexogenize model begin
        @delete lc => dlc_shk
        @delete linv => dlinv_shk
    end

    @equations model begin
        @delete _EQ1 _EQ2 _EQ3 _EQ4 _EQ5 _EQ6 _EQ7
    end

    @steadystate model begin
        @delete _SSEQ1 _SSEQ2
    end
end

function deconstructS1!(model::Model)
    @variables model @delete a b c

    @shocks model @delete b_shk c_shk

    @equations model begin
        @delete _EQ1 _EQ2 _EQ3
    end

    @steadystate model begin
        @delete _SSEQ1
    end
end

function deconstructS2!(model::Model)
    @variables model @delete y x
    
    @shocks model @delete x_shk

    @autoexogenize model begin
        @delete x => x_shk
    end

    @equations model begin
        @delete _EQ1 _EQ2
    end
end


function getInitialModel(s::Symbol)
    s == :E1 && return E1.newmodel()
    s == :E2 && return E2.newmodel()
    s == :E3 && return E3.newmodel()
    s == :E3nl && return E3nl.newmodel()
    s == :E6 && return E6.newmodel()
    s == :E7 && return E7.newmodel()
    s == :E7A && return E7A.newmodel()
    s == :S1 && return S1.newmodel()
    s == :S2 && return S2.newmodel()
end

function deconstructedModel()
    symbols = (:E1, :E2, :E3, :E3nl, :E6, :E7, :E7A, :S1, :S2)
    sym = symbols[rand(1:length(symbols))]
    m = getInitialModel(sym)

    sym == :E1 && deconstructE1!(m)
    sym == :E2 && deconstructE2!(m)
    sym == :E3 && deconstructE3!(m)
    sym == :E3nl && deconstructE3nl!(m)
    sym == :E6 && deconstructE6!(m)
    sym == :E7 && deconstructE7!(m)
    sym == :E7A && deconstructE7A!(m)
    sym == :S1 && deconstructS1!(m)
    sym == :S2 && deconstructS2!(m)

    return m
end

for i in 1:3
    @info "Model change variations $i"
    global getE1() = constructE1!(deconstructedModel())
    global getE2() = constructE2!(deconstructedModel())
    global getE3() = constructE3!(deconstructedModel())
    global getE3nl() = constructE3nl!(deconstructedModel())
    global getE6() = constructE6!(deconstructedModel())
    global getE7() = constructE7!(deconstructedModel())
    global getE7A() = constructE7A!(deconstructedModel())
    global getS1() = constructS1!(deconstructedModel())
    global getS2() = constructS2!(deconstructedModel())

    include("simtests.jl")
    include("sim_fo.jl")
    include("dynss.jl")
    include("stochsims.jl")
    include("sstests.jl")
    include("misc.jl")    
end

