
using LinearAlgebra
using Interpolations

using LinearSolve

include("constants.jl")
include("cross_sections.jl")

# the approach below solves the cascade equation for the attenuated flux,
# as described in 
#   https://arxiv.org/pdf/1706.09895.pdf
#   https://github.com/aaronvincent/nuFATE

# ================ Cascade Equation for Attenuation ===================

# E_vec is a vector of energy values (in eV); 
#   "nodes" of the linear system
function get_M_RHS(E_vec; g, mϕ, mχ, interaction_type::Type{T}) where (T <: Interaction)

    f_σ = (E -> get_σ(interaction_type)(E; g, mϕ, mχ))
    f_dσ_dE = ((E_i, E_f) -> (E_i > E_f) ? get_dσ_dE(interaction_type)(E_i, E_f; g, mϕ, mχ) : zero(E_i) )

    σ = [f_σ(E) for E in E_vec]
    dσ_dE = [f_dσ_dE(E0, E1) for E1 in E_vec, E0 in E_vec]

    ΔlogE = log.(E_vec[2:end]) .- log.(E_vec[1:end-1])

    # only terms above the diagonal
    C = zero(dσ_dE)
    N = length(E_vec)
    for j in 2:N, i in 1:(j-1)
        C[i,j] =  ΔlogE[j-1] * dσ_dE[i,j] * E_vec[j]^-1 * E_vec[i]^2
    end

    return UpperTriangular(C - Diagonal(σ))
end

# ================ Attenuated Flux ===================

# flux_f:   source flux, as a function of E (in eV)
# t:        dark matter column density.
function calc_attenuated_flux(E_vec; flux_f, t,
    g, mϕ, mχ, interaction_type)

    M = get_M_RHS(E_vec; g, mϕ, mχ, interaction_type)
    U = eigvecs(M)      # U[:, i]   = the ith eigenvector
    # column-normalize:
    # for col in eachindex(U[1,:])
    #     U[:,col] /= norm(U[:,col])
    # end
    λ = eigvals(M)      # λ[i]      = the ith eigenvalue
    
    # use flux * energy squared for numerical stability.
    EsqFlux_vec = (@. E_vec^2 * flux_f(E_vec))

    # solve the linear system: Uc = E²ϕ(E)
    prob = LinearProblem(U, EsqFlux_vec)
    c = solve(prob, SVDFactorization())

    # how bad was the solve?
    # println("cond of eigenvector matrix U: $(cond(U))")
    solve_err = sum( (U*c .- EsqFlux_vec)./EsqFlux_vec .> 0.01 ) / length(E_vec) 
    if solve_err*100 > 0.
        println("$(100*solve_err) % of solve points > 1% from true")
        # plot performance
        # p = plot(xscale=:log10, size=(200, 100))
        # plot!(E_vec, Esq_ϕ0, c=4, lw=6)
        # plot!(E_vec, U*c, c=2, lw=1.5, ls=:dash)
        # display(p)
    end

    EsqFlux_sol = U * (c .* exp.(λ * t / mχ))
    return EsqFlux_sol
end

"""
    get_att_flux_f(flux_f; <keyword args>)

    Compute the attenuated flux function by linearizing the cascade equation 
        at energy nodes, solving the linear system, and interpolating.

    Arguments:
        'flux_f'            source flux, as a function of [E in GeV]
                            dimensionful quantities of the flux should be factored out, ie.
                    
                                ϕ(E) = ϕ(E0) * flux_f(E)
                                flux_f(E) = ϕ(E) / ϕ(E0)

        'logE_min/max'      log of the min/max energy (in GeV)
        'interaction_type'  dark matter model
        'g, mϕ, mχ'         dark matter model parameters
        't'                 dark matter column density     
        'num_nodes'         number of interpolation nodes. 
                            note using too many may cause numerical instabilities.
"""
function get_att_flux_f(flux_f; t,
    g, mϕ, mχ, interaction_type,
    num_nodes=41, logE_min=2, logE_max=6.2)

    logE_vec = range(logE_min, logE_max, num_nodes)         # log(E / GeV)
    E_vec = exp10.(logE_vec) * GeV_to_eV

    flux_f_in_eV(E) = flux_f(E * eV_to_GeV)                 # 1
    EsqFlux = calc_attenuated_flux(E_vec; 
            flux_f=flux_f_in_eV, t, 
            g, mϕ, mχ, interaction_type)                    # eV²
    
    interp_EsqFlux = linear_interpolation(logE_vec, EsqFlux)
    att_flux_f(E) = interp_EsqFlux(log10(E)) / (E * GeV_to_eV)^2 
end