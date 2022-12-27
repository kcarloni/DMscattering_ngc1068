
using LinearAlgebra
using Interpolations

using LinearSolve

include("cross_sections.jl")

# ================ constants ===================

# using natural units --> everything has units of eV

# note: comes from setting hbar*c = 1 = ... m eV
const m_to_1oeV = 5.06773093741e6   # = [1/eV] / [m]
const cm_to_1oeV = 1e-2 * m_to_1oeV

# const kg_to_eV = 5.62e35

const pc_to_m = 3.0857e16
const kpc_to_1oeV = 1e3 * pc_to_m * m_to_1oeV
 
const GeV_to_eV = 1e9
const eV_to_GeV = 1e-9

# Avogadro's number
const N_A = 6.0221415e+23


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
    # return RHS, σ
    return UpperTriangular(C - Diagonal(σ))
end

# ================ Attenuated Flux ===================

# γ = spectral index of the initial neutrino flux
#   E^2 ϕ0 = const * (E/1GeV)^(2-γ) 
# x = DM column density
function calc_attenuated_flux(E_vec; g, mϕ, mχ, γ, t, interaction_type)

    M = get_M_RHS(E_vec; g, mϕ, mχ, interaction_type)

    U = eigvecs(M)      # U[:, i]   = the ith eigenvector
    # column-normalize:
    # for col in eachindex(U[1,:])
    #     U[:,col] /= norm(U[:,col])
    # end
    
    λ = eigvals(M)      # λ[i]      = the ith eigenvalue
    
    # no normalization
    Esq_ϕ0 = (E_vec).^(2-γ) 

    # solve the linear system: U x = Esq_ϕ0
    # c = U \ Esq_ϕ0
    prob = LinearProblem(U, Esq_ϕ0)
    c = solve(prob, SVDFactorization())

    # how bad was the solve?
    # println("cond of eigenvector matrix U: $(cond(U))")
    solve_err = sum( (U*c .- Esq_ϕ0)./Esq_ϕ0 .> 0.01 ) / length(Esq_ϕ0) 
    println("$(100*solve_err) % of solve points > 1% from true")
    # plot performance
    # p = plot(xscale=:log10, size=(200, 100))
    # plot!(E_vec, Esq_ϕ0, c=4, lw=6)
    # plot!(E_vec, U*c, c=2, lw=1.5, ls=:dash)
    # display(p)

    # need factor of 1/m
    Esq_ϕ = U * (c .* exp.(λ * t / mχ))
    return Esq_ϕ
end

# t = DM column density
# note that using too many nodes introduces numerical instability.
# return an interpolation object (function):
#   E -> flux(E), where E is in GeV
function get_f_flux(; 
    g, mϕ, mχ, γ, t, 
    num_nodes=41, logE_min=2, logE_max=6, interaction_type)

    logE_vec = range(logE_min, logE_max, num_nodes)         # log(E / GeV)
    E_vec = exp10.(logE_vec) * GeV_to_eV

    Esq_flux = calc_attenuated_flux(E_vec; g, mϕ, mχ, γ, t, interaction_type) # in eV
    interp_Esq_flux = linear_interpolation(logE_vec, Esq_flux) 

    # return linear_interpolation(logE_vec, Esq_flux)
    f_flux(E) = interp_Esq_flux(log10(E)) * (eV_to_GeV)^(2-γ) / E^2
end