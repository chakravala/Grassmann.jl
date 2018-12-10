using Multivectors
using Test

# write your own tests here
@test (@multibasis e s "++++"; e₁₂₄ * e₂₃ == e₁₃₄)
