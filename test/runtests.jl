using Grassmann
using Test

# write your own tests here
@test (@basis e s "++++"; e124 * e23 == e134)
