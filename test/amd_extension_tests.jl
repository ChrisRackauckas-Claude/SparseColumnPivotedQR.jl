using AMD
using Random
using SparseArrays
using SparseColumnPivotedQR
using Test

@testset "AMD ordering matches AMD.colamd" begin
    Random.seed!(14)
    m, n = 50, 40
    A = sprand(Float64, m, n, 0.1) + sparse(1:n, 1:n, ones(n), m, n)
    q = scpqr_analyze(A; ordering = :amd).q
    @test isperm(q)
    @test q == AMD.colamd(A)
end
