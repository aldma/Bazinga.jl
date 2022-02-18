@testset "Verbose" for T in [Float64]
    using ProximalOperators
    using ProximalAlgorithms
    using LinearAlgebra
    using Bazinga

    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    lam = R(0.1) * norm(A' * b, Inf)
    @test typeof(lam) == R

    f = LeastSquares(A, b)
    g = NormL1(lam)
    c = IdentityFunction()
    D = Bazinga.FreeSet()

    Lf = opnorm(A)^2

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ALPS" begin

        x0 = zeros(T, n)
        y0 = zeros(T, n)
        out = Bazinga.alps(f, g, c, D, x0, y0, verbose=true)
        x = out[1]
        it = out[3]
        subit = out[4]
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 10
        @test subit < 50
    end
end
