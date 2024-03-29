using ProximalAlgorithms
using ProximalOperators
using LinearAlgebra
using Bazinga
using Random
using Test

@testset "Nonconvex QP (tiny, $T)" for T in [Float64]
    Q = Matrix(Diagonal(T[-0.5, 1.0]))
    q = T[0.3, 0.5]
    low = T(-1.0)
    upp = T(+1.0)

    f = Quadratic(Q, q)
    g = IndBox(low, upp)
    c = IdentityFunction()
    D = ClosedSet(IndBox(low, upp))

    n = 2

    Lip = maximum(diag(Q))
    gamma = T(0.95) / Lip

    TOL = 1e-4

    @testset "ALPS" begin
        x0 = zeros(T, n)
        y0 = zeros(T, n)
        x0_backup = copy(x0)
        out = Bazinga.alps(f, g, c, D, x0, y0)
        x = out[1]
        it = out[3]
        subit = out[4]
        z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
        @test norm(x - z, Inf) / gamma <= TOL
        @test x0 == x0_backup
    end

    g = IndFree()

    @testset "ALPS" begin
        x0 = zeros(T, n)
        y0 = zeros(T, n)
        x0_backup = copy(x0)
        out = Bazinga.alps(f, g, c, D, x0, y0)
        x = out[1]
        it = out[3]
        subit = out[4]
        z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
        @test norm(x - z, Inf) / gamma <= TOL
        @test x0 == x0_backup
    end
end

@testset "Nonconvex QP (small, $T)" for T in [Float64]
    @testset "Random problem $k" for k = 1:5
        Random.seed!(k)

        n = 100
        A = randn(T, n, n)
        U, R = qr(A)
        eigenvalues = T(2) .* rand(T, n) .- T(1)
        Q = U * Diagonal(eigenvalues) * U'
        Q = 0.5 * (Q + Q')
        q = randn(T, n)

        low = T(-1.0)
        upp = T(+1.0)

        f = Quadratic(Q, q)
        g = IndBox(low, upp)
        c = IdentityFunction()
        D = ClosedSet(IndBox(low, upp))

        Lip = maximum(abs.(eigenvalues))
        gamma = T(0.95) / Lip

        TOL = 1e-4

        @testset "box indicator" begin
            x0 = zeros(T, n)
            y0 = zeros(T, n)
            x0_backup = copy(x0)
            out = Bazinga.alps(f, g, c, D, x0, y0)
            x = out[1]
            it = out[3]
            subit = out[4]
            z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
            @test norm(x - z, Inf) / gamma <= TOL
            @test x0 == x0_backup
        end

        g = IndFree()

        @testset "w/o box indicator" begin
            x0 = zeros(T, n)
            y0 = zeros(T, n)
            x0_backup = copy(x0)
            out = Bazinga.alps(f, g, c, D, x0, y0)
            x = out[1]
            it = out[3]
            subit = out[4]
            z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
            @test norm(x - z, Inf) / gamma <= TOL
            @test x0 == x0_backup
        end
    end
end
