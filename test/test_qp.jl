using OptiMo, Bazinga
using Printf

include("nonconvex_qp_tiny.jl")

prob = PPNCVXQP()

solver = Bazinga.PANOC( verbose=true, freq=1 )
@time out = solver( prob )
print( out )

@printf "\n\n"
solver = Bazinga.ZEROFPR( verbose=true, freq=1 )
@time out = solver( prob )
print( out )
