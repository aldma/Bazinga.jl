function out = shifted_geom_mean( in, shift )
%SHIFTED_GEOM_MEAN shifted geometric mean
% out = shifted_geom_mean( in, shift )
%
% refer to 
% http://plato.asu.edu/ftp/shgeom.html, accessed 20 Nov 2020
    assert( isvector(in) )
    assert( isscalar(shift) )
    assert( all(in >= 0) )
    assert( shift >= 0 )
    n = length( in );
    out = exp( sum( log( in + shift ) )/n ) - shift;
    return
end