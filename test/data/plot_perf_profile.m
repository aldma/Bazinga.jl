function hfig = plot_perf_profile( results, settings, styles )

assert( settings.qpdo )
assert( settings.qpalm || settings.osqp )

% QPDO
t_qpdo = results.qpdo.time(:);
f_qpdo = logical( results.qpdo.failed(:) );
t_qpdo(f_qpdo) = nan;
% QPALM
if settings.qpalm
    t_qpalm = results.qpalm.time(:);
    f_qpalm = logical( results.qpalm.failed(:) );
    t_qpalm(f_qpalm) = nan;
end
% OSQP
if settings.osqp
    t_osqp = results.osqp.time(:);
    f_osqp = logical( results.osqp.failed(:) );
    t_osqp(f_osqp) = nan;
end

% QPDO vs QPALM
if settings.qpalm
    T = [t_qpdo, t_qpalm];
    st = cell(1,2);
    st{1} = styles.qpdo;
    st{2} = styles.qpalm;
    
    h_qpalm = pair_perf_prof( T, st );
    drawnow
end

% QPDO vs OSQP
if settings.osqp
    T = [t_qpdo, t_osqp];
    st = cell(1,2);
    st{1} = styles.qpdo;
    st{2} = styles.osqp;
    
    h_osqp = pair_perf_prof( T, st );
    drawnow
end

% output
if settings.qpalm && settings.osqp
    hfig = cell(1,2);
    hfig{1} = h_qpalm;
    hfig{2} = h_osqp;
elseif settings.qpalm
    hfig = h_qpalm;
else
    hfig = h_osqp;
end

return
end



%=========================================================================%


function hfig = pair_perf_prof( T, styles )

    [np,ns] = size(T);
    assert( ns == 2 );

    % x-axis : performance ratio [-]
    % y-axis : fraction of problems solved [-]

    % problem-wise minimal performance
    Tmin = min( T,[],2 );

    % performance ratio = solver performance / minimal performance
    r = zeros(np,ns);
    for is = 1:ns
        r(:,is) = T(:,is) ./ Tmin;
    end
    
    % max performance ratio
    r_max = max( max( r ) );
    
    r_max_plot = 10^( ceil( log10( 1.1 * r_max ) ) );
    
    % Replace all NaN's with twice the max_ratio and sort.

    r( isnan(r) ) = 1.1 * r_max_plot;
    r = sort( r );

    % fraction of solved problems
    f = (1:np) / np;
    
    % Plot stair graphs with markers.
    plot_fn = @(x,y,style) plot( x, y,'LineStyle',style.linestyle,...
                                     'LineWidth',style.linewidth,...
                                     'DisplayName',style.name,...
                                     'Color',style.color);
                                 
    hfig = figure;
    hold on, grid on, box on
    for is = 1: ns
        [xs,ys] = stairs( r(:,is), f );
        plot_fn( xs, ys, styles{is} );
    end

    % Axis properties are set so that failures are not shown,
    % but with the max_ratio data points shown. This highlights
    % the "flatline" effect.

    xlim([0 r_max_plot]);
    ylim([0,1]);
    
    legend('show','Location','southeast')
    set(gca,'XScale','log')
    xlabel('Performance ratio')
    ylabel('Fraction of problems solved')
    

    return
end