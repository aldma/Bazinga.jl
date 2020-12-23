
close all
clear all %#ok
clc

addpath('~/Documents/MATLAB/matlab2tikz/src/');
settings.save_figures = true;

data_a4 = csvread( 'cutest_alpx_4.csv' );
data_a6 = csvread( 'cutest_alpx_6.csv' );
data_a8 = csvread( 'cutest_alpx_8.csv' );
data_i4 = csvread( 'cutest_ipopt_4.csv' );
data_i6 = csvread( 'cutest_ipopt_6.csv' );
data_i8 = csvread( 'cutest_ipopt_8.csv' );

fprintf('=============================================\n')

max_time = 100;
time_shift = 1;

fprintf('TOL 4 \n')
print_stats( data_a4, max_time, time_shift );
print_stats( data_i4, max_time, time_shift );
fprintf('TOL 6 \n')
print_stats( data_a6, max_time, time_shift );
print_stats( data_i6, max_time, time_shift );
fprintf('TOL 8 \n')
print_stats( data_a8, max_time, time_shift );
print_stats( data_i8, max_time, time_shift );

%%


figure
plot(nx,time,'.','MarkerSize',8)
xlabel('Problem size nx')
ylabel('Time t [s]')
set(gca,'XScale','log')
set(gca,'YScale','log')
drawnow


if isfield(settings,'save_figures') && settings.save_figures
    fprintf('saving figures...');
    fig_style = get_figure_style();
    
    print_tikz = @(figname) matlab2tikz( [figname,'.tikz'],...
                                      'width', '\columnwidth',...
                                      'height', '0.618\columnwidth',...
                                      'checkForUpdates', false,...
                                      'parseStrings', false,...
                                      'showInfo', false );
    figname = filename;
    % apply figure style
    hgexport(gcf,'dummy',fig_style,'applystyle', true);
    % store either .tikz  or .fig
    print_tikz( figname );
    pause(0.5)
    fprintf(' done!\n')
    close
else
    fprintf('NOT saving figures...\n');
end

% end of file
fprintf('\nThat`s all folks!\n')

function print_stats(data,maxtime,timeshift)
    time = data(:,2);
    solved = logical( data(:,6) );
    timep = time;
    timep(~solved) = maxtime;
    tsgm = shifted_geom_mean( timep, timeshift );
    fail = sum(~solved) / length(solved);
    fprintf('time [sgm]  : %8.4f s \n',tsgm)
    fprintf('failure rate: %8.4f \n',fail)
    return
end

function hfig = plot_data_profile( results, settings, styles )

assert( settings.qpdo || settings.qpalm || settings.osqp )

if settings.qpdo
    t1 = results.qpdo.time(:);
    f1 = logical( results.qpdo.failed(:) );
    t1(f1) = nan;
else
    t1 = [];
end
if settings.qpalm
    t2 = results.qpalm.time(:);
    f2 = logical( results.qpalm.failed(:) );
    t2(f2) = nan;
else
    t2 = [];
end
if settings.osqp
    t3 = results.osqp.time(:);
    f3 = logical( results.osqp.failed(:) );
    t3(f3) = nan;
else
    t3 = [];
end

T = [t1, t2, t3];
[np,ns] = size(T);

% x-axis : time [s]
% y-axis : fraction of problems solved [-]

min_time = min(min( T ));
max_time = max(max( T ));
max_time_plot = 1.1 * max_time;

% pad failed problems
T(isnan(T)) = 2*max_time;

% budget
t_npts = 1000;
t_vec_data = sort( unique( T(:) ) );
t_vec_pts = logspace( log10(min_time), log10(max_time_plot), t_npts - length(t_vec_data) );
t_vec = [t_vec_data(:); t_vec_pts(:)];
t_vec = sort( unique( t_vec ) );
nt = length(t_vec);

% for each solver, determine fraction of solved problems
f_mat = zeros(nt,ns);
for it = 1:nt
    for is = 1:ns
        f_mat(it,is) = sum( T(:,is) <= t_vec(it) ) / np;
    end
end

plot_tf_fn = @(t,f,style) plot( t, f,'LineStyle',style.linestyle,...
                                     'LineWidth',style.linewidth,...
                                     'DisplayName',style.name,...
                                     'Color',style.color);

hfig = figure;
hold on, grid on, box on
if settings.qpalm
    plot_tf_fn( t_vec, f_mat(:,2), styles.qpalm);
end
if settings.osqp
    plot_tf_fn( t_vec, f_mat(:,3), styles.osqp);
end
if settings.qpdo
    plot_tf_fn( t_vec, f_mat(:,1), styles.qpdo);
end
xlim([0 max_time_plot])
ylim([0,1])
legend('show','Location','northwest')
set(gca,'XScale','log')
xlabel('Time [s]')
ylabel('Fraction of problems solved')
drawnow

return
end

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

function s = get_solver_style()

s.qpdo = struct(...
                'name','QPDO',...
                'color',[252, 141, 98]/256,...
                'linestyle','-',...
                'linewidth',1.4,...
                'marker','o',...
                'markersize',6);
s.qpalm = struct(...
                'name','QPALM',...
                'color',[141, 160, 203]/256,...
                'linestyle','--',...
                'linewidth',1.4,...
                'marker','s',...
                'markersize',6);
s.osqp = struct(...
                'name','OSQP',...
                'color',[102, 194, 165]/256,...
                'linestyle',':',...
                'linewidth',1.4,...
                'marker','x',...
                'markersize',6);
return
end