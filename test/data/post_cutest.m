
close all
clear all %#ok
clc

addpath('~/Documents/MATLAB/matlab2tikz/src/');
settings.save_figures = true;

styles = get_solver_style();

data = struct();
data.alpx_4  = get_data_from_file( 'cutest_alpx_4' );
data.alpx_6  = get_data_from_file( 'cutest_alpx_6' );
data.alpx_8  = get_data_from_file( 'cutest_alpx_8' );
data.ipopt_4 = get_data_from_file( 'cutest_ipopt_4' );
data.ipopt_6 = get_data_from_file( 'cutest_ipopt_6' );
data.ipopt_8 = get_data_from_file( 'cutest_ipopt_8' );
fprintf('data loaded! \n')

% cut-off at max time
max_time = 100; % [s]

dn = fieldnames(data);
for k=1:numel(dn)
    data.(dn{k}) = apply_cutoff_maxtime( data.(dn{k}), max_time );
end

% print statistics
time_shift = 1; % [s]

for k=1:numel(dn)
    print_stats( data.(dn{k}), max_time, time_shift );
end

%% plot figure

figure
subplot(3,2,1)
title('tol 4 - perf')
plot_perf_profile( data.alpx_4, data.ipopt_4, styles );
ylabel('Fraction of problems solved')
subplot(3,2,3)
title('tol 6 - perf')
plot_perf_profile( data.alpx_6, data.ipopt_6, styles );
ylabel('Fraction of problems solved')
subplot(3,2,5)
title('tol 8 - perf')
plot_perf_profile( data.alpx_8, data.ipopt_8, styles );
xlabel('Performance ratio \tau')
ylabel('Fraction of problems solved')
subplot(3,2,2)
title('tol 4 - data')
plot_data_profile( data.alpx_4, data.ipopt_4, styles );
subplot(3,2,4)
title('tol 6 - data')
plot_data_profile( data.alpx_6, data.ipopt_6, styles );
subplot(3,2,6)
title('tol 8 - data')
plot_data_profile( data.alpx_8, data.ipopt_8, styles );
xlabel('Time t [s]')
drawnow

%% save figure
fprintf('saving figures...');
fig_style = get_figure_style();

print_tikz = @(figname) matlab2tikz( [figname,'.tikz'],...
                                  'width', '\columnwidth',...
                                  'height', '0.618\columnwidth',...
                                  'checkForUpdates', false,...
                                  'parseStrings', false,...
                                  'showInfo', false );
% apply figure style
hgexport(gcf,'dummy',fig_style,'applystyle', true);
% store either .tikz  or .fig
cleanfigure();
pause(0.5)
print_tikz( 'cutest' );
pause(0.5)
fprintf(' done!\n')

% end of file
fprintf('\nThat`s all folks!\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = get_data_from_file( filename )
    out = struct();
    out.filename = filename;
    filedata = csvread( [filename,'.csv'] );
    out.name = filename;
    out.time = filedata(:,2);
    out.solved = logical( filedata(:,6) );
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = apply_cutoff_maxtime( data, max_time )
    idx = (data.time > max_time);
    if any(idx)
        data.time(idx) = max_time;
        data.solved(idx) = false;
    end
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function print_stats(in,maxtime,timeshift)
    t = in.time;
    s = in.solved;
    t(~s) = maxtime;
    tsgm = shifted_geom_mean( t, timeshift );
    fail = sum(~s) / length(s);
    fprintf('%s \n', in.name)
    fprintf('time [sgm]  : %8.3f s \n',tsgm)
    fprintf('solved      : %d/%d \n',sum(s),length(s))
    fprintf('failure rate: %8.3f \n',fail)
    fprintf('\n')
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_data_profile( dalpx, dipopt, styles )

t1 = dalpx.time(:);
s1 = logical( dalpx.solved(:) );
t1(~s1) = nan;

t2 = dipopt.time(:);
s2 = logical( dipopt.solved(:) );
t2(~s2) = nan;

T = [t1, t2];
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

hold on, box on

plot_tf_fn( t_vec, f_mat(:,1), styles.alpx);

plot_tf_fn( t_vec, f_mat(:,2), styles.ipopt);

xlim([0 max_time_plot])
ylim([0,1])
legend('show','Location','southeast')
set(gca,'XScale','log')

return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_perf_profile( dalpx, dipopt, styles )

t1 = dalpx.time(:);
s1 = logical( dalpx.solved(:) );
t1(~s1) = nan;

t2 = dipopt.time(:);
s2 = logical( dipopt.solved(:) );
t2(~s2) = nan;

T = [t1, t2];
st = cell(1,2);
st{1} = styles.alpx;
st{2} = styles.ipopt;
    
np = size(T,1);
ns = 2;

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
% r_max = max( max( r ) );
r_max_plot = 1e3; % 10^( ceil( log10( 1.1 * r_max ) ) );

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

hold on, box on
for is = 1: ns
    [xs,ys] = stairs( r(:,is), f );
    plot_fn( xs, ys, st{is} );
end

% Axis properties are set so that failures are not shown,
% but with the max_ratio data points shown. This highlights
% the "flatline" effect.

xlim([0 r_max_plot]);
ylim([0,1]);

legend('show','Location','southeast')
set(gca,'XScale','log')

return
end