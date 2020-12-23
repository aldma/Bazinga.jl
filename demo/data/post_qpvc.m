
close all
clear all %#ok
clc

addpath('~/Documents/MATLAB/matlab2tikz/src/');
settings.save_figures = true;

filename = 'qpvc';
data = csvread( [filename,'.csv'] );

fprintf('=============================================\n')
fprintf(filename);
fprintf('\n');

nx          = data(:,2);
nvc         = data(:,3);
time        = data(:,4);
iter        = data(:,5);
subiter     = data(:,6);
cviolation  = data(:,7);
optimality  = data(:,8);
cslackness  = data(:,9);
solved      = data(:,10);

maxresidual = max(cviolation,optimality);
maxresidual = max(maxresidual,cslackness);

ntests = length( time );
nsolved = sum(solved == 1);

fprintf('solved %d out of %d (%6.2f) \n',nsolved,ntests,100*nsolved/ntests)
% print_stats('time',time,'%6.4f s')
% print_stats('iteration',iter,'%6.1f')
% print_stats('sub-iterations',subiter,'%6.1f')
% print_stats('max residual',maxresidual,'%6.4e')

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



% function print_stats(name,v,format)
%     fprintf('===== %s =====\n',name);
%     fprintf(['median      : ',format,' \n'],median(v));
%     fprintf(['1st quartile: ',format,' \n'],quantile(v,0.25));
%     fprintf(['3rd quartile: ',format,' \n'],quantile(v,0.75));
%     return
% end