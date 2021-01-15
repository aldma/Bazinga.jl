
close all
clear all %#ok
clc

save_figures = false;

if save_figures
    addpath('~/Documents/MATLAB/matlab2tikz/src/');
end

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

figure
plot(nx,time,'.','MarkerSize',8)
xlabel('Problem size nx')
ylabel('Time t [s]')
set(gca,'XScale','log')
set(gca,'YScale','log')
drawnow

%% store figure
if save_figures
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
