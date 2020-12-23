
close all
clear all %#ok
% clc

addpath('~/Documents/MATLAB/matlab2tikz/src/');
settings.save_figures = false;

filename = 'mpvca_4_grid';
% filename = 'mpvca_5_grid';
% filename = 'mpvca_5g_grid';
data = csvread( [filename,'.csv'] );

fprintf('=============================================\n')
fprintf(filename);
fprintf('\n');

xi = data(:,2:3);
xf = data(:,4:5);
iter = data(:,6);
time = data(:,7);

xmax = max(xi(:));
lightgray = [0.83, 0.83, 0.83];

x00 = [0; 0];
x05 = [0; 5];

npts = length( time );

fprintf('median time: %6.4f s \n',median(time))
fprintf('1st quartile time: %6.4f s \n',quantile(time,0.25))
fprintf('3rd quartile time: %6.4f s \n',quantile(time,0.75))

% figure
% histogram( iter )
% title('iterations')
% drawnow
% 
% figure
% histogram( log10(time) )
% title('log10( time )')
% drawnow

x00all = [];
x05all = [];
for i=1:npts
    xtmp = xf(i,:);
    xtmp = xtmp(:);
    xbas = xi(i,:);
    xbas = xbas(:);
    if isclose(xtmp, x00)
        x00all = [x00all; xi(i,:)]; %#ok
    elseif isclose(xtmp, x05)
        x05all = [x05all; xi(i,:)]; %#ok
    end
end

c00 = size(x00all,1);
c05 = size(x05all,1);
fprintf('n problems %d \n',npts)
fprintf('n x00 %d (%6.4f / 100) \n',c00,100*c00/npts)
fprintf('n x05 %d (%6.4f / 100) \n',c05,100*c05/npts)

figure
hold on, box on
fill([0,0,5*sqrt(2),xmax,xmax,0],[xmax,5*sqrt(2),0,0,xmax,xmax],lightgray,...
    'EdgeColor',lightgray,'FaceAlpha',0.7,'EdgeAlpha',0.7)
plot([0,0],[5,5*sqrt(2)],'-','Color',lightgray,'LineWidth',1.4)
if ~isempty(x00all)
    plot(x00all(:,1),x00all(:,2),'bo')
end
if ~isempty(x05all)
    plot(x05all(:,1),x05all(:,2),'rx')
end
drawnow

%% store figures
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
    grid on, box on
    hgexport(gcf,'dummy',fig_style,'applystyle', true);
    % store either .tikz  or .fig
    try
            print_tikz( figname );
        catch
            fprintf('Unable to store .tikz figure. Skipped.\n')
            savefig( figname );
    end
    pause(0.5)
    fprintf(' done!\n')
else
    fprintf('NOT saving figures...\n');
end

% end of file
fprintf('\nThat`s all folks!\n')



function bool = isclose(x,y)
    tol = 1e-3;
    bool = norm(x(:) - y(:),inf) <= tol;
    return
end