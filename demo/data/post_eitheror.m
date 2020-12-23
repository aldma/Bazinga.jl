
close all
clear all %#ok
% clc

addpath('~/Documents/MATLAB/matlab2tikz/src/');
settings.save_figures = true;

filename = 'eitheror_grid';
data = csvread( [filename,'.csv'] );

fprintf('=============================================\n')
fprintf(filename);
fprintf('\n');

xi = data(:,2:3);
xf = data(:,4:5);
iter = data(:,6);
time = data(:,7);
solved = data(:,8);

xmin = min(xi(:));
xmax = max(xi(:));
lightgray = [0.83, 0.83, 0.83];

x22 = [2; -2];
x44 = [4; 4];

ntests = length( time );
nsolved = sum(solved == 1);

fprintf('solved %d out of %d (%6.2f) \n',nsolved,ntests,100*nsolved/ntests)
print_stats('time',time,'%6.4f s')
print_stats('iteration',iter,'%6.1f')

%%
i22 = zeros(ntests,1);
i44 = zeros(ntests,1);
for i=1:ntests
    xtmp = xf(i,:);
    xtmp = xtmp(:);
    xbas = xi(i,:);
    xbas = xbas(:);
    if isclose(xtmp, x22)
        i22(i) = 1;
    elseif isclose(xtmp, x44)
        i44(i) = 1;
    end
end
i22 = logical( i22 );
i44 = logical( i44 );
i00 = ~i22 & ~i44;

x22all = xi(i22,:);
x44all = xi(i44,:);
x00all = xf(i00,:);

c22 = sum( i22 );
c44 = sum( i44 );

fprintf('n problems %d \n',ntests)
fprintf('n x22 %5d (%6.4f / 100) \n',c22,100*c22/ntests)
fprintf('n x44 %5d (%6.4f / 100) \n',c44,100*c44/ntests)

figure
hold on, box on
plot_feas_set();
if ~isempty(x22all)
    plot(x22all(:,1),x22all(:,2),'bo','MarkerSize',4)
end
if ~isempty(x44all)
    plot(x44all(:,1),x44all(:,2),'rx','MarkerSize',4)
end
if ~isempty(x00all)
    plot(x00all(:,1),x00all(:,2),'kv','MarkerSize',4)
end
xlim([-4,8])
ylim([-4,8])
drawnow

%% store figures
if isfield(settings,'save_figures') && settings.save_figures
    fprintf('saving figures...');
    fig_style = get_figure_style();
    
    print_tikz = @(figname) matlab2tikz( [figname,'.tikz'],...
                                      'width', '\columnwidth',...
                                      'height', '\columnwidth',...
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

function plot_feas_set()
    lightgray = [0.83, 0.83, 0.83];
    xmin = -4;
    xmax = 8;
    xv = 0;
    yv = 0;
    rad = sqrt(10);
    for i = 1:50
        xx = 2.0 * i / 50;
        yy = 1.0 - sqrt(10.0 - (xx - 3)^2);
        xv = [xv; xx];
        yv = [yv; yy];
    end
    xv = [xv; 2];
    yv = [yv; 3];
    xv = [xv; 4];
    yv = [yv; 4];
    for i = 1:50
        xx = 4 + (xmax - 4) * i / 50;
        yy = (xx^2) / 4;
        xv = [xv; xx];
        yv = [yv; yy];
    end
    xv = [xv; xmin];
    yv = [yv; xmax];
    for i = 0:50
        xx = xmin * (1 - i / 50);
        yy = (xx^2) / 4;
        xv = [xv; xx];
        yv = [yv; yy];
    end
    %
    fill(xv,yv,lightgray,...
        'EdgeColor',lightgray,'FaceAlpha',0.7,'EdgeAlpha',0.7)
    % plot([0,0],[5,5*sqrt(2)],'-','Color',lightgray,'LineWidth',1.4)
end

function bool = isclose(x,y)
    tol = 1e-3;
    bool = norm(x(:) - y(:),inf) <= tol;
    return
end

function print_stats(name,v,format)
    fprintf('===== %s =====\n',name);
    fprintf(['median      : ',format,' \n'],median(v));
    fprintf(['1st quartile: ',format,' \n'],quantile(v,0.25));
    fprintf(['3rd quartile: ',format,' \n'],quantile(v,0.75));
    return
end