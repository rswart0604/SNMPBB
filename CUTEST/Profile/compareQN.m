%----------------------- compareQN ---------------------------------------%
%
% Script to read experiment statistics
% and print outcomes to a performance profile
%
% Compares outcomes of quasi-Newton solvers
%
%-------------------------------------------------------------------------%
% 03/08/22, J.B., Initial version
% 11/14/22, J.B., Use weighted solvers
% 11/22/22, J.B., Updated experiment
% 09/26/23, J.B., Preparation for manuscript
% 10/06/23, J.B., Update to include trust-region plots
% 10/26/23, J.B., Renaming of data files
% 07/17/25, J.B., Comparison with lbfgsb

figpath = fullfile(pwd,'..','/Figs/');
statpath = '../Statistics';
addpath(genpath(statpath));

fignm1 = 'lbfgsb'; % BFGSR

%
% Which figure from the manuscript to plot
%   wfig = 1 (Figure 1)
%   wfig = 2 (Figure 2)
%

wfig = 1;


% Number of problems
% This is updated depending on which figure is desired
np = 91; %#ok<NASGU>

%
% Reading the data depending on wfig
%
if wfig == 1

    np = 91;

    [ ri1, numit1, numf1, t_aver1, ex1 ] = readData( 'LS2.stats', np );
    [ ri2, numit2, numf2, t_aver2, ex2 ] = readData( 'lbfgsb.stats', np );

    leg={   'compact', 'l-bfgs-b'};

else

    np = 161;

    [ ri1, numit1, numf1, t_aver1, ex1 ] = readData( 'bfgsLDLTR_Small.txt', np );
    [ ri2, numit2, numf2, t_aver2, ex2 ] = readData( 'bfgsTR_MS.txt', np );

    leg={   'Alg.4 (LDLTR)', 'Trust-Region'};

end


% Concatenate the data if both solvers were run on the same
% number of problems
if ri1==ri2
    ri = ri1;
    ex =[ex1 ex2];
    numit =[numit1 numit2];
    numf =[numf1 numf2];
    t_aver =[t_aver1 t_aver2];
end   

%
% Indicies of algorithms LDLTTR corresponds to the first index
%
indAlg = [1 2];


%
% Parameters for the performance plots
%
nms = length(leg);
mleg = cell(nms);
types.markers   = ['o' 'o']; %'s' 's'

% Initial line types
types.colors    = ['b' 'r' ]; %'k' 'y'
types.lines     = {'-', '-.'}; %'-',   '-'
isLog = 1;

ticks   = -2; 
ticke   = 5; 
XTick   = 2.^(ticks:ticke);
XLim    = [XTick(1),XTick(end)];
leglocation1 = 'SouthEast';
legFontSize1 = 15;
markerSize = 0;
lineWidths = [1.8, 1.6, 1.4, 1.2];
typesp.markers =['o' 'o' 'd' 'h' 'o'];

%
% Extended perf. profile for the time 
%
perf_ext_fnc(ex(1:ri,indAlg),t_aver(1:ri,indAlg),leg(indAlg),1,types,...
    leglocation1,XTick,XLim,legFontSize1,markerSize,lineWidths);

% On first plot "grid on" does not produce desired result,
% thus toggle once more
grid on; grid off;
grid on;
title('$\textnormal{Time}$','Interpreter','latex');

% Modified printing
fig                     = gcf;
fig.PaperPositionMode   = 'auto';
fig_pos                 = fig.PaperPosition;
fig.PaperSize           = [fig_pos(3) fig_pos(4)];
figname = [fignm1,'_TIME','.pdf'];

print(fig,'-dpdf',fullfile(figpath,figname),'-bestfit');

close ALL;


%
% Extended perf. profile for the iterations
%
perf_ext_fnc(ex(1:ri,indAlg),numf(1:ri,indAlg),leg(indAlg),1,types,...     % numit
    leglocation1,XTick,XLim,legFontSize1,markerSize,lineWidths);

% On first plot "grid on" does not produce desired result,
% thus toggle once more
grid on; grid off;
grid on;
title('$\textnormal{evals. objective}$','Interpreter','latex');
% title('$\textnormal{iterations}$','Interpreter','latex');

% Modified printing
fig                     = gcf;
fig.PaperPositionMode   = 'auto';
fig_pos                 = fig.PaperPosition;
fig.PaperSize           = [fig_pos(3) fig_pos(4)];
%figname = [fignm1,'_IT','.pdf'];
figname = [fignm1,'_NF','.pdf'];

print(fig,'-dpdf',fullfile(figpath,figname),'-bestfit');

close ALL;

