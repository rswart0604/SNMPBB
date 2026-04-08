function [ ri, numit, numf, t_aver, ex ] = readData( filename, np )
%readData reads the data from a CUTEST experiment in BFGSDEV
% stores number of iterations, function evaluations
% time and exit flag
%
%
% INPUTS:
% filename: Name of file
% np: Estimate of the number of problems
%
% OUTPUTS:
% ri: Index of last read problem
% numit: Number of iterations
% numf: Number of function evaluations
% t_aver: Computation time
% ex: Exit flag
%
%--------------------------------------------------------------------------
% 03/08/22, J.B., 
% 11/14/22, J.B., Update in the comment for how one line looks like

fid = fopen(filename);

% Storage containers
indAlg = 1;
nms = length(indAlg);
ex = zeros(np,nms);
t_aver = zeros(np,nms);
numit = zeros(np,nms);
numf = zeros(np,nms);

ri = 0;

line = fgets(fid);
% Read lines
while line~=-1
        
    %if line~=-1
       
        ri = ri+1;
        
        strsp = strsplit(line);
        
        % A line looks like
        % 'ALLINITU'    '4'    '14'    '15'    '0.009'    'Optimal'    ''
        % or it may look like (if condition number is included)
        % 'ALLINITU'    '4'    '14'    '15'    '0.009' '0'   'Optimal'    ''

        numi = str2double(strsp{3});
        nf = str2double(strsp{4});
        timep = str2double(strsp{5});
                
        if isnan(numi)==false
           
            numit(ri,indAlg) = numi;
            numf(ri,indAlg) = nf;
            t_aver(ri,indAlg) = timep;
            ex(ri,indAlg) = 1;
            
        else
            
            numit(ri,indAlg) = -1;
            numf(ri,indAlg) = -1;
            t_aver(ri,indAlg) = -1;
            ex(ri,indAlg) = 0;
            
        end
        
    %end
    
    line = fgets(fid);
    
end

fclose(fid);

end

