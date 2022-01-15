function root_mean_square_error = RMSE(state_sequence1,state_sequence2)
%RMSE calculates the root mean square error between state_sequence1 and
%state_sequence2 
%INPUT: state_sequence1: a cell array of size (total tracking time x 1),
%       each cell contains an object state mean vector of size
%       (state dimension x 1)
%       state_sequence2: a cell array of size (total tracking time x 1),
%       each cell contains an object state mean vector of size
%       (state dimension x 1)
%OUTPUT:root_mean_square_error: root mean square error --- scalar
    % format input as matrix:
    xs1 = cell2mat(cellfun(@(s){s'}, state_sequence1));
    xs2 = cell2mat(cellfun(@(s){s'}, state_sequence2));
    % calculate RMSE through matrix F-norm:
    dx = xs1 - xs2;
    root_mean_square_error = norm(dx, 'fro')/sqrt(numel(dx));
end
