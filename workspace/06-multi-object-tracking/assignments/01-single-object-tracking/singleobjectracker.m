classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1
            
            % parse measurements:
            [N, ~] = size(Z);
            % init output:
            estimates = cell(N, 1);
            
            % process measurements:
            for i = 1:N                
                % gating -- reduce data association candidates:
                [z_in_gate, ~] = obj.density.ellipsoidalGating(state, Z{i}, measmodel, obj.gating.size);
                
                % first, handle potential miss detection:
                miss_meas_likelihood = 1 - sensormodel.P_D;

                % then, handle qualified object detections: 
                if size(z_in_gate, 2) > 0
                    % get max. measurement likelihood:
                    meas_likelihood = exp(obj.density.predictedLikelihood(state, z_in_gate, measmodel));
                    meas_likelihood = sensormodel.P_D / sensormodel.intensity_c * meas_likelihood;
                    [max_meas_likelihood, I] = max(meas_likelihood);
                    
                    % if max. measurement likelihood is larger than miss
                    % detection likelihood, update estimate:
                    if max_meas_likelihood > miss_meas_likelihood
                        state = obj.density.update(state, z_in_gate(:, I), measmodel);
                    end
                end
                
                % save posterior estimate:
                estimates{i} = state.x;
                
                % predict, again:
                state = obj.density.predict(state, motionmodel);
            end  
        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1  
            
            % parse measurements:
            [N, ~] = size(Z);
            % init output:
            estimates = cell(N, 1);
            
            % process measurements:
            for i = 1:N                
                % gating -- reduce data association candidates:
                [z_in_gate, ~] = obj.density.ellipsoidalGating(state, Z{i}, measmodel, obj.gating.size);
                
                % init container for candidate pdf and corresponding
                % weights:
                [~, m] = size(z_in_gate);
                w = zeros(m + 1, 1);
                states = repmat(state, m + 1, 1);
               
                % get posterior for miss detection and corresponding weight:
                miss_meas_likelihood = 1 - sensormodel.P_D;
                w(1) = miss_meas_likelihood;
                states(1) = state;
               
                % get posteriors for candidate detections and corresponding weights: 
                if m > 0
                    % get measurement likelihoods:
                    meas_likelihoods = exp(obj.density.predictedLikelihood(state, z_in_gate, measmodel));
                    meas_likelihoods = sensormodel.P_D / sensormodel.intensity_c * meas_likelihoods;
                    
                    for j = 1:m
                        w(1 + j) = meas_likelihoods(j);
                        states(1 + j) = obj.density.update(state, z_in_gate(:, j), measmodel);
                    end
                end
                
                % normalize weights:
                [log_w, ~] = normalizeLogWeights(log(w));
                % prune candidates with small weight:
                I = (log_w >= obj.reduction.w_min);
                log_w = log_w(I);
                states = states(I);
                % normalize again:
                [log_w, ~] = normalizeLogWeights(log_w);
                
                % get merged posterior:
                state = obj.density.momentMatching(log_w, states);
                
                % save posterior estimate:
                estimates{i} = state.x;
                
                % predict, again:
                state = obj.density.predict(state, motionmodel);
            end
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1
            
            % parse measurements:
            [N, ~] = size(Z);
            
            % init input:
            w = ones(1, 1);
            states = repmat(state, 1, 1);
            
            % init output:
            estimates = cell(N, 1);
            
            % process measurements:
            for i = 1:N
                % get num. of active hypotheses:
                [M, ~] = size(states);
                
                % init container for candidate pdfs and corresponding
                % weights
                num_candidates = 0;
                w_check = ones(1, 1);
                states_check = repmat(state, 1, 1);
                
                % for each active hypothesis:
                for j = 1:M
                    % get current pdf and corresponding weight:
                    w_hat = w(j);
                    state_hat = states(j);
                    
                    % gating -- reduce data association candidates:
                    [z_in_gate, ~] = obj.density.ellipsoidalGating(state_hat, Z{i}, measmodel, obj.gating.size);
                    
                    % get posterior for miss detection and corresponding weight:
                    miss_meas_likelihood = w_hat*(1 - sensormodel.P_D);
                    
                    num_candidates = num_candidates + 1;
                    w_check(num_candidates, 1) = miss_meas_likelihood;
                    states_check(num_candidates, 1) = state_hat;
                    
                    % get posteriors for candidate detections and corresponding weights:
                    [~, m] = size(z_in_gate);
                    if m > 0
                        % get measurement likelihoods:
                        meas_likelihoods = exp(obj.density.predictedLikelihood(state_hat, z_in_gate, measmodel));
                        meas_likelihoods = w_hat * sensormodel.P_D / sensormodel.intensity_c * meas_likelihoods;
                    
                        for k = 1:m
                            num_candidates = num_candidates + 1;
                            w_check(num_candidates, 1) = meas_likelihoods(k);
                            states_check(num_candidates, 1) = obj.density.update(state_hat, z_in_gate(:, k), measmodel);
                        end
                    end
                end
                
                % normalize weights:
                [log_w_check, ~] = normalizeLogWeights(log(w_check));
                % prune candidates with small weight:
                I = (log_w_check >= obj.reduction.w_min);
                log_w_check = log_w_check(I, 1);
                states_check = states_check(I, 1);
                % normalize again:
                [log_w_check, ~] = normalizeLogWeights(log_w_check);
                
                % merge hypotheses:
                [log_w_check, states_check] = hypothesisReduction.merge(log_w_check, states_check, obj.reduction.merging_threshold, obj.density);
                
                % cap hypotheses:
                [log_w_check, states_check] = hypothesisReduction.cap(log_w_check, states_check, obj.reduction.M);
                % normalize again:
                [log_w_check, ~] = normalizeLogWeights(log_w_check);
                
                % save posteriors:
                w = exp(log_w_check);
                states = states_check;
                
                % save posterior estimate:
                [~, i_most_likely] = max(w);
                estimates{i} = states(i_most_likely).x;
                
                % predict, again:
                for j = 1:size(states, 1)
                    states(j) = obj.density.predict(states(j, 1), motionmodel);
                end
            end
        end
        
    end
end

