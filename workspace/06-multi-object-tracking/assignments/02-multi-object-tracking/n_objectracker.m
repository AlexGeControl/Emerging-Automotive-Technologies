classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
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
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
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
            %         used in TOMHT --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNFILTER tracks n object using global nearest neighbor
            %association 
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
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
            %       state dimension) x (number of objects)
            % get tracking horizon:
            [K, ~] = size(Z);
            
            % init output:
            estimates = cell(K, 1);
            
            % process measurements:
            for k = 1:K
                % get num. of objects:
                [~, N] = size(states);
                % get num. of measurements:
                Z_k = Z{k};
                [~, M] = size(Z_k);
                
                % initialize likelihood matrix L:
                L_meas = Inf * ones(N, M);
                L_miss = Inf * ones(N, N);
                J = false(1, M);
                
                % for each object:
                for n = 1:N
                    % get object state:
                    state = states(1, n);
                    
                    % gating -- reduce data association candidates:
                    [z_in_gate, j_in_gate] = obj.density.ellipsoidalGating(state, Z_k, measmodel, obj.gating.size);
                    j_in_gate = j_in_gate';
                    
                    % set negative log likelihood for miss detection:
                    L_miss(n, n) = -log(1 - sensormodel.P_D);
                    
                    % set negative log likelihood for potential object
                    % detections:
                    if size(z_in_gate, 2) > 0
                        L_meas(n, j_in_gate) = -log(sensormodel.P_D / sensormodel.intensity_c);
                        L_meas(n, j_in_gate) = L_meas(n, j_in_gate) - obj.density.predictedLikelihood(state, z_in_gate, measmodel)';
                        
                        J = J | j_in_gate;
                    end
                end

                % find optimal assignment:
                [J_assignment, ~, ~] = assign2D([L_meas(:, J), L_miss]);
                    
                % get posterior:
                J = find(J);
                [~, M] = size(J);
                for n = 1:N
                    % get object state:
                    state = states(1, n);
                    % get optimal association:
                    j_assignment = J_assignment(n, 1);
                    
                    % perform Kalman update for object detection:
                    if j_assignment <= M
                        j_assignment = J(1, j_assignment);
                        states(1, n) = obj.density.update(state, Z_k(:, j_assignment), measmodel);
                    end
                end
                
                % save posterior estimate
                X_k = repmat(states(1, 1).x, 1, N);
                for n = 1:N
                    X_k(:, n) = states(1, n).x;
                end
                estimates{k} = X_k;
                
                % predict:
                for n = 1:N
                    states(1, n) = obj.density.predict(states(1, n), motionmodel);
                end
            end
        end
        
        function estimates = JPDAfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %JPDAFILTER tracks n object using joint probabilistic data
            %association
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
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
            %       state dimension) x (number of objects)
            
            % get tracking horizon:
            [K, ~] = size(Z);
            
            % init output:
            estimates = cell(K, 1);
            
            % process measurements:
            for k = 1:K
                % get num. of objects:
                [~, N] = size(states);
                % get num. of measurements:
                Z_k = Z{k};
                [~, M] = size(Z_k);
                
                % initialize likelihood matrix L:
                L_meas = Inf * ones(N, M);
                L_miss = Inf * ones(N, N);
                J = false(1, M);
                
                % for each object:
                for n = 1:N
                    % get object state:
                    state = states(1, n);
                    
                    % gating -- reduce data association candidates:
                    [z_in_gate, j_in_gate] = obj.density.ellipsoidalGating(state, Z_k, measmodel, obj.gating.size);
                    j_in_gate = j_in_gate';
                    
                    % set negative log likelihood for miss detection:
                    L_miss(n, n) = -log(1 - sensormodel.P_D);
                    
                    % set negative log likelihood for potential object
                    % detections:
                    if size(z_in_gate, 2) > 0
                        L_meas(n, j_in_gate) = -log(sensormodel.P_D / sensormodel.intensity_c);
                        L_meas(n, j_in_gate) = L_meas(n, j_in_gate) - obj.density.predictedLikelihood(state, z_in_gate, measmodel)';
                        
                        J = J | j_in_gate;
                    end
                end

                % find optimal assignment:
                [J_assignment, ~, log_w] = kBest2DAssign([L_meas(:, J), L_miss], obj.reduction.M);
                log_w = -log_w;
                
                % normalize weights:
                [log_w, ~] = normalizeLogWeights(log_w);
                % prune candidates with small weight:
                I = (log_w >= obj.reduction.w_min);
                log_w = log_w(I);
                J_assignment = J_assignment(:, I);
                % normalize again:
                [log_w, ~] = normalizeLogWeights(log_w);
                
                % get posterior:
                J = find(J);
                [~, M] = size(J);
                [H, ~] = size(log_w);
                for n = 1:N
                    % get object state:
                    state = states(1, n);
                    
                    % init candidate posteriors and weights:
                    states_n = repmat(state, H, 1);
                    
                    % get candidate posteriors and weights:
                    for h = 1:H
                        j_assignment = J_assignment(n, h);
                    
                        % perform Kalman update for object detection:
                        if j_assignment <= M
                            j_assignment = J(1, j_assignment);
                                                    
                            states_n(h, 1) = obj.density.update(state, Z_k(:, j_assignment), measmodel);
                        end
                    end
                    
                    % merge all candidate posteriors through moment
                    % matching:
                    states(1, n) = obj.density.momentMatching(log_w, states_n);
                end
                
                % save posterior estimate
                X_k = repmat(states(1, 1).x, 1, N);
                for n = 1:N
                    X_k(:, n) = states(1, n).x;
                end
                estimates{k} = X_k;
                
                % predict:
                for n = 1:N
                    states(1, n) = obj.density.predict(states(1, n), motionmodel);
                end
            end 
        end
                
        function estimates = TOMHT(obj, states, Z, sensormodel, motionmodel, measmodel)
            %TOMHT tracks n object using track-oriented multi-hypothesis tracking
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
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
            %       state dimension) x (number of objects)
            
            % utility - TO-MHT hypothesis constructor
            function hypothesis = getHypothesis(states)
                % init:
                hypothesis = {};
            
                % init local hypotheses:
                [~, N] = size(states);
                hypothesis.objects = cell(N, 1);
                for n = 1:N
                    hypothesis.objects{n}.N = 0;
                    hypothesis.objects{n}.states = repmat(states(1, n), 1, 1);
                end
            
                % init factorization:
                hypothesis.log_w = zeros(1, 1);
                hypothesis.factorization = ones(1, N);
            end
            
            % get num. of objects:
            [~, N] = size(states);
            
            % get tracking horizon:
            [K, ~] = size(Z);
            
            %
            % init data structures for TO-MHT:
            %
            % prior and posterior:
            prior = getHypothesis(states);
            for n = 1:N
                prior.objects{n}.N = 1;
            end
            
            % init output:
            estimates = cell(K, 1);
            
            % process measurements:
            for k = 1:K
                % get current measurement:
                Z_k = Z{k};
                
                % get num. of measurements:
                [~, M] = size(Z_k);
                
                % for each object:
                for n = 1:N
                    % init local likelihood matrix L:
                    prior.objects{n}.L_meas = Inf * ones(prior.objects{n}.N, M);
                    prior.objects{n}.L_miss = Inf * ones(prior.objects{n}.N, N);
                    
                    % for each local hypothesis:
                    for i = 1:prior.objects{n}.N
                        % get prior:
                        state = prior.objects{n}.states(i, 1);
                        
                        % gating:
                        [z_in_gate, j_in_gate] = obj.density.ellipsoidalGating(state, Z_k, measmodel, obj.gating.size);
                        j_in_gate = j_in_gate';
                        
                        % set negative log likelihood for miss detection:
                        prior.objects{n}.L_miss(i, n) = -log(1 - sensormodel.P_D);
                    
                        % set negative log likelihood for object detections:
                        if size(z_in_gate, 2) > 0
                            prior.objects{n}.L_meas(i, j_in_gate) = -log(sensormodel.P_D / sensormodel.intensity_c);
                            prior.objects{n}.L_meas(i, j_in_gate) = prior.objects{n}.L_meas(i, j_in_gate) - obj.density.predictedLikelihood(state, z_in_gate, measmodel)';
                        end
                    end
                end
                
                % init candidate global hypothesis posterior:
                [H, ~] = size(prior.factorization);
                M_h = ceil(exp(prior.log_w)*obj.reduction.M);
                M_H = cumsum(M_h);
                prior_index = zeros(M_H(H, 1), 1);
                log_w = -Inf * ones(M_H(H, 1), 1);
                J_assignment = zeros(N, M_H(H, 1));
                
                % for each global hypothesis prior:
                for h = 1:H
                    % init global likelihood matrix L:
                    L_meas = Inf * ones(N, M);
                    L_miss = Inf * ones(N, N);
                    
                    % find optimal assignment:
                    J_h = false(1, M);
                    for n = 1:N
                        i = prior.factorization(h, n);
                        
                        L_meas(n, :) = prior.objects{n}.L_meas(i, :);
                        L_miss(n, :) = prior.objects{n}.L_miss(i, :);
                        
                        J_h = J_h | (L_meas(n, :) ~= Inf);
                    end
                    [J_assignment_h, ~, log_w_h] = kBest2DAssign([L_meas(:, J_h), L_miss], M_h(h, 1));
                    log_w_h = -log_w_h;
                    
                    % map to actual measurement index:
                    J_h = find(J_h);
                    [~, H_h] = size(J_assignment_h);
                    for h_h = 1:H_h
                        for n = 1:N
                            if J_assignment_h(n, h_h) <= size(J_h, 2)
                                J_assignment_h(n, h_h) = J_h(1, J_assignment_h(n, h_h));
                            else
                                J_assignment_h(n, h_h) = M + n;
                            end
                        end
                    end
                    
                    % save for normalization:
                    if h == 1
                        candidate_hypotheses_index = 1:H_h;
                    else
                        candidate_hypotheses_index = (M_H(h-1, 1)+1):(M_H(h-1, 1)+H_h);
                    end
                    
                    prior_index(candidate_hypotheses_index, 1) = h;
                    log_w(candidate_hypotheses_index, 1) = log_w_h + prior.log_w(h, 1);
                    J_assignment(:, candidate_hypotheses_index) = J_assignment_h;
                end

                % prune candidates with small weight:
                I = (log_w ~= -Inf);
                prior_index = prior_index(I);
                log_w = log_w(I);
                J_assignment = J_assignment(:, I);
                
                % normalize weights:
                [log_w, ~] = normalizeLogWeights(log_w);
                
                % prune candidates with small weight:
                I = (log_w >= obj.reduction.w_min);
                prior_index = prior_index(I);
                log_w = log_w(I);
                J_assignment = J_assignment(:, I);

                % capping:
                [~, I] = maxk(log_w, obj.reduction.M);
                prior_index = prior_index(I);
                log_w = log_w(I);
                J_assignment = J_assignment(:, I);

                % normalize weights:
                [log_w, ~] = normalizeLogWeights(log_w);
                
                % init posterior:
                [H, ~] = size(log_w);
                
                posterior = getHypothesis(states);
                posterior.log_w = log_w;
                posterior.factorization = zeros(H, N);
                
                local_hypothesis_uuid = cell(N, 1);
                for n = 1:N
                    local_hypothesis_uuid{n} = zeros(1, 1);
                end
                
                % for each object:
                for n = 1:N
                    % for each global hypothesis posterior:
                    for h = 1:H
                        i = prior.factorization(prior_index(h, 1), n);
                        j = J_assignment(n, h);
                        
                        % update selected local hypothesis:
                        state_uuid = (i-1)*M + j;
                        if ismember(state_uuid, local_hypothesis_uuid{n})
                            local_hypothesis_index = find(local_hypothesis_uuid{n} == state_uuid);
                        else
                            posterior.objects{n}.N = posterior.objects{n}.N + 1;
                            
                            local_hypothesis_index = posterior.objects{n}.N;
                            
                            state = prior.objects{n}.states(i, 1);
                            if j <= M
                                posterior.objects{n}.states(local_hypothesis_index, 1) = obj.density.update(state, Z_k(:, j), measmodel);
                            else
                                posterior.objects{n}.states(local_hypothesis_index, 1) = state;
                            end
                            
                            local_hypothesis_uuid{n}(local_hypothesis_index, 1) = state_uuid;
                        end
                        
                        % update global hypotheses look-up table:                  
                        posterior.factorization(h, n) = local_hypothesis_index;
                    end
                end
                
                % save posterior estimate
                X_k = repmat(states(1, 1).x, 1, N);
                for n = 1:N
                    % get most likely object state:
                    i = posterior.factorization(1, n);
                    % set as object estimation:
                    X_k(:, n) = posterior.objects{n}.states(i, 1).x;
                end
                estimates{k} = X_k;
                
                % predict:
                for n = 1:N
                    % init:
                    prior.objects{n}.N = posterior.objects{n}.N;
                    prior.objects{n}.states = posterior.objects{n}.states;
                                        
                    % perform update for each local hypothesis:
                    for i = 1:prior.objects{n}.N
                        prior.objects{n}.states(i, 1) = obj.density.predict(prior.objects{n}.states(i, 1), motionmodel);
                    end
                end
                
                % update global hypotheses likelihoods:
                prior.log_w = posterior.log_w;
               
                % update global hypotheses look-up table:
                prior.factorization = posterior.factorization;
            end
        end
    end
end

