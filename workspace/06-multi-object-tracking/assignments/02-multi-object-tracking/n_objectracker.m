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
            
        end
    end
end
