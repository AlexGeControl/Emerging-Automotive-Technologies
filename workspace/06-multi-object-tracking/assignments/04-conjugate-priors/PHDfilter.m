classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            
            % get num. of predicted hypotheses:
            [H_s, ~] = size(obj.paras.w);
            [~, H_b] = size(birthmodel);
            H = H_s + H_b;
            
            % update survival PHD:
            log_P_survival = log(P_S);
            for h = 1:H_s
                % update expected num. of detections:
                obj.paras.w(h, 1) = log_P_survival + obj.paras.w(h, 1);
                
                % perform Kalman update on prior:
                obj.paras.states(h, 1) = obj.density.predict(obj.paras.states(h, 1), motionmodel);
            end
            
            % set birth PHD:
            obj.paras.w((H_s + 1):(H_s + H_b), 1) = [birthmodel.w]';
            obj.paras.states((H_s + 1):(H_s + H_b), 1) = rmfield(birthmodel,'w')';
            
            % done:
            obj.paras.w = obj.paras.w(1:H, 1);
            obj.paras.states = obj.paras.states(1:H, 1);
        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            
            % get num. of active hypotheses:
            [H, ~] = size(obj.paras.w);
            [~, M] = size(z);
            
            % init:
            paras_updated.w = repmat(obj.paras.w(1, 1), H, 1);
            paras_updated.states = repmat(obj.paras.states(1, 1), H, 1);
            
            % set miss detection PHDs:
            log_P_miss = log(1 - P_D);
            for h = 1:H
                % update expected num. of detections:
                paras_updated.w(h, 1) = log_P_miss + obj.paras.w(h, 1);
                
                % update pdf:
                paras_updated.states(h, 1) = obj.paras.states(h, 1);
            end
            
            % cache intermediate results for object detection PHD detection:
            J_in_gate = false(H, M);
            Kalman = cell(H, 1);
            for h = 1:H
                % get predicted state:
                state = obj.paras.states(h, 1);
                
                % do gating:
                [~, j_in_gate] = obj.density.ellipsoidalGating(state, z, measmodel, gating.size);
                % save measurement association:
                J_in_gate(h, :) = j_in_gate;
                
                % get Kalman gain:
                Hx = measmodel.H(state.x);
                S = Hx*state.P*Hx' + measmodel.R;
                S = (S+S')/2;
                Kalman{h}.K = (state.P*Hx')/S;
                % get predicted measurement:
                Kalman{h}.z_pred = measmodel.h(state.x);
                % get posterior covariance:
                Kalman{h}.P = (eye(size(state.x,1)) - Kalman{h}.K*Hx)*state.P;
            end
            
            % set object detection PHDs:
            log_P_meas = log(P_D);
            h_updated = H;
            for m = 1:M
                z_m = z(:, m);
                
                i_in_gate = find(J_in_gate(:, m));
                n_in_gate = length(i_in_gate);
                
                if n_in_gate > 0
                    % perform Kalman update for qualified measurement:
                    for i = 1:n_in_gate
                        % get index of associated object:
                        h = i_in_gate(i, 1);
                        
                        % fetch predicted object state:
                        log_w = obj.paras.w(h, 1);
                        state = obj.paras.states(h, 1);
                        
                        % generate index of updated hypothesis:
                        h_updated = h_updated + 1;
                        
                        % do Kalman update:
                        K = Kalman{h}.K;
                        z_pred = Kalman{h}.z_pred;
                        P = Kalman{h}.P;
                        % update expected num. of observations:
                        paras_updated.w(h_updated, 1) = log_P_meas + obj.density.predictedLikelihood(state, z_m, measmodel) + log_w;
                        % update object state:
                        paras_updated.states(h_updated, 1).x = state.x + K*(z_m - z_pred);
                        paras_updated.states(h_updated, 1).P = P;                        
                    end
                    
                    % normalize weights:
                    w = exp(paras_updated.w((h_updated - n_in_gate + 1):(h_updated), 1));
                    paras_updated.w((h_updated - n_in_gate + 1):(h_updated), 1) = log(w / (intensity_c + sum(w)));
                end
            end
            
            % done:
            obj.paras = paras_updated;
        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            
            % get num. of objects:
            n = round(sum(exp(obj.paras.w)));
            n = min(n, length(obj.paras.w));
            
            % get top n most likely PHD means:
            [~, I] = maxk(obj.paras.w, n);
            
            % use the mean of the most likely object estimations as output:
            estimates = repmat(obj.paras.states(1, 1).x, 1, n);
            for i = 1:n
                estimates(:, i) = obj.paras.states(I(i), 1).x;
            end
        end
        
    end
    
end
