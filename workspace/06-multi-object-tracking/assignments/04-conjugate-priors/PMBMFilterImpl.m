classdef PMBMFilterImpl
    %PMBMFILTER is a class containing necessary functions to implement the
    %PMBM filter
    %Model structures need to be called:
    %    sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    properties
        density %density class handle
        paras   %%parameters specify a PMBM
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PMBMfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture)
            %       of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.PPP.w: weights of mixture components in PPP
            %       intensity --- vector of size (number of mixture
            %       components x 1) in logarithmic scale
            %       obj.paras.PPP.states: parameters of mixture components
            %       in PPP intensity struct array of size (number of
            %       mixture components x 1)
            %       obj.paras.MBM.w: weights of MBs --- vector of size
            %       (number of MBs (global hypotheses) x 1) in logarithmic 
            %       scale
            %       obj.paras.MBM.ht: hypothesis table --- matrix of size
            %       (number of global hypotheses x number of hypothesis
            %       trees). Entry (h,i) indicates that the (h,i)th local
            %       hypothesis in the ith hypothesis tree is included in
            %       the hth global hypothesis. If entry (h,i) is zero, then
            %       no local hypothesis from the ith hypothesis tree is
            %       included in the hth global hypothesis.
            %       obj.paras.MBM.tt: local hypotheses --- cell of size
            %       (number of hypothesis trees x 1). The ith cell contains
            %       local hypotheses in struct form of size (number of
            %       local hypotheses in the ith hypothesis tree x 1). Each
            %       struct has two fields: r: probability of existence;
            %       state: parameters specifying the object density
            
            obj.density = density_class_handle;
            obj.paras.PPP.w = [birthmodel.w]';
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = {};
        end
        
        function predicted = Bern_predict(obj,Bern,motionmodel,P_S)
            %BERN_PREDICT performs prediction step for a Bernoulli component
            %INPUT: Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar;
            %                          state: a struct contains parameters
            %                          describing the object pdf
            %       P_S: object survival probability
            % prob. of existence as:
            predicted.r = Bern.r * P_S;
            
            % object state density:
            predicted.state = obj.density.predict(Bern.state, motionmodel);
        end
        
        function [updated, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D)
            %BERN_UNDETECTED_UPDATE calculates the likelihood of missed
            %detection, and creates new local hypotheses due to missed
            %detection.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth local
            %       hypothesis in the ith hypothesis tree. 
            %       P_D: object detection probability --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %       with fields: r: probability of existence --- scalar;
            %                    state: a struct contains parameters
            %                    describing the object pdf
            %       lik_undetected: missed detection likelihood --- scalar
            %       in logorithmic scale
            % get prior Bernoulli:
            idx_hypo_tree = tt_entry(1, 1);
            idx_hypo = tt_entry(1, 2);
            Bern = obj.paras.MBM.tt{idx_hypo_tree}(idx_hypo, 1);
            updated = struct('r', Bern.r, 'state', Bern.state);
            
            % set log likelihood:
            lik_undetected = log((1.0 - Bern.r) + Bern.r*(1.0 - P_D));
            
            % create updated Bernoulli:
            updated.r = Bern.r*(1.0 - P_D) / ((1.0 - Bern.r) + Bern.r*(1.0 - P_D));
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            %BERN_DETECTED_UPDATE_LIK calculates the predicted likelihood
            %for a given local hypothesis. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth
            %       local hypothesis in the ith hypothesis tree.
            %       z: measurement array --- (measurement dimension x
            %       number of measurements)
            %       P_D: object detection probability --- scalar
            %OUTPUT:lik_detected: predicted likelihood --- (number of
            %measurements x 1) array in logarithmic scale 
            % get prior Bernoulli:
            idx_hypo_tree = tt_entry(1, 1);
            idx_hypo = tt_entry(1, 2);
            Bern = obj.paras.MBM.tt{idx_hypo_tree}(idx_hypo, 1);
            
            % get object detection likelihood:
            lik_detected = log(Bern.r) + log(P_D) + obj.density.predictedLikelihood(Bern.state,z,measmodel);
        end
        
        function updated = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            %BERN_DETECTED_UPDATE_STATE creates the new local hypothesis
            %due to measurement update. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %                 local hypotheses. (i,j) indicates the jth
            %                 local hypothesis in the ith hypothesis tree.
            %       z: measurement vector --- (measurement dimension x 1)
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar; 
            %                          state: a struct contains parameters
            %                          describing the object pdf 
            % get prior Bernoulli:
            idx_hypo_tree = tt_entry(1, 1);
            idx_hypo = tt_entry(1, 2);
            Bern = obj.paras.MBM.tt{idx_hypo_tree}(idx_hypo, 1);
            updated = struct('r', Bern.r, 'state', Bern.state);
            
            % set Bernoulli for object detection:
            updated.r = 1.0;
            updated.state = obj.density.update(Bern.state, z, measmodel);
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            %PPP_PREDICT performs predicion step for PPP components
            %hypothesising undetected objects.
            %INPUT: P_S: object survival probability --- scalar          
            % predict undetected objects:
            log_P_S = log(P_S);
            [N_u, ~] = size(obj.paras.PPP.w);
            for i = 1:N_u
                obj.paras.PPP.w(i, 1) = log_P_S + obj.paras.PPP.w(i, 1);
                obj.paras.PPP.states(i, 1) = obj.density.predict(obj.paras.PPP.states(i, 1), motionmodel);
            end
            
            % add new born objects:
            w_b = [birthmodel.w]';
            states_b = rmfield(birthmodel,'w')';
            
            % done:
            [N_b, ~] = size(w_b);
            obj.paras.PPP.w((N_u+1):(N_u + N_b), 1) = w_b;
            obj.paras.PPP.states((N_u+1):(N_u + N_b), 1) = states_b;
        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,indices,z,measmodel,P_D,clutter_intensity)
            %PPP_DETECTED_UPDATE creates a new local hypothesis by
            %updating the PPP with measurement and calculates the
            %corresponding likelihood.
            %INPUT: z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %       clutter_intensity: Poisson clutter intensity --- scalar
            %       indices: boolean vector, if measurement z is inside the
            %       gate of mixture component i, then indices(i) = true
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %             scalar;
            %             state: a struct contains parameters describing
            %             the object pdf
            %       lik_new: predicted likelihood of PPP --- scalar in
            %       logarithmic scale 
            % init mixture components:
            w = obj.paras.PPP.w(indices, 1);
            states = obj.paras.PPP.states(indices, 1);
                        
            % do update:
            N_d = length(w);
            log_P_D = log(P_D);
            for i = 1:N_d
                % get measurement likelihood:
                w(i) = log_P_D + w(i) + obj.density.predictedLikelihood(states(i), z, measmodel);
                % do Kalman update on object state:
                states(i) = obj.density.update(states(i), z, measmodel);
            end
            
            % get birth likelihood:
            % lik_birth = sum(exp(w));
            
            % set new object likelihood:
            % lik_new = log(clutter_intensity + lik_birth);
            
            % create new object:
            % Bern.r = lik_birth / (clutter_intensity + lik_birth);
            
            % normalize weights:
            [w, lik_birth] = normalizeLogWeights(w);
            [~, lik_new] = normalizeLogWeights([lik_birth, log(clutter_intensity)]);
            Bern.r = exp(lik_birth - lik_new);
            Bern.state = obj.density.momentMatching(w, states);
        end
        
        function obj = PPP_undetected_update(obj,P_D)
            %PPP_UNDETECTED_UPDATE performs PPP update for missed detection.
            %INPUT: P_D: object detection probability --- scalar
            % if undetected objects stay undetected:
            [N_u, ~] = size(obj.paras.PPP.w);
            log_P_UD = log(1.0 - P_D);
            for i = 1:N_u
                obj.paras.PPP.w(i, 1) = log_P_UD + obj.paras.PPP.w(i, 1);
            end
        end
        
        function obj = PPP_reduction(obj,prune_threshold,merging_threshold)
            %PPP_REDUCTION truncates mixture components in the PPP
            %intensity by pruning and merging
            %INPUT: prune_threshold: pruning threshold --- scalar in
            %       logarithmic scale
            %       merging_threshold: merging threshold --- scalar
            [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.prune(obj.paras.PPP.w, obj.paras.PPP.states, prune_threshold);
            if ~isempty(obj.paras.PPP.w)
                [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.merge(obj.paras.PPP.w, obj.paras.PPP.states, merging_threshold, obj.density);
            end
        end
        
        function obj = Bern_recycle(obj,prune_threshold,recycle_threshold)
            %BERN_RECYCLE recycles Bernoulli components with small
            %probability of existence, adds them to the PPP component, and
            %re-index the hypothesis table. If a hypothesis tree contains no
            %local hypothesis after pruning, this tree is removed. After
            %recycling, merge similar Gaussian components in the PPP
            %intensity
            %INPUT: prune_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold are pruned ---
            %       scalar
            %       recycle_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold needed to be
            %       recycled --- scalar
            
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold & x.r>=prune_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Here, we should also consider the weights of different MBs
                    idx_t = find(idx);
                    n_h = length(idx_t);
                    w_h = zeros(n_h,1);
                    for j = 1:n_h
                        idx_h = obj.paras.MBM.ht(:,i) == idx_t(j);
                        [~,w_h(j)] = normalizeLogWeights(obj.paras.MBM.w(idx_h));
                    end
                    %Recycle
                    temp = obj.paras.MBM.tt{i}(idx);
                    obj.paras.PPP.w = [obj.paras.PPP.w;log([temp.r]')+w_h];
                    obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                end
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Remove Bernoullis
                    obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                    %Update hypothesis table, if a Bernoulli component is
                    %pruned, set its corresponding entry to zero
                    idx = find(idx);
                    for j = 1:length(idx)
                        temp = obj.paras.MBM.ht(:,i);
                        temp(temp==idx(j)) = 0;
                        obj.paras.MBM.ht(:,i) = temp;
                    end
                end
            end
            
            %Remove tracks that contains no valid local hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                %Ensure the algorithm still works when all Bernoullis are
                %recycled
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i) > 0;
                [~,~,obj.paras.MBM.ht(idx,i)] = unique(obj.paras.MBM.ht(idx,i),'rows','stable');
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows','stable');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
            
        end
        
        function obj = PMBM_predict(obj,P_S,motionmodel,birthmodel)
            %PMBM_PREDICT performs PMBM prediction step.
            % predict on each detected object:
            N_d = length(obj.paras.MBM.tt);
            if N_d > 0
                for i = 1:N_d
                    H_i = length(obj.paras.MBM.tt{i});
                    for j = 1:H_i
                        obj.paras.MBM.tt{i}(j) = Bern_predict(obj,obj.paras.MBM.tt{i}(j),motionmodel,P_S);
                    end
                end
            end
            
            % predict on each undetected object:
            obj = PPP_predict(obj,motionmodel,birthmodel,P_S);
        end
        
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,w_min,M)
            %PMBM_UPDATE performs PMBM update step.
            %INPUT: z: measurements --- array of size (measurement
            %       dimension x number of measurements)
            %       gating: a struct with two fields that specifies gating
            %       parameters: P_G: gating size in decimal --- scalar;
            %                   size: gating size --- scalar.
            %       wmin: hypothesis weight pruning threshold --- scalar in
            %       logarithmic scale
            %       M: maximum global hypotheses kept
            
%             %
%             % get num. of measurements:
%             %
%             [~, N_z] = size(z);
%             
%             %
%             % perform ellipsoidal gating for possible undetected objects, 
%             % represented by PPP:
%             %
%             N_ppp = length(obj.paras.PPP.w);
%             I_ppp = false(N_z, N_ppp);
%             I_z_ppp = false(N_z, 1);
%             for i = 1:N_ppp
%                 % get current mixture component:
%                 state = obj.paras.PPP.states(i, 1);
%                 
%                 % get indices of measurements inside current gate:
%                 [~, I_ppp(:, i)] = obj.density.ellipsoidalGating(state, z, measmodel, gating.size);
%                 
%                 % track all measurements inside the gate of possible
%                 % undetected objects:
%                 I_z_ppp = I_z_ppp | I_ppp(:, i);
%             end
%             
%             %
%             % perform ellipsoidal gating for possible detected objects,
%             % represented by MBM:
%             %
%             N_ber = length(obj.paras.MBM.tt);
%             I_ber = cell(N_ber, 1);
%             I_z_ber = false(N_z, 1);
%             for i = 1:N_ber
%                 % get num. of local hypotheses:
%                 H_i = length(obj.paras.MBM.tt{i, 1});
%                 
%                 I_ber{i, 1} = false(N_z, H_i);
%                 
%                 for h_i = 1:H_i
%                     % get current object state:
%                     state = obj.paras.MBM.tt{i, 1}(h_i, 1).state;
%                     
%                     % get indices of measurements inside current gate:
%                     [~, I_ber{i, 1}(:, h_i)] = obj.density.ellipsoidalGating(state, z, measmodel, gating.size);
%                 
%                     % track all measurements inside the gate of possible
%                     % undetected objects:
%                     I_z_ber = I_z_ber | I_ber{i, 1}(:, h_i);
%                 end
%             end
%             
%             %
%             % process measurements from possible detected objects 
%             % as optimal assignment problem
%             %
%             % use only measurements from possible detected objects:
%             z_d = z(:, I_z_ber);
%             [~, N_z_d] = size(z_d);
%             I_ber = cellfun(@(x) x(I_z_ber,:), I_ber, 'UniformOutput',false);
%             % candidate posterior of detected objects:
%             tt = cell(N_ber, 1);
%             % corresponding likelihood:
%             L_detected = cell(N_ber, 1);
%             for i = 1:N_ber
%                 % get num. of local hypotheses:
%                 H_i = length(obj.paras.MBM.tt{i});
%                 
%                 tt{i} = cell(H_i*(1 + N_z_d), 1);
%                 L_detected{i} = -Inf * ones(H_i, 1 + N_z_d);
%                 
%                 for h_i = 1:H_i
%                     tt_entry = [i, h_i];
%                     candidate_index_offset = (h_i - 1)*(1 + N_z_d);
%                     I_ber_h_i = I_ber{i, 1}(:, h_i);
%                     
%                     % possible miss detection:
%                     [tt{i}{candidate_index_offset + 1, 1}, ...
%                      L_detected{i}(h_i, 1)] ...
%                      = Bern_undetected_update(obj,tt_entry,sensormodel.P_D);
%                  
%                     % possible object detection:
%                     L_detected{i}(h_i, [false, I_ber_h_i']) = ...
%                         Bern_detected_update_lik(obj,tt_entry,z_d(:, I_ber_h_i),measmodel,sensormodel.P_D);
%                     for j = 1:N_z_d
%                         if I_ber_h_i(j, 1)
%                             tt{i}{candidate_index_offset + 1 + j, 1} = ...
%                                 Bern_detected_update_state(obj,tt_entry,z_d(:, j),measmodel);
%                         end
%                     end
%                 end
%             end
%             % if measurements could also come from undetected objects:
%             I_ppp_d = I_ppp(I_z_ber,:);
%             I_z_d = I_z_ppp(I_z_ber,:);
%             % corresponding likelihood:
%             L_birth = -Inf * ones(N_z_d, N_z_d);
%             for j = 1:N_z_d
%                 tt{N_ber + j} = cell(1, 1);
%                 if any(I_ppp_d(j, :))
%                     [tt{N_ber + j}{1}, L_birth(j, j)] = ...
%                         PPP_detected_update(obj, I_ppp_d(j, :), z_d(:, j), measmodel, sensormodel.P_D, sensormodel.intensity_c);
%                 else
%                     L_birth(j, j) = log(sensormodel.intensity_c);
%                 end
%             end
%                         
%             %
%             % generate candidate posterior hypotheses:
%             %
%             H_prior = length(obj.paras.MBM.w);
%             H_posterior = 0;
%             w = [];
%             ht = zeros(1, N_ber + N_z_d);
%             if H_prior == 0 
%                 % if there is no pre-existing global hypothesis:
%                 H_posterior = 1;
%                 w = zeros(1, 1);
%                 ht = zeros(1, N_z_d);
%                 ht(1, I_z_d) = 1;
%             else
%                 for h_prior = 1:H_prior
%                     % 
%                     % for each prior global hypothesis
%                     % generate candidate posterior hypotheses through
%                     % optimal assignment
%                     %
%                     w_prior = obj.paras.MBM.w(h_prior, 1);
%                     L_h = Inf * ones(N_z_d, N_ber + N_z_d);
%                     
%                     for i = 1:N_ber
%                         h_i = obj.paras.MBM.ht(h_prior, i);
%                         if h_i > 0
%                             L_h(:, i) = -(L_detected{i}(h_i, 2:end) - L_detected{i}(h_i, 1))';
%                             w_prior = w_prior + L_detected{i}(h_i, 1);
%                         end
%                     end
%                     L_h(:, (N_ber + 1):(N_ber + N_z_d)) = -L_birth;
%                     
%                     if isempty(L_h)
%                         w_likelihood = 0;
%                         I_assignment_h = 0;
%                     else
%                         M_h = ceil(exp(obj.paras.MBM.w(h_prior)+log(M)));
%                         % solve with Murty's algorithm:
%                         [I_assignment_h,~,w_likelihood] = kBest2DAssign(L_h,M_h);
%                         % solve with Gibbs sampling:
%                         % [I_assignment_h,w_likelihood] = assign2DByGibbs(L_h,100,M_h);
%                     end
%                     
%                     % save candidate posterior likelihood:
%                     w = [w;w_prior - w_likelihood];
%                     
%                     % generate candidate posterior hypothesis:
%                     M_h = length(w_likelihood);
%                     ht_h = zeros(M_h,N_ber + N_z_d);
%                     for h_posterior = 1:M_h
%                         % process possible association with detected
%                         % objects:
%                         for i = 1:N_ber
%                             h_i = obj.paras.MBM.ht(h_prior, i);
%                             if h_i > 0
%                                 candidate_index_offset = (h_i - 1)*(1 + N_z_d);
%                                 j = find(I_assignment_h(:,h_posterior)==i, 1);
%                                 
%                                 if isempty(j)
%                                     % no object detection:
%                                     ht_h(h_posterior,i) = candidate_index_offset+1;
%                                 else
%                                     % with object detection:
%                                     ht_h(h_posterior,i) = candidate_index_offset+1+j;
%                                 end
%                             end
%                         end
%                         % process possible association with new birth
%                         % objects:
%                         for i = (N_ber+1):(N_ber + N_z_d)
%                             j = find(I_assignment_h(:,h_posterior)==i, 1);
%                             if ~isempty(j) && I_z_d(j)
%                                 % detection from new birth object:
%                                 ht_h(h_posterior,i) = 1;
%                             end
%                         end
%                     end
%                     
%                     % update candidate posteriors:
%                     H_posterior = H_posterior + M_h;
%                     ht = [ht;ht_h];
%                 end
%                 
%                 % normalize global hypothesis weights:
%                 w = normalizeLogWeights(w);
%             end
%             
%             %
%             % for all measurements that inside the gate of undetected but not inside the gate of detected
%             % append all of them into existing global hypotheses
%             % this will not change candidate posterior weights
%             %
%             I_z_u = (I_z_ppp & (~I_z_ber));
%             z_u = z(:,I_z_u);
%             N_z_u = size(z_u,2);
%             I_ppp_u = I_ppp(I_z_u,:);
%             for j = 1:N_z_u
%                 tt{N_ber+N_z_d+j,1} = cell(1, 1);
%                 [tt{N_ber+N_z_d+j,1}{1}, ~] = ...
%                     PPP_detected_update(obj,I_ppp_u(j,:),z_u(:,j),measmodel,sensormodel.P_D,sensormodel.intensity_c);
%             end
%             ht = [ht ones(H_posterior,N_z_u)];
%             disp(ht);
%             
%             % update undetected objects that are still undetected:
%             obj = PPP_undetected_update(obj,sensormodel.P_D);
%             
%             % prune: 
%             [w, I] = hypothesisReduction.prune(w,1:H_posterior,w_min);
%             H_posterior = length(w);
%             ht = ht(I,:);
%             w = normalizeLogWeights(w);
%             
%             % capping:
%             [w, I] = hypothesisReduction.cap(w,1:H_posterior,M);
%             ht = ht(I,:);
%             obj.paras.MBM.w = normalizeLogWeights(w);
%             
%             % remove unused local hypotheses:
%             if ~isempty(ht)
%                 I = sum(ht,1) >= 1;
%                 ht = ht(:,I);
%                 tt = tt(I);
%             end
%             
%             % re-index local hypotheses:
%             N_ber = length(tt);
%             obj.paras.MBM.tt = cell(N_ber,1);
%             for i = 1:N_ber
%                 ht_i = ht(:,i);
%                 tt_i = tt{i}(unique(ht_i(ht_i~=0), 'stable'));
%                 obj.paras.MBM.tt{i} = [tt_i{:}]';
%             end
%             
%             % re-index global hypotheses:
%             for i = 1:N_ber
%                 I = ht(:,i) > 0;
%                 [~,~,ht(I,i)] = unique(ht(I,i),'rows','stable');
%             end
%             
%             % done:
%             obj.paras.MBM.ht = ht;
            %
            % get num. of measurements:
            %
            m = size(z,2);                      %number of measurements received

            %
            % perform gating for each mixture component in the PPP intensity
            %
            nu = length(obj.paras.PPP.states);  % number of mixture components in PPP intensity
            gating_matrix_u = false(m,nu);
            used_meas_u = false(m,1);           % measurement indices inside the gate of undetected objects
            for i = 1:nu
                % identify possible measurements from current mixture component:
                [~,gating_matrix_u(:,i)] = ...
                    obj.density.ellipsoidalGating(obj.paras.PPP.states(i),z,measmodel,gating.size);
                used_meas_u = used_meas_u | gating_matrix_u(:,i);
            end
            
            %
            % perform ellipsoidal gating for each detected object candidate in the MBM density
            %
            n_tt = length(obj.paras.MBM.tt);    % number of pre-existing hypothesis trees
            likTable = cell(n_tt,1);            % initialise likelihood table, one for each hypothesis tree
            gating_matrix_d = cell(n_tt,1);
            used_meas_d = false(m,1);           % measurement indices inside the gate of detected objects
            for i = 1:n_tt
                % num. of local hypotheses for object i:
                num_hypo = length(obj.paras.MBM.tt{i});
                % construct gating matrix:
                gating_matrix_d{i} = false(m,num_hypo);
                for j = 1:num_hypo
                    % identify possible measurements from current detected object candidate:
                    [~,gating_matrix_d{i}(:,j)] = obj.density.ellipsoidalGating(obj.paras.MBM.tt{i}(j).state,z,measmodel,gating.size);
                    used_meas_d = used_meas_d | gating_matrix_d{i}(:,j);
                end
            end
            
            %
            % identify qualified measurements:
            %
            % measurement inside the gate -- either from detected or from undetected:
            used_meas = used_meas_d | used_meas_u;
            % measurement inside the gate of undetected objects but not inside the gate of detected objects:
            used_meas_u_not_d = used_meas > used_meas_d;
            
            % Update detected objects
            % obtain measurements that are inside the gate of detected objects
            z_d = z(:,used_meas_d);
            m = size(z_d,2);
            gating_matrix_d = cellfun(@(x) x(used_meas_d,:), gating_matrix_d, 'UniformOutput',false);
            n_tt_upd = n_tt + m;                % number of hypothesis trees
            hypoTable = cell(n_tt_upd,1);       % initialise hypothesis table, one for each hypothesis tree
            for i = 1:n_tt
                % number of local hypotheses in hypothesis tree i
                num_hypo = length(obj.paras.MBM.tt{i});
                % initialise likelihood table for hypothesis tree i
                likTable{i} = -inf(num_hypo,m+1);
                % initialise hypothesis table for hypothesis tree i
                hypoTable{i} = cell(num_hypo*(m+1),1);
                for j = 1:num_hypo
                    % Missed detection
                    [hypoTable{i}{(j-1)*(m+1)+1},likTable{i}(j,1)] = Bern_undetected_update(obj,[i,j],sensormodel.P_D);
                    % Update with measurement
                    likTable{i}(j,[false;gating_matrix_d{i}(:,j)]) = ...
                        Bern_detected_update_lik(obj,[i,j],z_d(:,gating_matrix_d{i}(:,j)),measmodel,sensormodel.P_D);
                    for jj = 1:m
                        if gating_matrix_d{i}(jj,j)
                            hypoTable{i}{(j-1)*(m+1)+jj+1} = Bern_detected_update_state(obj,[i,j],z_d(:,jj),measmodel);
                        end
                    end
                end
            end
            
            % Update undetected objects
            lik_new = -inf(m,1);
            gating_matrix_ud = gating_matrix_u(used_meas_d,:);
            % Create new hypothesis trees, one for each measurement inside
            %the gate 
            for i = 1:m
                if any(gating_matrix_ud(i,:))
                    [hypoTable{n_tt+i,1}{1}, lik_new(i)] = ...
                        PPP_detected_update(obj,gating_matrix_ud(i,:),z_d(:,i),measmodel,sensormodel.P_D,sensormodel.intensity_c);
                else
                    %For measurements not inside the gate of undetected
                    %objects, set likelihood to clutter intensity
                    lik_new(i) = log(sensormodel.intensity_c);
                end
            end
            used_meas_ud = sum(gating_matrix_ud, 2) >= 1; 
            
            %Cost matrix for first detection of undetected objects
            L2 = inf(m);
            L2(logical(eye(m))) = -lik_new;
            
            %Update global hypothesis
            w_upd = [];             
            ht_upd = zeros(0,n_tt_upd);
            H_upd = 0;
            
            %Number of global hypothesis
            H = length(obj.paras.MBM.w);
            if H == 0 %if there is no pre-existing hypothesis tree
                w_upd = 0;
                H_upd = 1;
                ht_upd = zeros(1,m);
                ht_upd(used_meas_ud) = 1;
            else
                for h = 1:H
                    %Cost matrix for detected objects
                    L1 = inf(m,n_tt);
                    lik_temp = 0;
                    for i = 1:n_tt
                        hypo_idx = obj.paras.MBM.ht(h,i);
                        if hypo_idx~=0
                            L1(:,i) = -(likTable{i}(hypo_idx,2:end) - likTable{i}(hypo_idx,1));
                            %we need add the removed weights back to
                            %calculate the updated global hypothesis weight
                            lik_temp = lik_temp + likTable{i}(hypo_idx,1);
                        end
                    end
                    %Cost matrix of size m-by-(n+m)
                    L = [L1 L2];
                    
                    if isempty(L)
                        %Consider the case that no measurements are inside
                        %the gate, thus missed detection
                        gainBest = 0;
                        col4rowBest = 0;
                    else
                        %Obtain M best assignments using Murty's algorithm
                        [col4rowBest,~,gainBest] = kBest2DAssign(L,ceil(exp(obj.paras.MBM.w(h)+log(M))));
                        %Obtain M best assignments using Gibbs sampling
%                       [col4rowBest,gainBest] = assign2DByGibbs(L,100,ceil(exp(obj.paras.MBM.w(h)+log(M))));
                    end
                    
                    %Restore weights
                    w_upd = [w_upd;-gainBest+lik_temp+obj.paras.MBM.w(h)];
                    
                    %Update global hypothesis look-up table
                    Mh = length(gainBest);
                    ht_upd_h = zeros(Mh,n_tt_upd);
                    for j = 1:Mh
                        ht_upd_h(j,1:n_tt_upd) = 0;
                        for i = 1:n_tt
                            if obj.paras.MBM.ht(h,i) ~= 0
                                idx = find(col4rowBest(:,j)==i, 1);
                                if isempty(idx)
                                    %missed detection
                                    ht_upd_h(j,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+1;
                                else
                                    %measurement update
                                    ht_upd_h(j,i) = (obj.paras.MBM.ht(h,i)-1)*(m+1)+idx+1;
                                end
                            end
                        end
                        for i = n_tt+1:n_tt_upd
                            idx = find(col4rowBest(:,j)==i, 1);
                            if ~isempty(idx) && used_meas_ud(idx)
                                %measurement update for PPP
                                ht_upd_h(j,i) = 1;
                            end
                        end
                    end
                    H_upd = H_upd + Mh;
                    ht_upd = [ht_upd;ht_upd_h];
                end
                
                %Normalize global hypothesis weights
                 w_upd = normalizeLogWeights(w_upd);
                
            end
            
            %
            % for all measurements that inside the gate of undetected but not inside the gate of detected
            % append all of them into existing global hypotheses
            %
            z_u_not_d = z(:,used_meas_u_not_d);
            num_u_not_d = size(z_u_not_d,2);
            gating_matrix_u_not_d = gating_matrix_u(used_meas_u_not_d,:);
            for i = 1:num_u_not_d
                [hypoTable{n_tt_upd+i,1}{1}, ~] = ...
                    PPP_detected_update(obj,gating_matrix_u_not_d(i,:),z_u_not_d(:,i),measmodel,sensormodel.P_D,sensormodel.intensity_c);
            end
            ht_upd = [ht_upd ones(H_upd,num_u_not_d)];
            
            %Update undetected objects with missed detection
            obj = PPP_undetected_update(obj,sensormodel.P_D);
            
            %Prune hypotheses with weight smaller than the specified
            %threshold 
            [w_upd, hypo_idx] = hypothesisReduction.prune(w_upd,1:H_upd,w_min);
            ht_upd = ht_upd(hypo_idx,:);
            w_upd = normalizeLogWeights(w_upd);
            
            %Keep at most M hypotheses with the highest weights
            [w_upd, hypo_idx] = hypothesisReduction.cap(w_upd,1:length(w_upd),M);
            ht_upd = ht_upd(hypo_idx,:);
            obj.paras.MBM.w = normalizeLogWeights(w_upd);
            
            %Remove empty hypothesis trees
            if ~isempty(ht_upd)
                idx = sum(ht_upd,1) >= 1;
                ht_upd = ht_upd(:,idx);
                hypoTable = hypoTable(idx);
                n_tt_upd = size(ht_upd,2);
            end
            
            %Prune local hypotheses that do not appear in maintained global
            %hypotheses 
            obj.paras.MBM.tt = cell(n_tt_upd,1);
            for i = 1:n_tt_upd
                temp = ht_upd(:,i);
                hypoTableTemp = hypoTable{i}(unique(temp(temp~=0), 'stable'));
                obj.paras.MBM.tt{i} = [hypoTableTemp{:}]';
            end
            
            %Re-index hypothesis table
            for i = 1:n_tt_upd
                idx = ht_upd(:,i) > 0;
                [~,~,ht_upd(idx,i)] = unique(ht_upd(idx,i),'rows','stable');
            end
            
            obj.paras.MBM.ht = ht_upd;
        end
        
        function estimates = PMBM_estimator(obj,threshold)
            %PMBM_ESTIMATOR performs object state estimation in the PMBM
            %filter
            %INPUT: threshold (if exist): object states are extracted from
            %       Bernoulli components with probability of existence no
            %       less than this threhold in Estimator 1. Given the
            %       probabilities of detection and survival, this threshold
            %       determines the number of consecutive misdetections
            %OUTPUT:estimates: estimated object states in matrix form of
            %       size (object state dimension) x (number of objects)
            %%%
            %First, select the multi-Bernoulli with the highest weight.
            %Second, report the mean of the Bernoulli components whose
            %existence probability is above a threshold. 
            % get max. likelihood hypothesis:            
            [~, I] = max(obj.paras.MBM.w);
            max_likelihood_ht = obj.paras.MBM.ht(I, :);
                        
            % get object states estimation:
            N_d = length(obj.paras.MBM.tt);            
            N = 0;
            estimates = [];
            for i = 1:N_d
                idx_local_hypo = max_likelihood_ht(i);
                if idx_local_hypo > 0
                    local_hypo = obj.paras.MBM.tt{i}(idx_local_hypo);
                    if local_hypo.r >= threshold
                        N = N + 1;
                        
                        if N == 1
                            estimates = repmat(local_hypo.state.x, 1, 1);
                        else
                            estimates(:, N) = local_hypo.state.x;
                        end
                    end
                end
            end
        end
    
    end
end
