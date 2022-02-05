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
            
            %
            % get num. of measurements:
            %
            [~, N_z] = size(z);
            
            %
            % perform ellipsoidal gating for mixture components in PPP:
            %
            N_ppp = length(obj.paras.PPP.w);
            I_ppp = false(N_ppp, N_z);
            for i = 1:N_ppp
                % get current mixture component:
                state = obj.paras.PPP.states(i, 1);
                
                % get indices of measurements in gate:
                [~, I_ppp(i, :)] = obj.density.ellipsoidalGating(state, z, measmodel, gating.size);
            end
                        
            %
            % cache likelihood matrix, birth sub-matrix:
            %
            L_birth = Inf * ones(N_z, N_z);
            states_birth = cell(N_z, 1);
            
            for j = 1:N_z
                if any(I_ppp(:, j))
                    [state_birth, l_birth] = PPP_detected_update(obj, I_ppp(:, j), z(:, j), measmodel, sensormodel.P_D, sensormodel.intensity_c);
                    % save candidate birth:
                    states_birth{j} = state_birth;
                    L_birth(j, j) = -l_birth;
                end
            end

            if ~isempty(obj.paras.MBM.tt)
                N_d = length(obj.paras.MBM.tt);
                
                %
                % cache likelihood matrix, detection sub-matrix:
                %
                states_meas = cell(N_d, 1);
                L_miss = cell(N_d, 1);
                L_update = cell(N_d, 1);

                for i = 1:N_d
                    H_i = length(obj.paras.MBM.tt{i});

                    states_meas{i} = repmat(obj.paras.MBM.tt{i}(1, 1), H_i*(1 + N_z), 1);
                    L_miss{i} = zeros(H_i, 1);
                    L_update{i} = cell(H_i, 1);

                    for h_i = 1:H_i
                        % create tt_entry:
                        tt_entry = [i, h_i];

                        % cache miss detection posterior:
                        [state_miss, l_miss] = Bern_undetected_update(obj,tt_entry,sensormodel.P_D);
                        states_meas{i}((h_i-1)*(1 + N_z) + 1 + N_z, 1) = state_miss;
                        L_miss{i}(h_i, 1) = l_miss;

                        %
                        % perform ellipsoidal gating for potential object
                        % detections:
                        %
                        % get predicted state:
                        state = obj.paras.MBM.tt{i}(h_i).state;

                        % init posterior states:
                        L_update{i}{h_i} = Inf * ones(N_z, 1);

                        % get indices of measurements in gate:
                        [z_mb, I_mb] = obj.density.ellipsoidalGating(state, z, measmodel, gating.size);
                        [~, N_mb] = size(z_mb);
                        if N_mb > 0
                            % first, cache likelihood matrix, update sub-matrix:
                            L_update{i}{h_i}(I_mb, 1) = l_miss - Bern_detected_update_lik(obj,tt_entry,z_mb,measmodel,sensormodel.P_D);

                            % for each potential object detection in gate,
                            % create corresponding posterior:
                            J_mb = find(I_mb);
                            for j = 1:N_mb
                                states_meas{i}((h_i-1)*(1 + N_z) + J_mb(j), 1) = Bern_detected_update_state(obj,tt_entry,z_mb(:, j),measmodel);
                            end
                        end
                    end
                end

                %
                % solve optimal assignment problem for each global hypothesis:
                %
                H = length(obj.paras.MBM.w);

                M_h = ceil(exp(obj.paras.MBM.w)*M);
                M_H = cumsum(M_h);

                prior_index = zeros(M_H(H, 1), 1);
                w = -Inf * ones(M_H(H, 1), 1);
                J_assignment = zeros(N_d + N_z, M_H(H, 1));

                for h = 1:H                    
                    % get current global hypothesis:
                    ht = obj.paras.MBM.ht(h, :);
                    
                    % get prior log likelihood:
                    w_prior = obj.paras.MBM.w(h, 1);
                    for i = 1:N_d
                        h_i = ht(1, i);
                        if h_i > 0
                            w_prior = w_prior + L_miss{i}(h_i, 1);
                        end
                    end
                    
                    % construct likelihood matrix:
                    L_h = Inf * ones(N_z, N_d + N_z);

                    for i = 1:N_d
                        if ht(i) > 0
                            L_h(:, i) = L_update{i}{ht(i)};
                        end
                    end
                    L_h(:,(N_d+1):(N_d + N_z)) = L_birth;
                    
                    I_h = false(N_z, 1);
                    for i = 1:(N_d+N_z)
                        I_h = I_h | (L_h(:, i) ~= Inf);
                    end
                    
                    J_h = false(1, N_d + N_z);
                    for j = 1:N_z
                        J_h = J_h | (L_h(j, :) ~= Inf);
                    end
                    
                    if ~(any(I_h) && any(J_h))
                        continue;
                    end
                    
                    % find optimal assignment:
                    [~, J_assignment_h, w_likelihood] = kBest2DAssign(L_h(I_h, J_h), M_h(h, 1));
                    w_likelihood = -w_likelihood;
                    
                    % map to actual measurement index:
                    I_h = find(I_h);
                    J_h = find(J_h);
                    [~, H_h] = size(J_assignment_h);
                    % disp(I_h)
                    % disp(J_assignment_h)
                    for h_h = 1:H_h
                        for j = 1:length(J_h)
                            if J_assignment_h(j, h_h) > 0
                                J_assignment_h(j, h_h) = I_h(J_assignment_h(j, h_h), 1);
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
                    w(candidate_hypotheses_index, 1) = w_likelihood + w_prior;
                    J_assignment(J_h, candidate_hypotheses_index) = J_assignment_h;
                end
                
                % prune candidates with small weight:
                I = (w ~= -Inf);
                
                if any(I)
                    % prune invalid candidates:
                    prior_index = prior_index(I);
                    w = w(I);
                    J_assignment = J_assignment(:, I);

                    % normalize weights:
                    [w, ~] = normalizeLogWeights(w);

                    % prune candidates with small weight:
                    I = (w >= w_min);
                    prior_index = prior_index(I);
                    w = w(I);
                    J_assignment = J_assignment(:, I);

                    % capping:
                    [~, I] = maxk(w, M);
                    prior_index = prior_index(I);
                    w = w(I);
                    J_assignment = J_assignment(:, I);

                    % normalize weights --- posterior MBM w is ready:
                    [w, ~] = normalizeLogWeights(w);

                    % update global hypothesis look-up table:
                    obj.paras.MBM.w = w;

                    % disp(prior_index)
                    % disp(J_assignment)

                    % init posterior ht:
                    H = length(obj.paras.MBM.w);
                    ht = zeros(H, N_d);

                    % process detected objects:
                    for i = 1:N_d
                        % init mask for posterior local hypothesis:
                        posterior_local_hypo_mask = false(length(states_meas{i}), 1);
                        for h = 1:H
                            % get local hypothesis index:
                            h_i = obj.paras.MBM.ht(prior_index(h), i);

                            if h_i > 0
                                % get assigned measurement index:
                                j = J_assignment(i, h);
                                if j == 0
                                    j = N_z + 1;
                                end

                                % get posterior local hypothesis index:
                                posterior_local_hypo_index = (h_i - 1)*(1 + N_z) + j;
                                
                                % disp([i, h_i])
                                % disp(obj.paras.MBM.tt{i}(h_i).state.x')
                                % disp('--->')
                                % disp(states_meas{i}(posterior_local_hypo_index).state.x')
                                % disp('--UPDATE DONE---')
                                
                                % mark candidate as selected:
                                posterior_local_hypo_mask(posterior_local_hypo_index, 1) = true;

                                % record it in global hypothesis look-up table:
                                ht(h, i) = posterior_local_hypo_index;
                            else
                                ht(h, i) = 0;
                            end
                        end

                        % create encoder for posterior local hypotheses:
                        posterior_local_hypo_index = find(posterior_local_hypo_mask);

                        if ~isempty(posterior_local_hypo_index)
                            [keys, values] = sort(posterior_local_hypo_index);
                            posterior_local_hypo_encoder = containers.Map(keys, values);

                            % set posterior global hypotheses:
                            for h = 1:H
                                if ht(h, i) > 0
                                    ht(h, i) = posterior_local_hypo_encoder(ht(h, i));
                                end
                            end

                            % set posterior local hypotheses:
                            obj.paras.MBM.tt{i} = states_meas{i}(posterior_local_hypo_mask, 1); 
                        end
                    end

                    % process potential new birth objects:
                    N_b = N_d;
                    for i = (N_d + 1):(N_d + N_z)
                        if any(J_assignment(i, :) > 0)
                            N_b = N_b + 1;
                            ht(:, N_b) = int8(J_assignment(i, :) > 0)';
                            obj.paras.MBM.tt{N_b} = repmat(states_birth{i - N_d}, 1, 1);
                        end
                    end

                    % set posterior global hypotheses:
                    obj.paras.MBM.ht = ht;
                    N_l = 0;
                    for i = 1:N_b
                        if ~all(ht(:, i) == 0)
                            N_l = N_l + 1;
                            obj.paras.MBM.ht(:, N_l) = ht(:, i);
                            obj.paras.MBM.tt{N_l} = obj.paras.MBM.tt{i};
                        end
                    end
                    obj.paras.MBM.ht = obj.paras.MBM.ht(:, 1:N_l);
                    tt = cell(N_l, 1);
                    for i = 1:N_l
                        tt{i} = obj.paras.MBM.tt{i};
                    end
                    obj.paras.MBM.tt = tt;
                end
            end

            if isempty(obj.paras.MBM.tt)
                % if no detected object yet, initialize MBM:    
                N_b = 0;
                for j = 1:N_z
                    if L_birth(j, j) ~= Inf
                        N_b = N_b + 1;
                    end
                end
                
                obj.paras.MBM.w = zeros(1, 1);
                obj.paras.MBM.ht = ones(1, N_b);
                obj.paras.MBM.tt = cell(N_b, 1);
                for j = 1:N_z
                    if L_birth(j, j) ~= Inf
                        obj.paras.MBM.tt{j} = repmat(states_birth{j}, 1, 1);
                        % disp(states_birth{j}.state.x')
                    end
                end
                % disp(L_birth)
                disp('Insert new birth directly.')
            end
            
            % update mixture components in PPP:
            obj = PPP_undetected_update(obj,sensormodel.P_D);
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
