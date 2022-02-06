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
            N_z = size(z,2);

            %
            % perform gating for undetected objects:
            %
            N_u = length(obj.paras.PPP.states);
            I_u = false(N_z,N_u);
            I_z_u = false(N_z,1);           
            for i = 1:N_u
                % identify possible measurements from current undetected object:
                [~,I_u(:,i)] = ...
                    obj.density.ellipsoidalGating(obj.paras.PPP.states(i),z,measmodel,gating.size);
                % track all measurements that are inside the gate of 
                % any undetected objects:
                I_z_u = I_z_u | I_u(:,i);
            end
            
            %
            % perform gating for detected objects:
            %
            N_d = length(obj.paras.MBM.tt);
            L_detected = cell(N_d,1);
            I_d = cell(N_d,1);
            I_z_d = false(N_z,1);
            for i = 1:N_d
                % num. of local hypotheses for object i:
                H_i = length(obj.paras.MBM.tt{i});
                % construct gating matrix:
                I_d{i} = false(N_z,H_i);
                for j = 1:H_i
                    % identify possible measurements from current detected object:
                    [~,I_d{i}(:,j)] = ...
                        obj.density.ellipsoidalGating(obj.paras.MBM.tt{i}(j).state,z,measmodel,gating.size);
                    % track all measurements that are inside the gate of
                    % any detected objects:
                    I_z_d = I_z_d | I_d{i}(:,j);
                end
            end

            %
            % identify measurements that could come from 
            % either detected or undetected objects
            %
            I_z_u_and_d = I_z_u(I_z_d);
            %
            % identify measurements that could only come from undetected
            % objects
            %
            I_z_u_not_d = (I_z_u & (~I_z_d));
            
            %
            % generate candidate local posterior hypotheses for detected
            % objects
            %
            z_d = z(:,I_z_d);
            N_z = size(z_d,2);
            I_d = cellfun(@(x) x(I_z_d,:), I_d, 'UniformOutput',false);              
            tt = cell(N_d+N_z,1);
            for i = 1:N_d
                % num. of local hypotheses for object i:
                H_i = length(obj.paras.MBM.tt{i});
                
                % initialise likelihood table for hypothesis tree i
                L_detected{i} = -inf(H_i,1+N_z);
                % initialise hypothesis table for hypothesis tree i
                tt{i} = cell(H_i*(1+N_z),1);
                
                for h_i = 1:H_i
                    offset = (h_i-1)*(1+N_z);
                    tt_entry = [i, h_i];
                    
                    % possible miss detection:
                    [tt{i}{offset+1},L_detected{i}(h_i,1)] = ...
                        Bern_undetected_update(obj,tt_entry,sensormodel.P_D);
                    
                    % possible object detections:
                    L_detected{i}(h_i,[false;I_d{i}(:,h_i)]) = ...
                        Bern_detected_update_lik(obj,tt_entry,z_d(:,I_d{i}(:,h_i)),measmodel,sensormodel.P_D);
                    for j = 1:N_z
                        if I_d{i}(j,h_i)
                            tt{i}{offset+1+j} = Bern_detected_update_state(obj,tt_entry,z_d(:,j),measmodel);
                        end
                    end
                end
            end
            
            %
            % generate candidate local posterior hypotheses for new birth
            % objects
            %
            L_birth = -inf(N_z,N_z);
            I_u_and_d = I_u(I_z_d,:);
            for j = 1:N_z
                if any(I_u_and_d(j,:))
                    [tt{N_d+j,1}{1}, L_birth(j,j)] = ...
                        PPP_detected_update(obj,I_u_and_d(j,:),z_d(:,j),measmodel,sensormodel.P_D,sensormodel.intensity_c);
                else
                    %
                    % for possible detected object measurements 
                    % not inside the gate of undetected objects
                    % set likelihood to clutter intensity
                    %
                    L_birth(j,j) = log(sensormodel.intensity_c);
                end
            end 
            
            %
            % init candidate posterior hypotheses:
            %
            H_prior = length(obj.paras.MBM.w);
            
            H_posterior = 0;
            w = [];             
            ht = zeros(0,N_d+N_z);
            
            if H_prior == 0 
                % if there is no global hypothesis:
                H_posterior = 1;
                w = 0;
                ht = zeros(1,N_z);
                ht(I_z_u_and_d) = 1;
            else
                %
                % generate candidate posterior hypotheses from prior
                % through optimal assignment
                %
                for h = 1:H_prior
                    % init candidate posterior likelihood:
                    w_prior = obj.paras.MBM.w(h);
                    
                    % build likelihood matrix:
                    L = inf(N_z,N_d + N_z);
                    for i = 1:N_d
                        h_i = obj.paras.MBM.ht(h,i);
                        if h_i > 0
                            % update likelihoods from detected object:
                            L(:,i) = -(L_detected{i}(h_i,2:end) - L_detected{i}(h_i,1));
                            % update candidate posterior likelihood:
                            w_prior = w_prior + L_detected{i}(h_i,1);
                        end
                    end
                    % update likelihoods from possible undetected objects:
                    L(:, (N_d+1):(N_d+N_z)) = -L_birth;
                    
                    if isempty(L)
                        %Consider the case that no measurements are inside
                        %the gate, thus missed detection
                        w_likelihood = 0;
                        I_assignment = 0;
                    else
                        % get M for current hypothesis:
                        M_h = ceil(exp(obj.paras.MBM.w(h)+log(M)));
                        
                        % solve with Murty's algorithm:
                        [I_assignment,~,w_likelihood] = ...
                            kBest2DAssign(L,M_h);
                        % solve with Gibbs sampling:
                        % [I_assignment,~,w_likelihood] = ...
                        %   assign2DByGibbs(L,100,M_h);
                    end
                    
                    % extract candidate posterior hypotheses:
                    H_posterior_h = length(w_likelihood);
                    ht_h = zeros(H_posterior_h,N_d+N_z);
                    for h_posterior = 1:H_posterior_h
                        % init:
                        ht_h(h_posterior,:) = 0;
                        
                        % handle possible object detections:
                        for i = 1:N_d
                            h_i = obj.paras.MBM.ht(h,i);
                            offset = (h_i-1)*(1+N_z);
                            
                            if h_i > 0
                                j = find(I_assignment(:,h_posterior)==i, 1);
                                if isempty(j)
                                    % get posterior from miss detection:
                                    ht_h(h_posterior,i) = offset+1;
                                else
                                    % get posterior from object detection:
                                    ht_h(h_posterior,i) = offset+1+j;
                                end
                            end
                        end
                        
                        % handle possible new birth detections:
                        for i = (N_d+1):(N_d+N_z)
                            j = find(I_assignment(:,h_posterior)==i, 1);
                            if ~isempty(j) && I_z_u_and_d(j)
                                % get posterior from object birth:
                                ht_h(h_posterior,i) = 1;
                            end
                        end
                    end
                    
                    % update candidate posterior hypotheses:
                    H_posterior = H_posterior + H_posterior_h;
                    w = [w;w_prior-w_likelihood];
                    ht = [ht;ht_h];
                end
            end
            
            %
            % for all measurements that are only inside the gate of undetected
            % append all of them into existing global hypotheses
            %
            z_u_not_d = z(:,I_z_u_not_d);
            N_z_u_not_d = size(z_u_not_d,2);
            I_u_not_d = I_u(I_z_u_not_d,:);
            for i = 1:N_z_u_not_d
                [tt{N_d+N_z+i,1}{1}, ~] = ...
                    PPP_detected_update(obj,I_u_not_d(i,:),z_u_not_d(:,i),measmodel,sensormodel.P_D,sensormodel.intensity_c);
            end
            %
            % this expansion will add a commom multiplication factor to
            % all existing candidate posterior hypotheses
            %
            w = normalizeLogWeights(w);
            ht = [ht ones(H_posterior,N_z_u_not_d)];
            
            % update undetected objects that stay undetected:
            obj = PPP_undetected_update(obj,sensormodel.P_D);
            
            % prune:
            [w, idx] = hypothesisReduction.prune(w,1:H_posterior,w_min);
            w = normalizeLogWeights(w);
            ht = ht(idx,:);
            
            % capping:
            [w, idx] = hypothesisReduction.cap(w,1:length(w),M);
            ht = ht(idx,:);
            
            % set posterior hypothesis weights:
            obj.paras.MBM.w = normalizeLogWeights(w);
            
            % remove unused objects:
            if ~isempty(ht)
                idx = sum(ht,1) >= 1;
                ht = ht(:,idx);
                tt = tt(idx);
            end
            
            % get num. of detected objects:
            N_d = size(ht,2);
            
            % compress local hypotheses:  
            obj.paras.MBM.tt = cell(N_d,1);
            for i = 1:N_d
                ht_i = ht(:,i);
                tt_i = tt{i}(unique(ht_i(ht_i~=0), 'stable'));
                obj.paras.MBM.tt{i} = [tt_i{:}]';
            end
            
            % re-index global hypothesis table:
            for i = 1:N_d
                idx = ht(:,i) > 0;
                [~,~,ht(idx,i)] = unique(ht(idx,i),'rows','stable');
            end
            
            % set posterior hypothesis look-up table:
            obj.paras.MBM.ht = ht;
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
