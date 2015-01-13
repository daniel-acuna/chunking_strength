%% analyze data for FMRI study
% add path
% load data from movement times and fingers
fd = load('../chunk_inference/full_data/mt_with_fingers.mat');
% add code for chunking inference
addpath('../chunk_inference/src/');
%% select what to fit
fit_er = true;
fit_er_er = true;
fit_rt = true;
fit_rt_rt = true;
fit_T = true;
fit_rho = true;
fit_rho_er = true;
all_results = {};
% check how sessions are organized
parfor subject_id = sort(unique(fd.mt.subject_id))'
    try
        scan_chunk_strength_ds = dataset();
        for sequence_id = 1:2
            disp(['Analyzing subject ' num2str(subject_id)]);
            disp(['Analyzing sequence ' num2str(sequence_id)]);
            subject_data = fd.mt(fd.mt.subject_id == subject_id & ...
                fd.mt.sequence_id == sequence_id, :);

            % perform detrending        
            % remove outliers
            subject_data = subject_data(subject_data.sequence_press <= 10, :);

            std_mt = std(subject_data.movement_time);
            mean_mt = mean(subject_data.movement_time);

            idx_good = (subject_data.movement_time <= mean_mt + 3*std_mt) & ...
                (subject_data.movement_time >= mean_mt - 3*std_mt);

            subject_data = subject_data(idx_good, :);

            % perform detrending
            % detrend
            exponential_model = ['movement_time ~ a0' ...
                '+ a1*exp((b1/100)*(sequence_trial-1))'];
            initial_values = [0.18 0.38 -0.17];        

            opts = statset('Display','off','TolFun',1e-5, ...
                'MaxIter', 100);

            % Fit
            nlmf = NonLinearModel.fit(subject_data, ...
                exponential_model, initial_values, 'Options', opts);

            % add detrending movement time to dataset
            subject_data.detrended_mt = nlmf.Residuals.Raw;

            % get the non-scan sessions
            scan_ids = find(subject_data.session_type == 99);
            % start
            scan_starts = scan_ids([false (diff(scan_ids) > 1)']);

            nonscan_ids = find(subject_data.session_type ~= 99);
            scan_ends = nonscan_ids([false (diff(nonscan_ids) > 1)']) - 1;
            if length(scan_ends) < length(scan_starts)
                scan_ends(end+1) = size(subject_data, 1); %#ok<SAGROW>
            end

            % analyze the trials before scans
            for scan_session_id = length(scan_starts)
                % trials before scan sessions
                trials_b4 = subject_data(1:(scan_starts(scan_session_id) - 1), :);
                trials_b4 = trials_b4(trials_b4.session_type ~= 99, :);

                % get matrix form
                [rt_seq, er_seq, trial_seq, day_seq] = ...
                    mt_to_seq_trial(trials_b4, ...
                    trials_b4.movement_time - nlmf.predict(trials_b4), ...
                    trials_b4.error);
                % checking that the trial sequence and day remain the same
                if (any(any(diff(trial_seq, [], 2))) == true) || ...
                        (any(any(diff(day_seq, [], 2))) == true)
                    error('Trial sequence or day sequence are not the same during training');
                end
                % learn chunking parameters


                chunk_structures = create_chunks_nospace('n_seqlen', size(rt_seq, 2));
                [rho, self_t, ~, fm, T, rho_er, v, v_er, ...
                    initial_dist, mean_pause, mean_inchunk, ...
                    mean_pause_er, mean_inchunk_er, ...
                    chunks, cor_chunks, ~] = ...
                    chunk_hmm_learn_param(rt_seq, er_seq, 'verbose', false, ...
                    'fit_rt', fit_rt, 'fit_rt_rt', fit_rt_rt, ...
                    'fit_er', fit_er, 'fit_er_er', fit_er_er, ...
                    'fit_T', fit_T, 'fit_rho', fit_rho, 'fit_rho_er', fit_rho_er, ...
                    'chunks', chunk_structures);

                % run HMM algorithm on scan sessions
                trials_scans = ...
                    subject_data(scan_starts(scan_session_id):scan_ends(scan_session_id), :);
                % further cleaning
                trials_scans = trials_scans(trials_scans.session_type == 99, :);

                % transform into matrix form
                [rt_seq_test, er_seq_test, trial_seq_test, day_seq_test] = ...
                    mt_to_seq_trial(trials_scans, ...
                    trials_scans.movement_time - nlmf.predict(trials_scans), ...
                    trials_scans.error);
                
                if (any(any(diff(trial_seq_test, [], 2))) == true) || ...
                        (any(any(diff(day_seq_test, [], 2))) == true)
                    error('Trial sequence or day sequence are not the same during testing');
                end
                
                % performance on testing data
                [~, ~, ~, p_obs_un] = ...
                    create_emission_for_chunks(rho, rho_er, ...
                    self_t, rt_seq_test, ...
                    er_seq_test, 'v', v, 'v_er', v_er, ...
                    'fit_er', fit_er || fit_er_er, ...
                    'fit_mt', fit_rt || fit_rt_rt, ...
                    'mean_pause', mean_pause, ...
                    'mean_inchunk', mean_inchunk, ...
                    'mean_pause_er', mean_pause_er, ...
                    'mean_inchunk_er', mean_inchunk_er, ...
                    'chunks', chunks, ...
                    'cor_chunks', cor_chunks);

                % run forward-backward algorithm
                [fm, ~, gamma, log_like, marg_epsilon] = ...
                    hmm_inference(p_obs_un, T, 'initial_dist', fm(end, :));

                entropy = sum(-gamma .* log(gamma), 2);
                
                % add entropy to dataset
                scan_chunk_strength_ds = [scan_chunk_strength_ds;
                    dataset({repmat(subject_id, size(rt_seq_test, 1), 1), 'subject_id'}, ...
                    {repmat(sequence_id, size(rt_seq_test, 1), 1), 'sequence_id'}, ...
                    {repmat(scan_session_id, size(rt_seq_test, 1), 1), 'scan_session'}, ...
                    {trial_seq_test(:, 1), 'within_day_trial'}, ...
                    {day_seq_test(:, 1), 'training_day'}, ...
                    {entropy(2:end), 'entropy'})]; 

            end
        end
    catch me
        disp(['Subject ' num2str(subject_id) ' failed']);
        disp(me.message);
    end
    all_results{subject_id} = scan_chunk_strength_ds;
end

% % %% save results
final_results = (vertcat(all_results{:}));
final_results.training_day = final_results.training_day + 1;
% %%
% export(final_results, 'File', 'scan_sessions_chunk_strengths.cvs', ...
%     'delimiter', ',')
% %%
