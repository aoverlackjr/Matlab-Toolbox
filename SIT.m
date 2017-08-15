classdef SIT < handle
    % System Identification Toolkit
    
    properties
    end
    
    methods
        
        function sit = SIT()
            
        end
        
    end
    
    methods (Static)
        
        function state_transition_matrix = transition_matrix(data, dt)
            
            % Asses size of data. Data should be inserted as column data,
            % where each row is equally spaced in time.
            [nSteps,nVar] = size(data);

            % Get the time derivatives of the data:
            var_dot = SIT.differentiate(data,dt);

            % Pre-allocate the aggregate transition matrix.
            TM = zeros(nVar*(nSteps-1),nVar*nVar);

            % Pre allocate the aggregate state matrix.
            V = zeros(nVar*(nSteps-1),1);

            % Fill the aggregate matrix and vector.
            for i = 1:nVar
                row = (i-1)*nSteps + 1;
                col = 1 + (i-1)*nVar;
                TM(row:row+nSteps-2,col:col+nVar-1) = data(1:end-1,:); 
                V(row:row+nSteps-2,1) = var_dot(:,i);
            end
            
            % Perform pseudo inverse to obtain the state transition matrix in
            % vector form.
            pseudo_inverse = (TM'*TM)\TM';
            vectorMatrix = pseudo_inverse*V;

            % Refactor the state transition matrix from the vector
            state_transition_matrix = zeros(nVar,nVar);
            for r = 1:nVar
                base = 1+(r-1)*nVar;
                state_transition_matrix(r,:) = vectorMatrix(base:base+nVar-1,1)';
            end
            
            % The matrix is still in the form x_1 = x_0 + A*x_0;
            % Adding the unity matrix will make it of the form: x_1 =
            % A*x_0;
            state_transition_matrix = state_transition_matrix + eye(nVar);
            
        end
        
        function var_dot = differentiate(data,dt)
           
            [nSteps,nVar] = size(data);
            
            var_dot = zeros(nSteps-1,nVar);

            % Calculate the first derivative.
            for vr = 1:nVar
                for step = 1:nSteps-1
                    var_dot(step,vr) = ( data(step+1,vr) - data(step,vr) ) / dt;
                end
            end
            
        end
        
        function [state_matrices_history, bias_matrix, std_matrix] = examine_state_matrix_evolution(data,window, dt)
            % Runs along the data, including more every time, and generates
            % an evolution of TM matrices, to see if it converges over
            % time, an as such if the matrix is converging to an inherent
            % behaviour.
            
            % Get data size:
            [nPoints,nVars] = size(data);
            
            % If window is set to -1 the window is incrementally increased.
            if window == -1
               
                % Pre allocate results
                state_matrices_history = zeros(nPoints,nVars,nVars);

                % First data set must include at least a data length of nVars,
                % otherwise the pseudo inverse does not work
                index = 1;
                for end_pt = nVars:nPoints
                    input_data = data(1:end_pt,:);
                    state_matrices_history(index,:,:) = SIT.transition_matrix(input_data, dt);
                    index = index + 1;
                end
                
            else
               
                end_point = nPoints-window+1;
                state_matrices_history = zeros(end_point,nVars,nVars);
                ind = 1;
                for start_index = 1:end_point
                    input_data = data(start_index:start_index+window-1,:);
                    state_matrices_history(ind,:,:) = SIT.transition_matrix(input_data, dt);
                    ind = ind + 1;
                end
                
                
            end
            
            % Get the bias of the elements (mean) as wel as the std
            % deviation.
            bias_matrix = zeros(nVars,nVars);
            std_matrix  = zeros(nVars,nVars);
            for i = 1:nVars
                for j = 1:nVars
                    bias_matrix(i,j) = mean(state_matrices_history(:,i,j)); 
                    std_matrix(i,j) = std(state_matrices_history(:,i,j));
                end
            end
                
            
        end
        
        function [output] = filter_gaussian_kernel(data, kernel_size, sigma)
            % Runs gaussian kernel over the data. The input space is an
            % integer which defines the window size over which the kernel
            % is run.
            % Sigma is the standard deviation of the Gaussian bell curve.
            
            % Assuming for now data is in column format.
            [nPoints,~] = size(data);
            
            % Pre-allocate the output:
            output = zeros(nPoints,1);
            
            % Run past all the data points
            for ptNr = 1:nPoints
               
                % Set usm of weights to zero
                K_sum = 0.0;
                out_data_pt = 0.0;
                
                % Run past all the points in the input space
                for wPt = -kernel_size:1:kernel_size
                   
                    % Get the index of the data according to the point in
                    % the window.
                    data_index = ptNr + wPt;
                    
                    % If the window's point is within the data set:
                    if (data_index >= 1) && (data_index <= nPoints)
                       
                        % Get the weight factor:
                        K = exp( -(ptNr - wPt)^2 / (2*sigma^2));
                        K_sum = K_sum + K;
                        
                        % Add to the output:
                        out_data_pt = out_data_pt + K*data(data_index);
                        
                    end

                end
                
                % Divide the output by the sum of weigths to normalize:
                out_data_pt = out_data_pt/K_sum;
                % Put data in output:
                output(ptNr,1) = out_data_pt;
                
            end
            
        end
        
        function [std_dev, residual, upper_lower_residual] = extract_filtered_properties(data, filtered_data)
            % Function extracts some data from the original data and the
            % filtered data, in order to analyze statistically the
            % performance of the filter.
            % Assumes the data and filtered_data are colums, and are of
            % equal length.
            
            % Check how big the data is.
            [nPoints,~] = size(data);
            
            % Pre-allocate the residual data.
            upper_lower_residual = zeros(nPoints,1);
            residual = data - filtered_data;
            
            % Get standard deviation from the residual data.
            std_dev = std(residual);
            
            % Get the positive and negative residuals
            for i = 1:nPoints
                if residual(i) > 0
                    upper_lower_residual(i) = 1; %residual(i);
                elseif residual(i) < 0
                    upper_lower_residual(i) = -1; %residual(i);
                end
            end
            
        end
        
        function [max_lag_period] = get_lag_period(upper_lower_residual)
           % Gets the length (in steps) of the leag lead period, on the
           % basis of the upper_lower residual vector.
           % Used for filter tuning.
           
           % Get size of input data.
           [nPoints,~] = size(upper_lower_residual);
           
           % Set output to zero
           max_lag_period = 0;
           
           prev_point = upper_lower_residual(1);
           accumulated_period = 0;
           
           % Loop all points to find the maximum length of a period.
           for i = 2:nPoints
               
               if upper_lower_residual(i) == prev_point
                   % If previous point is same as current one, add to
                   % measured period length and move forward.
                   accumulated_period = accumulated_period + 1;                   
               else
                   % Apparently the period has come to an end.
                   % Check if this period was bigger than the previous one:
                   if accumulated_period > max_lag_period
                       % Store new maximum:
                       max_lag_period = accumulated_period;
                   end
                   
                   % Set accumulated period back to zero
                   accumulated_period = 0;
                   
               end
               
               prev_point = upper_lower_residual(i);
               
           end
            
            
        end
        
        function [output, std_dev_error, kernel_tuned, lags] = find_optimal_gaussian_filter(data, max_kernel, max_lag)
            % This function loops the filter with different sigma values,
            % until the lag properties are matched to the sigma values.
            % Heuristially, this is assumed to be a good filter, without
            % too much lag, but with good noise filtering.
            
            % Set to initial value:
            kernel_tuned = 1;
            % Loop the length of the input space, just to cover the whole
            % ground. Should stop earlier though.
            lags = zeros(max_kernel,1);
            for kernel_test = 1:max_kernel
                % By starting with sigma as 1, we start with a very noisy
                % first filter. (all-pass)
                % Try out the filter:
                output = SIT.filter_gaussian_kernel(data, kernel_test, max_kernel);
                % Get the properties with this output.
                [std_dev_error, ~, upper_lower_residual] = SIT.extract_filtered_properties(data, output);
                % See if the lag period is about the same as the sigma.
                max_lag_period = SIT.get_lag_period(upper_lower_residual);
                lags(kernel_test) = max_lag_period;
                % If the max lag is the same or bigger as the sigma value,
                % the testing is stopped.
                if max_lag_period > max_lag
                    kernel_tuned = kernel_test-1;
                    break;
                end
                
            end
            %kernel_tuned = -1;
        end
        
    end
    
end
