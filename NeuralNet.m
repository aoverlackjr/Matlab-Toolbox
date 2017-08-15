classdef NeuralNet < handle
    
% Variable Selection: Input layer will need to have as many nodes as there are inputs. 
% Use decision trees or random forest to first identify the relevant inputs. 
% You can also use dependence, correlation and dimensionality reduction techniques.
% Topology: 90% - One hidden Layer and 10% -Two hidden layer. 
% You may go more than that but then there will be a high chance of over fitting in the model.
% Number of Nodes: One hidden node for each class. 0.5 to 3 times the input neurons. 
% There's a geometric pyramid rule that says that where input has m nodes and output has n nodes, the hidden layer should have sqrt(m?n)
% Nodes and Data: H*(I+O)+H+O. H=Hidden Layer, I=Input , O=Output.
% Weights: The weights should be 1/100th of the amount of training data set.
% Baum-Haussler Rule : Used to determine the number of neurons in hidden layer. 
% Nhidden ? (Ntrain x Etolerance) / (Npts + Noutputs) Page on buffalo.edu 
% Sensitivity: Measure the output of the network when all inputs are at their average value. 
% Average value being the center of the test set. 
% measure the output of the network when each input is modified,one at a time,to be at it's min and max values(-1 and 1).
     
    properties % Base properties
        
        % Structural parameters
        NR_OF_NODES;
        NR_OF_INPUTS;
        NR_OF_HIDDEN;
        NR_OF_OUTPUTS;
        OUTP_ADDR;
        WEIGHT_ADDR;
        
        % Settings
        LEARNING_RATE                           = 1;
        DELTA_RELATIVE                          = 0.01;
        DELTA_ABSOLUTE                          = 0.01;
        ABSOLUTE                                = 1;
        
        RETAIN_CONNECTIONS                      = 1;
        MEMORY                                  = 0;
        WEIGHT_NORMALIZATION                    = 1;
        
        DATA_SIZE_LEARNING_RATE_COMPENSATION    = 0;
        ZERO_INIT_BIAS                          = 0;
        BIAS_OFF                                = 0;
        SUPERVISOR_ON                           = 0;
        MOMENTUM_ON                             = 0;
        OUTPUT_ERROR_TOLERANCE                  = 0.01;
        
        % Data and functionality
        bias_vector;
        weight_matrix;
        signal_field;
        fire_func;
        tanh_flattening = 1;
        
        % Learning settings
        supervisor;
        sup_n_filter                            = 4;            % Number of points used for filtering the cost output to eliminate the effect of noise.
        cdl                                     = 0.00001;      % cost derivative limit, used to decide of learning has stagnated.
        momentum_lambda;
        mem;
        cost_tolerance;
        COST_CRITERIUM_REACHED = 0;
        Q_MATRIX_ADAPTATION = 0;    % Bool that controls whether the Q-matrix of the UKF is updated.
        
        % Debugging settings
        VERBOSE_LEARNING = 0;
        
    end
    
    % Neural network training specific UKF properties
    properties (GetAccess = 'public', SetAccess = 'private')
        P0;                         % Error covariance matrix (initial)
        R;                          % Current standard (static) noise matrix
        R_actual;                   % Corrected (used) noise matrix
        Q;                          % Process noise matrix
        P;                          % Error covariance matrix (storage)
        notifications = {};         % Notifications list
        UKFcallNr;                  % The number of calls to the filter while in existence
        K;                          % The Kalman gain
        alpha;                      % UKF constant
        ki;                         % UKF constant
        beta;                       % UKF constant
        lambda;                     % UKF constant; defined by alpha, ki and number of states
        c;                          % UKF constant; defined by number of states and lambda
        sqrt_c;                     % The square root of c;
        W_mean;                     % Weights for the means of the unscented transform
        W_cov;                      % Weights for the covariances of the unscented transform
        n_states;                   % Number of states of the UKF
        n_measurements;             % Number of measurements of the UKF
        nr_of_copoints;             % Two times the nr of states plus one
        X;                          % Matrix containing sigma points
        LAMB_Q              = 0.01;
        store;
        state_prev;
        INITIALIZED = 0;
    end
    
    methods % Base NN methods
        
        function obj = NeuralNet(nInputs, nHidden, nOutputs, MEM_BOOL, BIAS_BOOL)
            if BIAS_BOOL == 1
                obj.BIAS_OFF = 0;
            elseif BIAS_BOOL == 0
                obj.BIAS_OFF = 1;
            end
            obj.MEMORY = MEM_BOOL;
            obj.create_network(nInputs, nHidden, nOutputs);
            obj.populate_weights(MEM_BOOL, obj.ZERO_INIT_BIAS);
            obj.use_sigmoid;
        end
        
        function create_network(obj, nInputs, nHidden, nOutputs)
           
            % Some housekeeping
            obj.NR_OF_NODES   = nInputs + sum(nHidden) + nOutputs;
            obj.NR_OF_INPUTS  = nInputs;
            obj.NR_OF_HIDDEN  = nHidden;
            obj.NR_OF_OUTPUTS = nOutputs;
            
            % Pre-allocating the weight, bias and signal storage
            % Bias-space is also allocated for input nodes, this is done
            % for ease of index operations. Biases for input nodes not
            % actually used in calculations.
            obj.bias_vector   = zeros(obj.NR_OF_NODES,1);
            obj.weight_matrix = zeros(obj.NR_OF_NODES);
            obj.signal_field  = zeros(obj.NR_OF_NODES,1); 
            
            % By default, outputs are at the end of the signal field.
            obj.OUTP_ADDR = obj.NR_OF_NODES-(obj.NR_OF_OUTPUTS-1):obj.NR_OF_NODES;
            
        end
        
        function create_weight_addr(obj)
            obj.WEIGHT_ADDR = [];
            index = 1;
            for row = 1:obj.NR_OF_NODES
                for col = 1:obj.NR_OF_NODES
                    if obj.weight_matrix(row,col) ~= 0
                        obj.WEIGHT_ADDR(index,:) = [row, col];
                        index = index+1;
                    end
                end
            end
        end
        
        function populate_weights(obj, MEM_ON, ZERO_INIT_BIAS)
           
            % Layered network?
            [~,m] = size(obj.NR_OF_HIDDEN);
            
            % Layered network
            %if m > 1
            
                % populate input weights
                obj.weight_matrix(1:obj.NR_OF_INPUTS, obj.NR_OF_INPUTS+1:obj.NR_OF_INPUTS+obj.NR_OF_HIDDEN(1)) = randn(obj.NR_OF_INPUTS, obj.NR_OF_HIDDEN(1));
                
                for layerNr = 1:m
                   
                    if layerNr == m
                        
                        n_start = obj.NR_OF_NODES - obj.NR_OF_OUTPUTS - obj.NR_OF_HIDDEN(m) + 1;
                        n_stop  = obj.NR_OF_NODES - obj.NR_OF_OUTPUTS;
                        m_start = obj.NR_OF_NODES - obj.NR_OF_OUTPUTS + 1 ;
                        m_stop  = obj.NR_OF_NODES;
                        n_size  = obj.NR_OF_HIDDEN(m);
                        m_size  = obj.NR_OF_OUTPUTS;
                        
                        if obj.WEIGHT_NORMALIZATION
                            mult = 1/sqrt(n_size*m_size);
                        else
                            mult = 1;
                        end
                        
                        obj.weight_matrix(n_start:n_stop, m_start:m_stop) = randn(n_size, m_size)*mult;
                    
                    elseif layerNr == 1
                        
                        n_start = obj.NR_OF_INPUTS + 1;
                        n_stop  = obj.NR_OF_INPUTS + obj.NR_OF_HIDDEN(1);
                        m_start = obj.NR_OF_INPUTS + obj.NR_OF_HIDDEN(1) + 1;
                        m_stop  = obj.NR_OF_INPUTS + obj.NR_OF_HIDDEN(1) + obj.NR_OF_HIDDEN(2);
                        n_size  = obj.NR_OF_HIDDEN(1);
                        m_size  = obj.NR_OF_HIDDEN(2);
                        
                        if obj.WEIGHT_NORMALIZATION
                            mult = 1/sqrt(n_size*m_size);
                        else
                            mult = 1;
                        end
                        
                        obj.weight_matrix(n_start:n_stop, m_start:m_stop) = randn(n_size, m_size)*mult;

                    else
                        n_start = obj.NR_OF_INPUTS + sum(obj.NR_OF_HIDDEN(1:layerNr-1)) + 1;
                        n_stop  = obj.NR_OF_INPUTS + sum(obj.NR_OF_HIDDEN(1:layerNr));
                        m_start = obj.NR_OF_INPUTS +sum(obj.NR_OF_HIDDEN(1:layerNr)) + 1;
                        m_stop  = obj.NR_OF_INPUTS +sum(obj.NR_OF_HIDDEN(1:layerNr+1));
                        n_size  = obj.NR_OF_HIDDEN(layerNr);
                        m_size  = obj.NR_OF_HIDDEN(layerNr+1);
                        
                        if obj.WEIGHT_NORMALIZATION
                            mult = 1/sqrt(n_size*m_size);
                        else
                            mult = 1;
                        end
                        
                        obj.weight_matrix(n_start:n_stop, m_start:m_stop) = randn(n_size, m_size)*mult;
                        
                    end
                    
                end
                
%             else % Completely connected network
%                 % Generate random initial weights
%                 obj.weight_matrix(:,obj.NR_OF_INPUTS+1:end) = randn(obj.NR_OF_NODES, sum(obj.NR_OF_HIDDEN)+obj.NR_OF_OUTPUTS);
%             end
            
            % No memory in system
            if MEM_ON == 0
                for row = 1:obj.NR_OF_NODES
                    for col = 1:row
                        obj.weight_matrix(row,col) = 0;
                    end
                end
            end
            
            if ZERO_INIT_BIAS == 0
                % Implemented to avoid issue found that first layer of
                % nodes does not 'learn' because biases are pre-set.
                obj.bias_vector = randn(obj.NR_OF_NODES,1);
            else
                obj.bias_vector = zeros(obj.NR_OF_NODES,1);
            end
            
            % If bias is off set to zero anyway.
            if obj.BIAS_OFF == 1
                obj.bias_vector = zeros(obj.NR_OF_NODES,1);
            end
            
            % Some housekeeping.
            obj.create_weight_addr();
            
        end
        
        function zero_bias(obj)
            obj.bias_vector   = zeros(obj.NR_OF_NODES,1);
        end
        
        function zero_weights(obj)
            obj.weight_matrix = zeros(obj.NR_OF_NODES, obj.NR_OF_NODES);
        end
        
        function output = run(obj, inputs)
           
            % Implementation to ensure hidden layers are not directly
            % accessible by exceeding input addresses by the user:
            for i = 1:obj.NR_OF_INPUTS
                obj.signal_field(i) = inputs(i);
            end
           
            for nodeNr = obj.NR_OF_INPUTS+1:obj.NR_OF_NODES   
                node_output = obj.fire_func(obj.signal_field, obj.weight_matrix(:,nodeNr), obj.bias_vector(nodeNr));
                obj.signal_field(nodeNr) = node_output;
            end
            
            output = obj.signal_field(obj.OUTP_ADDR)';
            
        end
        
        function clear_signals(obj)
            obj.signal_field  = zeros(obj.NR_OF_NODES,1);
        end
        
        function total_cost = get_cost(obj,inputs, exp_output)
           
            % Eliminate the effect of memory in the cost function.
            obj.clear_signals();
            
            [nSets,~] = size(inputs);
            
            total_cost = 0.0;
            
            for setNr = 1:nSets
                out = obj.run(inputs(setNr,:));
                C = 0.0;
                for i = 1:obj.NR_OF_OUTPUTS
                    %C = C + 0.5*(exp_output(setNr,i) - out(1,i))^2;
                    C = C + (exp_output(setNr,i) - out(1,i))^2;
                    %C = C + abs(exp_output(setNr,i) - out(i)); 
                end
                total_cost = total_cost + C;
            end
            
            % 'normalize' the cost by data size.
            total_cost = total_cost/nSets;
            
        end
        
        function [dCdw,dCdb,C_ref] = differentiate(obj, inputs, exp_output)
           
            dCdw = zeros(obj.NR_OF_NODES);
            dCdb = zeros(obj.NR_OF_NODES,1);
            
            C_ref = obj.get_cost(inputs, exp_output);
            
            for col = obj.NR_OF_INPUTS+1:obj.NR_OF_NODES
                for row = 1:col-1
                    if obj.RETAIN_CONNECTIONS == 1 && obj.weight_matrix(row,col) == 0
                        dw = 0;
                    else
                        if obj.ABSOLUTE == 1
                            dw = obj.DELTA_ABSOLUTE;
                        else
                            dw = abs(obj.weight_matrix(row,col))*obj.DELTA_RELATIVE;
                        end
                    end
                    
                    obj.weight_matrix(row,col) = obj.weight_matrix(row,col) + dw;
                    
                    C = obj.get_cost(inputs, exp_output);
                    
                    dC = C-C_ref;
                    
                    if dw == 0
                        dCdw(row,col) = 0.0;
                    else
                        dCdw(row,col) = dC/dw;
                    end
                    
                    obj.weight_matrix(row,col) = obj.weight_matrix(row,col) - dw;
                end
            end
            
            %bias_storage = obj.bias_vector;
            if obj.BIAS_OFF == 0
                for bNr = 1:obj.NR_OF_NODES
                        if obj.ABSOLUTE == 1
                            db = obj.DELTA_ABSOLUTE;
                        else
                            db = abs(obj.bias_vector(bNr))*obj.DELTA_RELATIVE;
                        end

                        obj.bias_vector(bNr) = obj.bias_vector(bNr) + db;

                        C = obj.get_cost(inputs, exp_output);

                        dC = C-C_ref;

                        if db == 0
                            dCdb(bNr) = 0.0;
                        else
                            dCdb(bNr) = dC/db;
                        end

                        obj.bias_vector(bNr) = obj.bias_vector(bNr) - db;
                end
            end
        end
        
        function C = learn(obj, inputs, exp_output)
           
            [dCdw,dCdb,C] = differentiate(obj, inputs, exp_output);
            
            compensation_w = 1;
            compensation_b = 1;
            
            if obj.DATA_SIZE_LEARNING_RATE_COMPENSATION == 1 % Maybe look at normalizing cost eg. /N
                % If the input training data set is large, the dCdw and
                % dCdb matrices will have very large numbers in them
                % because the cost scales up with the amount of data. This
                % causes the correction to the weight matrix to be large in
                % magnitude, causing the weights to become very large
                % themselves, causing in turn a saturatig effect in the
                % firing functions, subsequently stagnating further
                % learning. Here we dynamically adjust the magnitude of the
                % derivative matrices based on the largest correction to
                % the weights allowed by the correction limit parameter
                
                % Get maximum element size in dCdw and dCdb;
                max_dCdw = max(max(abs(dCdw)));
                max_dCdb = max(abs(dCdb));
                
                % Calculate scaling factor to account for the numerical
                % differentiation step size being exceeded (possibly)
                if max_dCdw ~= 0
                    if max_dCdw > obj.DELTA_ABSOLUTE
                        compensation_w = obj.DELTA_ABSOLUTE/max_dCdw;
                        %disp('limited');
                    end
                end
                
                if max_dCdb ~= 0
                    if max_dCdb > obj.DELTA_ABSOLUTE
                        compensation_b = obj.DELTA_ABSOLUTE/max_dCdb;
                    end
                end
            end
                 
            if obj.MOMENTUM_ON
                dCdw = dCdw + obj.momentum_lambda*obj.mem.dCdw_prev;
                dCdb = dCdb + obj.momentum_lambda*obj.mem.dCdb_prev;
            end
            
            obj.weight_matrix = obj.weight_matrix - obj.LEARNING_RATE*compensation_w*dCdw;
            obj.bias_vector   = obj.bias_vector   - obj.LEARNING_RATE*compensation_b*dCdb;
           
            obj.mem.dCdw_prev = dCdw;
            obj.mem.dCdb_prev = dCdb;
        end
        
        function [total_costs, hWeights, hBias] = study(obj, inputs, outputs, repetition_limit)
            
            total_costs = zeros(repetition_limit,1);
            % Reset cost criterium
            obj.COST_CRITERIUM_REACHED = 0;
            % [nw,mw] = size(obj.weight_matrix);
            % nb = length(obj.bias_vector);
            
            %[nr_of_samples,~] = size(outputs);
            obj.cost_tolerance = obj.OUTPUT_ERROR_TOLERANCE*obj.NR_OF_OUTPUTS; %*nr_of_samples;
                        
            % hWeights = zeros(nw,mw,repetition_limit);
            % hBias    = zeros(repetition_limit,nb);

            hWeights = cell(repetition_limit,1);
            hBias = cell(repetition_limit,1);
            
            if obj.SUPERVISOR_ON
                obj.initialize_supervisor(repetition_limit, obj.sup_n_filter, obj.cdl);
            end
            
            
            for repNr = 1:repetition_limit
               
                hWeights{repNr} = obj.weight_matrix;
                hBias{repNr} = obj.bias_vector';
                
                if obj.VERBOSE_LEARNING == 1
                    disp(['Learning repetition: ',num2str(repNr), ' ',datestr(now)]);
                end
                
                total_costs(repNr,1) = obj.learn(inputs, outputs);
                
                if obj.SUPERVISOR_ON
                    obj.supervise( repNr , total_costs(repNr,1) );
                    if obj.COST_CRITERIUM_REACHED
                        disp('Cost criterium reached. Stopping study cycle.');
                        break
                    end
                end
            end
            
        end
        
        function [cost_evolution, hWeights, hBias] = train(obj, input_set, output_set, repetitions, TRACK_COST)
           
            % Random training sequence.
            
            % Get size of training set. Does not do size checking here.
            % Input and output training data is expected to be in column
            % format, where a row represents a input and matching output
            % array.
            [n_inputs,~] = size(input_set);
            
            index_fraction = 1/n_inputs;
            
            cost_evolution = zeros(repetitions,1);
            
            % Pre allocate the weight and bias history
            [nw,mw] = size(obj.weight_matrix);
            nb = length(obj.bias_vector);
            hWeights = zeros(nw,mw,repetitions);
            hBias    = zeros(repetitions,nb);
            
            
            for repNr = 1:repetitions
               
                hWeights(:,:,repNr) = obj.weight_matrix;
                hBias(repNr,:) = obj.bias_vector';
                
                % Get random index of the training set to avoid biasing
                % towards a solution.
                
                index = ceil(rand/index_fraction);
                
                % Learn the randomly selected training data point.
                obj.learn(input_set(index,:), output_set(index,:));
                
                if TRACK_COST == 1 
                    cost_evolution(repNr) = obj.get_cost(input_set, output_set);
                end
                
            end

        end
        
        function add_node_in_layer(obj, hidden_layer)
           
            % This function adds a node to the selected layer with random
            % weights and bias.
            
            index = obj.NR_OF_INPUTS + obj.NR_OF_HIDDEN(hidden_layer);
            
            % Store the four parts of the matrix as split by the row
            % and column of the new node to save existing weights.
            UL = obj.weight_matrix(1:index, 1:index);
            UR = obj.weight_matrix(1:index, index+1:obj.NR_OF_NODES);
            LL = obj.weight_matrix(index+1:obj.NR_OF_NODES, 1:index);
            LR = obj.weight_matrix(index+1:obj.NR_OF_NODES, index+1:obj.NR_OF_NODES);
            
            Ub = obj.bias_vector(1:index);
            Lb = obj.bias_vector(index+1:obj.NR_OF_NODES);
            
            obj.NR_OF_HIDDEN(hidden_layer) = obj.NR_OF_HIDDEN(hidden_layer) + 1;

            % Create new weigth matrix and bias vector.    
            obj.create_network(obj.NR_OF_INPUTS, obj.NR_OF_HIDDEN, obj.NR_OF_OUTPUTS);
            obj.populate_weights(obj.MEMORY, obj.ZERO_INIT_BIAS);
            
            % Overwrite the old existing weights.
            obj.weight_matrix(1:index, 1:index) = UL;
            obj.weight_matrix(1:index, index+2:obj.NR_OF_NODES) = UR;
            obj.weight_matrix(index+2:obj.NR_OF_NODES, 1:index) = LL;
            obj.weight_matrix(index+2:obj.NR_OF_NODES, index+2:obj.NR_OF_NODES) = LR;
            
            obj.bias_vector(1:index) = Ub;
            obj.bias_vector(index+2:obj.NR_OF_NODES) = Lb;
            
            obj.create_weight_addr();
            
%             if hidden_layer > length(obj.NR_OF_HIDDEN)
%                 % A layer is requested that does not exist:
%                 disp('ERROR: cannot add node to non-existent layer');
%             else
%                 % First, the index of the new node (-1) in the N2 matrix is
%                 % calulated.
%                 index = obj.NR_OF_INPUTS + obj.NR_OF_HIDDEN(hidden_layer);
%                 
%                 % Store the four parts of the matrix as split by the row
%                 % and column of the new node.
%                 UL = obj.weight_matrix(1:index, 1:index);
%                 UR = obj.weight_matrix(1:index, index+1:obj.NR_OF_NODES);
%                 LL = obj.weight_matrix(index+1:obj.NR_OF_NODES, 1:index);
%                 LR = obj.weight_matrix(index+1:obj.NR_OF_NODES, index+1:obj.NR_OF_NODES);
%                 
%                 col_insert_top = zeros(index,1);
%                 col_insert_bottom = zeros(obj.NR_OF_NODES - index, 1);
%                 row_insert = zeros(1, obj.NR_OF_NODES + 1);
%                 
%                 % Construct new weight and bias fields.
%                 obj.weight_matrix = [UL, col_insert_top, UR;
%                                      row_insert;
%                                      LL, col_insert_bottom, LR];
%                                  
%                 obj.bias_vector = [obj.bias_vector(1:index); 
%                                    randn();
%                                    obj.bias_vector(index+1:obj.NR_OF_NODES)];
%                 
%                 % Next, new random weights are added to the input and
%                 % output of the new node connections;
%                 
%                 
%                                  
%                                  
%                 
%                 % Update parameters:
%                 obj.NR_OF_NODES = obj.NR_OF_NODES + 1;
%                 obj.NR_OF_HIDDEN(hidden_layer) = obj.NR_OF_HIDDEN(hidden_layer) + 1;
%                 obj.OUTPUT_ADDR = obj.OUTPUT_ADDR + 1;
                
            
            
        end
        
        function activate_momentum(obj, MOM_BOOL, lambda)
            % Momentum (low pass filtering of weight updates) can be used
            % for online learning and faster gradient descent.
            
            if MOM_BOOL
                obj.MOMENTUM_ON = 1;
                obj.momentum_lambda = lambda;
                obj.mem.dCdw_prev = zeros(obj.NR_OF_NODES);
                obj.mem.dCdb_prev = zeros(obj.NR_OF_NODES,1);
            else
                obj.MOMENTUM_ON = 0;
            end
            
        end
        
        function initialize_supervisor(obj, nr_of_reps, n_filter, cdl)
            
           obj.supervisor.cost_history = zeros(nr_of_reps,1);
           obj.supervisor.learning_rate_history = zeros(nr_of_reps,1);
           obj.supervisor.n_filter = n_filter;
           obj.supervisor.relative_cost_derivative_limit = cdl;
           obj.supervisor.c0 = 0;
           obj.supervisor.c_filt = 0;
           obj.supervisor.c_filt_prev = 0;
            
        end
        
        function supervise(obj, repNr, cost_now)
           
            if repNr == 1
                obj.supervisor.c0 = cost_now;
                obj.supervisor.c_filt_prev = cost_now;
            end
            
            obj.supervisor.cost_history(repNr) = cost_now;
            obj.supervisor.learning_rate_history(repNr) = obj.LEARNING_RATE;
            
            if repNr >= obj.supervisor.n_filter
                
                obj.supervisor.c_filt = sum(obj.supervisor.cost_history(repNr-obj.supervisor.n_filter+1:repNr))/obj.supervisor.n_filter;
                relative_cost_1 = obj.supervisor.c_filt/obj.supervisor.c0;
                relative_cost_0 = obj.supervisor.c_filt_prev/obj.supervisor.c0;
                d_relative_cost = relative_cost_1 - relative_cost_0;
                
                if d_relative_cost > 0
                    % If cost is increasing it's probably
                    % because the step size of weight differentiation is too
                    % large.
                    
                    % If error is increased, reduce the learning rate;
                    obj.LEARNING_RATE = obj.LEARNING_RATE*0.75;
                    %disp(['Changed learning rate to ', num2str(obj.LEARNING_RATE), ' at step # ', num2str(repNr)]);
                    
                else
                    
                    % If relative cost is decreasing, increase the learning
                    % rate a bit. (bold driver).
                    obj.LEARNING_RATE = obj.LEARNING_RATE*1.02;
                
                end
                
                if abs(d_relative_cost) < obj.supervisor.relative_cost_derivative_limit
                    %disp(['Cost decrease has evened out at step # ',num2str(repNr)]);
                    % pruning or adding nodes here.
                    % See if adding a node in the first layer helps.
                    disp(['Cost decrease has evened out at step # ',num2str(repNr), '. Adding a node to the first layer']);
                    obj.add_node_in_layer(1);
                    % If node is added, momentum must be reinitialized.
                    obj.activate_momentum(obj.MOMENTUM_ON, obj.momentum_lambda);
                end
                
                obj.supervisor.c_filt_prev = obj.supervisor.c_filt;
            end
            
            if cost_now < obj.cost_tolerance
                obj.COST_CRITERIUM_REACHED = 1;
            end
        end
        
        function use_sigmoid(obj)
           
            obj.fire_func = @obj.sigmoid;
            
        end
        
        function use_tanh(obj)
           
            obj.fire_func = @obj.tanhyp;
            
        end
        
        function use_sine(obj)
            
            obj.fire_func = @obj.sine;
            
        end
        
        function use_linear(obj)
            
            obj.fire_func = @obj.linear;
            
        end
        
    end
        
    methods % UKF methods
        
        function init_UKF(obj, P0, Q, R)
            
            obj.UKFcallNr = 1;
            [n_weights,~] = size(obj.WEIGHT_ADDR);
            
            if obj.BIAS_OFF
                obj.P0 = eye(n_weights)*P0;
                obj.P = obj.P0;
                obj.R  = eye(obj.NR_OF_OUTPUTS)*R;
                obj.Q  = eye(n_weights)*Q;
            else
                obj.P0 = eye(n_weights+obj.NR_OF_NODES)*P0;
                obj.P = obj.P0;
                obj.R  = eye(obj.NR_OF_OUTPUTS)*R;
                obj.Q  = eye(n_weights+obj.NR_OF_NODES)*Q;
            end
                
            
            if obj.BIAS_OFF
                obj.n_states = n_weights;
            else
                obj.n_states = n_weights + obj.NR_OF_NODES;
            end
            
            obj.n_measurements = obj.NR_OF_OUTPUTS;
            
            obj.setUKFConstants(1e-3, 0, 2);
            obj.nr_of_copoints = obj.n_states*2 + 1;
            % Set P matrix to initial value
            obj.state_prev = param2vec(obj);
            
            obj.INITIALIZED = 1;
            
        end 
        
        function setUKFConstants(obj,alpha, ki, beta)
            obj.alpha       = alpha;
            obj.ki          = ki;
            obj.beta        = beta;
            obj.lambda      = alpha^2*(obj.n_states+ki)-obj.n_states;
            obj.c           = obj.n_states+obj.lambda;
            obj.sqrt_c      = sqrt(obj.c);
            obj.W_mean      = [obj.lambda/obj.c, ones(1,2*obj.n_states)/(2*obj.c)];
            obj.W_cov       = [obj.lambda/obj.c + (1-alpha^2+beta), ones(1,2*obj.n_states)/(2*obj.c)];
        end
        
        function update_UKF(obj, input, exp_output)
        
            if obj.UKFcallNr == 1 % First call to this function
                
                % Because its the first call: return initial estimates (as
                % per init function).
                state_est = obj.state_prev;

            else

                % Construct observation vector.
                observation = exp_output';
                
                % Generate updated sigma points
                statevec = obj.state_prev;
                addmat = obj.sqrt_c*chol(obj.P)';
                Y = statevec(:,ones(1,obj.n_states));
                obj.X = [statevec, Y+addmat, Y-addmat];
                
                % Perform unscented transforms on the process.
                mean_p            = zeros(obj.n_states,1);
                samplingpoints_p  = zeros(obj.n_states,obj.nr_of_copoints);
                for j = 1:obj.nr_of_copoints
                    % Unity propagation
                    samplingpoints_p(:,j) = obj.X(:,j);
                    mean_p = mean_p + obj.W_mean(j)*samplingpoints_p(:,j);   
                end
                deviations_p = samplingpoints_p - mean_p(:,ones(1,obj.nr_of_copoints));
                Pp = deviations_p*diag(obj.W_cov)*deviations_p' + obj.Q; 
                
                % Perform unscented transforms on the measurements;
                mean_m = zeros(obj.n_measurements,1);
                samplingpoints_m = zeros(obj.n_measurements, obj.nr_of_copoints);
                for  j = 1:obj.nr_of_copoints
                    obj.implement_state(samplingpoints_p(:,j));
                    samplingpoints_m(:,j) = obj.run(input);
                    mean_m = mean_m + obj.W_mean(j)*samplingpoints_m(:,j);
                end
                deviations_m = samplingpoints_m - mean_m(:,ones(1,obj.nr_of_copoints));
                Pm = deviations_m*diag(obj.W_cov)*deviations_m' + obj.R;
                
                % Compute Kalman gain
                P_cross = deviations_p*diag(obj.W_cov)*deviations_m';
                obj.K = P_cross/Pm;
                
                % Perform state update
                state_est = mean_p + obj.K*(observation - mean_m);
                obj.P = Pp - obj.K*P_cross';
                
                if obj.Q_MATRIX_ADAPTATION
                    Dx = state_est - mean_p;
                    Qstar = Dx*Dx'; 
                    obj.Q = (1-obj.LAMB_Q)*obj.Q + obj.LAMB_Q*(Qstar - obj.Q);
                end
                
                % Store state values
                obj.state_prev = state_est;
                
            end

            obj.implement_state(state_est);
            
            % The call number is updated:
            obj.UKFcallNr = obj.UKFcallNr+1; 
            
        end
        
        function [state_vec] = param2vec(obj)
            [n,~] = size(obj.WEIGHT_ADDR);
            state_vec = zeros(n,1);
            for i = 1:n
                state_vec(i,1) = obj.weight_matrix(obj.WEIGHT_ADDR(i,1), obj.WEIGHT_ADDR(i,2));
            end
            if ~obj.BIAS_OFF
                state_vec = [state_vec; obj.bias_vector];
            end
        end
        
        function implement_state(obj, state_vec)
            [n,~] = size(obj.WEIGHT_ADDR);
            for i = 1:n
                obj.weight_matrix(obj.WEIGHT_ADDR(i,1), obj.WEIGHT_ADDR(i,2)) = state_vec(i);
            end
            if ~obj.BIAS_OFF
                obj.bias_vector = state_vec(n+1:n+obj.NR_OF_NODES,1); 
            end
        end
        
        function [cost_hist] = LearnKalman(obj, inputs, outputs, max_cycles, COST_BOOL)
           
            cost_hist = zeros(max_cycles,1);
            
            % Start Kalman filter
            if obj.INITIALIZED

                % Obtain size of data to learn.
                [n_sets,~] = size(inputs);

                % Start learning
                for cycleNr = 1:max_cycles
                    rand_index = 1 + round(rand*(n_sets-1));
                    obj.update_UKF(inputs(rand_index,:), outputs(rand_index,:));
                    if COST_BOOL
                        cost_hist(cycleNr) = obj.get_cost(inputs, outputs);
                    end
                    if obj.VERBOSE_LEARNING
                        disp(['Cycle nr: ', num2str(cycleNr)]);
                    end
                end
            else
                disp('Error: Kalman filter is not initialized');
            end
        end
        
    end
        
    methods %(Static)
        
        function out = sigmoid(obj, input_vec, weight_vec, bias)
           
            sze = size(input_vec);
            sum = 0;
            for i = 1:sze
                sum = sum + input_vec(i)*weight_vec(i);
            end
            z = sum + bias;
            out = 1/(1+exp(-z));
            
        end
        
        function out = tanhyp(obj, input_vec, weight_vec, bias)
        
            sze = size(input_vec);
            sum = 0;
            for i = 1:sze
                sum = sum + input_vec(i)*weight_vec(i);
            end
            z = (sum + bias)*obj.tanh_flattening;
            out = tanh(z);
            
        end
        
        function out = sine(obj, input_vec, weight_vec, bias)
            % Experimental firing function to see if system identification
            % and adaptation in relation to periodic systems (with periodic
            % training data) will work better. This to try and avoid the
            % problems arising with a fixed training domain causing the
            % output to be 'saturated' or overfitted outside of the domain which was
            % trained in.
            
            sze = size(input_vec);
            sum = 0;
            for i = 1:sze
                sum = sum + input_vec(i)*weight_vec(i);
            end
            z = (sum + bias);
            out = sin(z);
            
        end
        
        function out = linear(obj, input_vec, weight_vec, bias)
           
            sze = size(input_vec);
            sum = 0;
            for i = 1:sze
                sum = sum + input_vec(i)*weight_vec(i);
            end
            out = sum + bias;
            
        end
        
    end
    
end