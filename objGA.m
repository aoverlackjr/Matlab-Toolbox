classdef objGA < handle
% Genetic algorithm object
% Generic structure for keeping the population and doing the cross over,
% mutation and selection of chromosomes.
% Evaluation of chromosome functions and assigning fitness is done outside
% of this object.
% How to use:
% 1 Start object with objGA()
% 2 Assign properties as required: gene length, chromosome length and rates,
%   etc.
% 3 Create a population with create_population();
% 4 Run (external) algorithm that interprets the individuals and assigns
%   fitness to them (obj.population.chromosomes and obj.population.fitness)
% 5 cycle the population (evolution)
% 6 Repeat step 4 and 5 until a successful chromosome is found
    
    properties % DEFAULT VALUES
        
        CROSSOVER_RATE = 0.7;               % The chance of cross-over occuring when two chromosomes are mated
        MUTATION_RATE = 0.001;              % The chance of an element in the chromosome to mutate when offspring is generated
        POP_SIZE = 100;                     % The number of chromosomes in a population. Must be an even number!
        CHROMO_LENGTH = 300;                % The length of a chromosome. This is a multiple of the gene length
        GENE_LENGTH = 4;                    % The lenght of the genes. If FLOAT is on, this is one. If DIGITAL, this is the byte size.
        MAX_ALLOWABLE_GENERATIONS = 400;    % The allowable number generations for a evolution simulation run. (stop criterion) 
        MAX_MUTATION_PERTURBATION = 0.3;    % If FLOAT mode is engaged, this is the maximum perturbation of a mutation.
        ELITISM = 1;                        % If this mode is on, the two fittests individuals of a generation go over to the next (unmodified)
        
        population;                         % The storage of the active population. Structure: population.chromosomes & population.fitness
        history;                            % The storage of the previous populations.
        mode;                               % The active mode: DIGITAL or FLOAT, depending in the use case, and on how phenotypes are encoded.
        
    end
    
    properties (GetAccess = 'private', SetAccess = 'private')
       
        temp_population;                    % Temporary storage of population when cycling between generations is calculated
        gen_nr;                             % The generation number
        mode_list = {'DIGITAL', 'FLOAT'}    % The possible modes: FLOAT: gene is a floating point number, DIGITAL: gene is a byte
        
    end
    
    methods
        
        function obj = objGA()
            % Concstructor. After constructor is called, the user should
            % configure the GA, by indicating the chromosome length (as a
            % minimum).
            obj.population.chromosomes = [];
            obj.population.fitness     = [];
            obj.set_mode('DIGITAL');
        end
        
        function set_crossover_rate(obj, pc)
           
            % The chance of cross over to occur when two chromosomes are
            % mated.
            if ((pc > 0.0) && (pc <= 1.0))
                obj.CROSSOVER_RATE = pc;
            else
                disp('Cross-over rate must be between 0 and 1. Value not changed.')
            end
            
        end
        
        function set_mutation_rate(obj, pm)
           
            % The chance of mutation to take place when offspring is
            % produced.
            if ((pm > 0.0) && (pm <= 1.0))
                obj.MUTATION_RATE = pm;
            else
                disp('Mutation rate must be between 0 and 1. Value not changed.')
            end
            
        end
        
        function set_population_size(obj, pop_size)
           
            % The number of individuals (chromosomes) in a population. Must
            % be an even number.
            if ((pop_size > 0) && (mod(pop_size,2) == 0))
                obj.POP_SIZE = pop_size;
            else
                disp('Population size must be greater than zero and an even number.');
            end
            
        end
        
        function set_chromosome_length(obj, chromo_length)
            % The length of the chromosome. This must be equal to the
            % number of floats (if floating point mode 'FLOAT'is used) in
            % the chromosomes, or on the number of bytes x the number of
            % bits per byte. The number of bits per byte is then indicated
            % in the GENE_LENGTH property
            if(chromo_length > 0)
                if(strcmp(obj.mode,'DIGITAL'))
                    if(mod(chromo_length,obj.GENE_LENGTH) == 0)
                        obj.CHROMO_LENGTH = chromo_length;
                    else
                        disp('Warning: chromosome length not a multiple of current gene length!.')
                        % Set the chromosome length anyway to avoid
                        % configuration deadlock.
                        obj.CHROMO_LENGTH = chromo_length;
                    end
                else
                    obj.CHROMO_LENGTH = chromo_length;
                end
            else
                disp('Chromosome length must be greater than zero.');
            end
        end
        
        function set_gene_length(obj, gene_length)
            % The size of the gene. If FLOAT mode is on, this is set to 1.
            % If DIGITAL mode is on, this is the size of the byte.
            if(gene_length > 0)
                if(strcmp(obj.mode,'DIGITAL'));
                   if(mod(obj.CHROMO_LENGTH,gene_length) == 0)
                       obj.GENE_LENGTH = gene_length;
                   else
                       disp('Warning: The requested gene size is not compatible with the chromosome size in digital mode.');
                       obj.GENE_LENGTH = gene_length;
                   end
                else
                    obj.GENE_LENGTH = gene_length;
                end
            else
                disp('Gene length must be greater than zero.');
            end
        end
        
        function set_max_generations(obj, max_gen)
           
            % The maximum number of generations. The use of this number can
            % be used 'outside' the object, for instance in the evolution
            % loop. This number does not directly inhibit functionality in
            % the object itself.
            if(max_gen > 0)
                obj.MAX_ALLOWABLE_GENERATIONS = max_gen;
            else
                disp('Maximum number of generations must be greate than zero.');
            end
            
        end
        
        function set_max_perturbation(obj, max_perturb)
            
            % The maximum perturbation of a floating point number (gene) in the
            % chromosome during mutation. This number is not bound to
            % anything. Only used if FLOAT mode is on.
            
            obj.MAX_MUTATION_PERTURBATION = max_perturb;
            
        end
        
        function set_elitism(obj, on_off_string)
           
            % The switch that enables elitism. When elitism is activated, the two fittest 
            % individuals of a population go over (unchanged) to the next generation
            % when the cycle is performed.
            switch on_off_string
                case 'on'
                    obj.ELITISM = 1;
                case 'off'
                    obj.ELITISM = 0;
            end
            
        end
        
        function create_population(obj)
           
            % This function must be called before the GA can be used. It
            % creates a first (random) population and pre-allocates the
            % fitness array.
            
            % First generation
            obj.gen_nr = 1;
            
            % Pre-allocate the chromosome population according to the 
            % chromosome and population size, which should be indicated 
            % before this function is called.
            obj.population.chromosomes = zeros(obj.POP_SIZE, obj.CHROMO_LENGTH);
            obj.population.fitness     = zeros(obj.POP_SIZE, 1);
            
            switch obj.mode
                
                case 'DIGITAL'
                    
                    % In case DIGITAL mode is active, load population with random bit chromosomes. 
                    for chrNr = 1:obj.POP_SIZE
                        obj.population.chromosomes(chrNr,:) = obj.get_random_bits(obj.CHROMO_LENGTH);
                    end
                    
                case 'FLOAT'
               
                    % In case FLOAT mode is active, load population with random floatin point numbers between 0 and 1. 
                    for chrNr = 1:obj.POP_SIZE
                        obj.population.chromosomes(chrNr,:) = obj.get_random_floats(obj.CHROMO_LENGTH);
                    end
                    
            end
            
            % Pre-allocate history
            obj.history = cell(obj.MAX_ALLOWABLE_GENERATIONS,1);
            for genNr = 1:obj.MAX_ALLOWABLE_GENERATIONS
                obj.history{genNr}.chromosomes = zeros(obj.POP_SIZE, obj.CHROMO_LENGTH);
                obj.history{genNr}.fitness     = zeros(obj.POP_SIZE, 1);
            end
            
            % Load first population in history.
            obj.history{1,1} = obj.population;
        end
        
        function chromo = mutate_chromosome(obj, chromo_in)
            % this function runs the mutation of a chromosome as per the
            % mode and mutation probability.
            
            switch obj.mode
                
                case 'DIGITAL'
            
                    % Run along all bits and if the probability is met, the
                    % bit is flipped.
                    nBits = length(chromo_in);
                    chromo = chromo_in;
                    for bitNr = 1:nBits
                        if (rand < obj.MUTATION_RATE)
                            if chromo(bitNr) == 1
                                chromo(bitNr) = 0;
                            else
                                chromo(bitNr) = 1;
                            end
                        end
                    end
                    
                case 'FLOAT'
                    
                    % Run along all floats in the chromosome and add a
                    % random mutation of (-1 < r < 1)*MAX_MUTATION_PERTURBATION
                    nFloats = length(chromo_in);
                    chromo = chromo_in;
                    for floatNr = 1:nFloats
                        if (rand < obj.MUTATION_RATE)
                            chromo(floatNr) = chromo(floatNr) + (rand-rand)*obj.MAX_MUTATION_PERTURBATION;
                        end
                    end
                    
            end
       
        end
        
        function [offspring1, offspring2] = cross_over(obj, chromo1, chromo2)
           
            % The cross-over function takes two 'parents' and at a random
            % intermediate point crosses and exchanges parts of the
            % chromosome. At a later stage, dual or mutliple point
            % cross-over may be implemented in this function as a setting.
            
            offspring1 = chromo1;
            offspring2 = chromo2;
            
            if(rand < obj.CROSSOVER_RATE)
               
                crossover_index = floor(rand*length(chromo1));
                
                offspring1 = [chromo1(1,1:crossover_index),chromo2(1,crossover_index+1:length(chromo2))];
                offspring2 = [chromo2(1,1:crossover_index),chromo1(1,crossover_index+1:length(chromo1))];
                
            end
            
        end
        
        function chromo = roulette(obj)
            % The roulette function chooses an individual from the
            % population on the basis of probability and the fitness of
            % individuals.
            total_fitness = obj.get_total_fitness();
            slider = rand*total_fitness;
            cumulative_fitness = 0.0;
            for chrNr = 1:obj.POP_SIZE
                cumulative_fitness = cumulative_fitness + obj.population.fitness(chrNr);
                if cumulative_fitness >= slider
                    chromo = obj.population.chromosomes(chrNr,:);
                    break;
                end
            end
        end
        
        function total_fitness = get_total_fitness(obj)
            % Helper function to obtain the total fitness (sum) of the
            % active population.
            total_fitness = sum(obj.population.fitness);
        end
        
        function assign_fitness(obj, indivNr, fitness)
            obj.population.fitness(indivNr) = fitness;
        end
        
        function cycle_population(obj)
           
            % This is the cycle function that cycles the population from
            % one generation to the next. This function should be called
            % from outside the object, once the run of one generation is
            % finished and the fitnes is assigned to all individuals.
            
            % First, put the population in history.
            obj.history{obj.gen_nr,1} = obj.population;
            
            % Make a temporary population for cross-over, mutations etc.
            obj.temp_population.chromosomes = zeros(obj.POP_SIZE, obj.CHROMO_LENGTH); 
            obj.temp_population.fitness     = zeros(obj.POP_SIZE, 1);
            
            chrNr = 0;
            
            % Take pairs of individuals from the population two by two and
            % mate them. The offspring is placed in the temporary
            % population.
            while (chrNr <= obj.POP_SIZE)
            
                chromo1 = roulette(obj);
                chromo2 = roulette(obj);
                
                [offspring1, offspring2] = obj.mate(chromo1, chromo2);
                
                chrNr = chrNr+1;
                obj.temp_population.chromosomes(chrNr,:) = offspring1;
                chrNr = chrNr+1;
                obj.temp_population.chromosomes(chrNr,:) = offspring2;
                
            end
            
            % If elitism is active, two strongest individuals are allowed
            % to pass on to the new generation unchanged.
            if obj.ELITISM == 1
               
                [elite1, elite2] = find_fittest_pair_in_population(obj);
                
                obj.temp_population.chromosomes(1,:) = elite1;
                obj.temp_population.chromosomes(2,:) = elite2;
                
            end
            
            % Copy temporary population into the new active one.
            obj.population = obj.temp_population;
            obj.gen_nr = obj.gen_nr + 1;
        end
        
        function [offspring1, offspring2] = mate(obj, chromo1, chromo2)
           
            % The mating function, sequence of cross-over and mutation.
            [offspring1, offspring2] = cross_over(obj, chromo1, chromo2);
            offspring1 = mutate_chromosome(obj, offspring1);
            offspring2 = mutate_chromosome(obj, offspring2);
            
        end
        
        function [chromo, fitness, generation, individual] = find_fittest_chromo_in_history(obj)
           % Function finds the fittest chromosome (individual) in the
           % entire history, and returns the generation, index (individual
           % number) and the chromosome of the individual.
           
            generation = 0;
            individual = 0;
            max_fitness = -1;
            
            for genNr = 1:obj.MAX_ALLOWABLE_GENERATIONS;
               
                [maxfit,ind] = max(obj.history{genNr}.fitness);

                if maxfit > max_fitness
                    max_fitness = maxfit;
                    generation = genNr;
                    individual = ind;
                end
                
            end
            
            chromo  = obj.history{genNr}.chromosomes(individual,:);
            fitness = obj.history{genNr}.fitness(individual);
            
        end
        
        function [chromo1, chromo2] = find_fittest_pair_in_population(obj)
           
            % Helper function for the elitism selection that finds the two
            % best individuals.
            fitness = obj.population.fitness;
            [~,individual1] = max(fitness);
            fitness(individual1) = 0;
            [~,individual2] = max(fitness);
            
            chromo1 = obj.population.chromosomes(individual1,:);
            chromo2 = obj.population.chromosomes(individual2,:);
            
        end
        
        function [avFitness, maxFitness] = get_fitness_evolution(obj)
           
            % Helper function that gets two arrays; one is the evolution of
            % the average (per generation/population) fitness and the
            % maximum fitness.
            nGenerations = obj.MAX_ALLOWABLE_GENERATIONS;
            
            avFitness = zeros(nGenerations,1);
            maxFitness = zeros(nGenerations,1);
            
            for genNr = 1:nGenerations
                avFitness(genNr) = sum(obj.history{genNr}.fitness)/obj.POP_SIZE;
                maxFitness(genNr) = max(obj.history{genNr}.fitness);
                
            end
            
        end
        
        function set_mode(obj, mode_string)
        
            switch mode_string
                
                case 'FLOAT'
                    obj.mode = 'FLOAT';
                    obj.set_gene_length(1);
                case 'DIGITAL'
                    obj.mode = 'DIGITAL';
                    
            end
            
        end
        
    end
    
    methods (Static)
       
        function random_byte = get_random_bits(byte_size)
            random_byte = zeros(1,byte_size);
            for bitNr = 1:byte_size
                if (rand > 0.5)
                    random_byte(bitNr) = 1;
                end 
            end
        end
        
        function random_floats = get_random_floats(size)
           
            random_floats = zeros(1,size);
            for floatNr = 1:size
                    random_floats(floatNr) = rand-rand;
            end
            
        end
        
        function nr = binary2num(byte)
            nBits = length(byte);
            nr = 0;
            add_val = 1;
            for bitNr = 1:nBits %bitNr = nBits:-1:1
               if byte(bitNr) == 1
                   nr = nr + add_val;
               end
                add_val = add_val*2;
            end
        end
        
        function byte = num2binary(num,nBits)

            powOf2 = 2.^[0:nBits-1];

            %# do a tiny bit of error-checking
            if num > sum(powOf2)
               error('not enough bits to represent the data')
            end

            byte = zeros(1,nBits);

            ct = nBits;

            while num>0
                if num >= powOf2(ct)
                    num = num-powOf2(ct);
                    byte(ct) = 1;
                end
                ct = ct - 1;
            end
            
        end
        
    end
    
end