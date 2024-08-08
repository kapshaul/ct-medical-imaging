%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Medical Imaging Shot Noise Removal 
% - Poisson noise removal via Expectation Maximization
% - Yong-Hwan Lee and Tony Storey
%
% Given:
%   P(Lambda) = (exp(A_ij*theta_i) * (A_ij*theta_i)^Y_ij)/(Y_ij!)
%   y_n = Poisson((A*th)n)   n = 1,2,3...n
%   use EM algorithm    
%   
%
%   
%
%           3x3 Cross section of Voxel model
%           each pixel modeles absorbtion coef
%                    
%    |        *--------------------*       |  
%    | y3---\ |      |      |      |       | 
%    |   ---/ |  p1  |  p2  |  p3  |       | 
%    -        *--------------------*       - 
%    | y2---\ |      |      |      |       | 
%    |   ---/ |  p4  |  p5  |  p6  |       | 
%    -        *--------------------*       - 
%    | y1---\ |      |      |      |       | 
%    |   ---/ |  p7  |  p8  |  p9  |       | 
%    |        *--------------------*       |  
%
%
%             -------|------|-------
%                y9     y10    y11
%                ||     ||     ||
%                \/     \/     \/
%             *--------------------*          
%             |      |      |      |        
%             |  p1  |  p2  |  p3  |        
%             *--------------------*        
%             |      |      |      |        
%             |  p4  |  p5  |  p6  |        
%             *--------------------*        
%             |      |      |      |        
%             |  p7  |  p8  |  p9  |        
%             *--------------------*         
%   
%             -------|------|-------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear
clc

%% Initialize
gain = [0.1 1 5 10];                        % Gains 0.1 to 10
max_gain_stages = length(gain);
num_iter = 20;                              % Iteration Number
monte = 10000;                              % Montecarlo Number
n = 9;                                      % A matrix length n
m = 16;                                     % A matrix length m

%% Model Matrix A
% Pre-defined Model
A = [0 0 0 0 0 0 1 1 1;         
     0 0 0 1 1 1 0 0 0;
     1 1 1 0 0 0 0 0 0;
     0 0 0 0 0 0 1 0 0;
     0 0 0 1 0 0 0 1 0;
     1 0 0 0 1 0 0 0 1;
     0 1 0 0 0 1 0 0 0;
     0 0 1 0 0 0 0 0 0;
     1 0 0 1 0 0 1 0 0;
     0 1 0 0 1 0 0 1 0;
     0 0 1 0 0 1 0 0 1;
     1 0 0 0 0 0 0 0 0;
     0 1 0 1 0 0 0 0 0;
     0 0 1 0 1 0 1 0 0;
     0 0 0 0 0 1 0 1 0;
     0 0 0 0 0 0 0 0 1];

% Random Model
A = random_model(m, n);

% Display the model matrix
disp("A = ");
disp(A);

%% True data and Poisson noise for absorption coefs
% Create Noised Observation
base_value = unifrnd(0, 1000, [n, 1]);      % Base parameter vector
x = gain(1)*base_value;                     % True parameter vector
y = sum(poissrnd(A.*x'), 2);                % Observation vector

%% EM Algorithm
% Gain value sweep
for gain_val = 1:max_gain_stages
    
    MSE = 0;
    CRLB = 0;
    % Monte-Carlo Runs
    for k = 1:monte
        % Observation Update With New Noise
        x = gain(gain_val)*base_value;
        y = sum(poissrnd(A.*x'), 2);

        % Starting point from Method of Moments
        X = inv(A'*A)*A'*y;

        % Starting EM Algorithm N-iter
        for iter = 1:num_iter

            % Using Stirling's approximation to prevent inf values
            log_factorial_y = y.*log(y) - y;
            lambda = A*X;

            % Compute log likelihood 
            likelihood(k, iter) =  sum(-lambda + y.*log(lambda) - log_factorial_y);
            
            % Compute EM algorithm to get a new lambda
            X = EM_algorithm(A, y, X)';
            
            % Compute MSE
            error = x-X;
            MSE(k, iter) = mean(error.^2);

            % Re-calculate CRLB
            CRLB_par = diag(inv(A'*diag(1./(A*X))*A));
            CRLB(k, iter) = mean(CRLB_par);             % Construct CRLB

        end % End of EM Algorithm N-iter
        
    end % End of Monte_Carlo runs
   
    % Compute the mean values of MSE and CRLB with normalization by gain^2
    CRLB_monte_mean(gain_val) = mean(CRLB(:, num_iter)) / gain_val^2;
    MSE_monte_mean(gain_val) = mean(MSE(:, num_iter)) / gain_val^2;
 
end % End gain value sweep

%% Plots for Convergence, MSE, CRLB, and Log-likelihood
% Plot log-likelihood
figure
plot(mean(likelihood, 1), '-r')
title('Log Likelihood')
xlabel('EM Iteration')
ylabel('Log Likelihood') 

% Plot MSE
figure
plot(mean(MSE, 1), '-r')
title('MSE')
xlabel('EM Iteration') 
ylabel('MSE')

% Plot CRLB after "monte" monte carlo runs
figure
hold on
loglog(CRLB_monte_mean, 'b')
loglog(MSE_monte_mean, 'r')    
hold off
xlabel(['gain factor ', num2str(gain(1)), '\leq gain \leq', num2str(gain(max_gain_stages))])
ylabel('MSE & CRLB')
legend('CRLB', 'MSE', 'Location', 'northwest')
title('CRLB&MSE')
