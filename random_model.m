function A = random_model(rows, cols)
    % Initialize the matrix with zeros
    A = zeros(rows, cols);
    
    % Loop through each row
    for i = 1:rows
        % Randomly select 3 columns to place 1s
        idx = randperm(cols, 3);  
        A(i, idx) = 1;
    end
end