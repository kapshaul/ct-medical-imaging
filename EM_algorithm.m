function X_new = EM_algorithm(A, y, X)    
    n = length(X);
    m = length(y);

    % Starting EM Algorithm
    for j = 1:n
        for i = 1:m
            den1 = A*X;
            % Compute the probability
            prob(i) = y(i)*A(i,j)/den1(i);
        end
        % E step to copute expectation
        expectation(j) = sum(prob);
        den2(j) = sum(A(:,j));
        % M step to maximize likelihood
        X_new(j) = X(j)/den2(j)*expectation(j);
    end
end