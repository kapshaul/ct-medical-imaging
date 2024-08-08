function crlb = CRLB(A, X)
    % CRLB for each parameter
    crlb_par = diag(inv(A'*diag(1./(A*X))*A));
    % Construct CRLB
    crlb = mean(crlb_par);
end
