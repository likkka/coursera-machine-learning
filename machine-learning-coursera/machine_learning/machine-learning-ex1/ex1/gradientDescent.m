function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %fprintf("J_history[%d] = %f \n", iter, J_history(iter))
    %fprintf("sum(x1):%f,sum(x2):%f", sum(X(1)),sum(X(2)))
    

    if iter > 1
        if J_history(iter) > J_history(iter-1)
            break;
        end 
    end

    h_theta = y-X*theta;
    theta(1) = theta(1) + alpha * sum((h_theta).* X(:,1))/m;
    theta(2) = theta(2) + alpha * sum((h_theta).* X(:,2))/m;
    %fprintf("theta: %f , %f \n" , theta(1), theta(2));

end

end
