function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

disp('***** theta ******')
disp(size(theta))

% size of X     = 5000 x 401
% size of theta = 401 x 1
hx = sigmoid(X * theta);
disp(size(hx))
tempTheta = theta;
tempTheta(1) = 0;

temphx = y.*log(hx) + (1 - y).*log(1 - hx);
disp(size(temphx))
J = (-1/m)*sum(temphx) + (lambda/(2*m))*sum(tempTheta.^2);
disp(size(J))
% disp('******* J *******');
% disp(J);

temp = hx;
error = temp - y;
grad = (1/m)*(X'*error) + (lambda/m)*tempTheta;

grad = grad(:);

% disp('******* grad *******');
% disp(grad);

end
