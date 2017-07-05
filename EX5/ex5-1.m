% Regularized Linear Regression
x = load('ex5Linx.dat');
y = load('ex5Liny.dat');

plot(x,y,'o')

x = [ones(size(x)), x, x .^2, x .^3, x .^4, x .^5];

theta = zeros(size(x(1,:)'));

function J = RegCost(x, y, theta, lambda) 
	J = 1 ./ (2 .* rows(y)) .* ((x * theta - y)' * (x * theta - y) + lambda .* theta(2:end,:)' * theta(2:end,:));
endfunction

function T = Normal(x, y, lambda)
	T = (x' * x + lambda .* diag([0, ones(size(x(1,2:end)))]')) \ (x' * y);
endfunction

disp([Normal(x,y,0), Normal(x,y,1), Normal(x,y,10)]);
