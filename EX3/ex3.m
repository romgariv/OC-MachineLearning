y = load('ex3y.dat');
x = load('ex3x.dat');

m = rows(x);
x = [ones(m, 1), x];
x_unscaled = x;
% Preprocessing data
% Rescale all features to be
% On the same scale
sigma = std(x);
mu = mean(x);

x(:, 2) = (x(:,2) - mu(2))./ sigma(2);
x(:, 3) = (x(:,3) - mu(3))./ sigma(3);

% Gradient descent
for alpha = [1, 0.5, 0.1, 0.05, 0.01]
	num_iters = 50;
	theta = zeros(size(x(1,:)))';
	J = zeros(num_iters,1);
	for i=1:num_iters
		J(i) = 1/(2. * m) .* (x * theta - y)' * (x * theta - y);
		theta = theta - alpha .* 1./(2 .* m) .* ( x' * (x * theta - y));
	end
	% Plot results of J
	figure;
	plot(0:49, J(1:50), '-')
	xlabel('Number of iterations')
	ylabel('Cost J')	
end

% Normal equations
x = x_unscaled;
theta = inv(x' * x) * (x' * y);
disp(theta)
% Predict price of 1650 sqft house + 3 bedrooms
disp([1, 1650, 3] * theta)
