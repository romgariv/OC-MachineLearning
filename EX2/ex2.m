x = load('ex2x.dat');
y = load('ex2y.dat');

figure
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')

m = length(y)
x = [ones(m, 1), x];

alpha = 0.07;

theta = [0,0]'

% Problem 1:
% Implement 1 iteration of batch gradient descent
score = x * theta;
theta = theta - alpha .* 1/ m  .* ( x' * (score - y) );
disp(theta);

% Problem 2:
% Run batch gradient descent 
% until theta converges (1500 iterations)

i = 1;
while (i < 1500)
	score = x * theta;
	theta = theta - alpha .* 1/ m  .* ( x' * (score - y) );
	i++;
endwhile

disp(theta)

% Plot results
figure
plot(x(:,2), y, 'o', x(:,2), x * theta, '-')
ylabel('Height in meters')
xlabel('Age in years')
