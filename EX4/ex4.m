x = load('ex4x.dat');
y = load('ex4y.dat');
x = [ones(size(x(:,1))), x];

pos = find(y == 1);
neg = find(y == 0);

figure
plot(x(pos, 2), x(pos, 3), '+', x(neg,2), x(neg, 3), 'o');

% Newton's Method
function h = sigmoid(x, theta)
	h = 1 ./ (1 + exp(- x * theta ));
endfunction

function J = LL(x, y, theta)
	J = (1 / rows(y)) * (-y' * log(sigmoid(x, theta)) - (1 - y)' * log(1 - sigmoid(x,theta)));
endfunction

function dJ = grad(x, y, theta)
	 dJ = x' * (sigmoid(x,theta) - y);
endfunction

function H = hessian(x, y, theta)
	H = x' * diag(sigmoid(x, theta) .* (1 - sigmoid(x, theta))) * x;
endfunction

go = 1;
theta = zeros(size(x(1,:)'));
old_LL =  LL(x,y, theta);
new_LL = old_LL;
counter = 0;
while (go == 1)
	printf('Iteration: %d \n', counter);
        printf('Current Log-Likelihood: %d \n', new_LL);
	theta = theta - hessian(x,y,theta)\grad(x,y,theta);	
	new_LL = LL(x,y, theta);
	printf('Iteration: %d \n', counter);
	printf('New Log-Likelihood: %d \n', new_LL);

	if (abs(old_LL - new_LL) < 0.0001)
		go = 0;	
	else
		old_LL = new_LL;
		counter ++;
	endif
	
end

disp('Final Theta:');
disp(theta);
