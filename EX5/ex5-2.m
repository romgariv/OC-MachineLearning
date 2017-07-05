% Regularized Logistic Regression
x = load('ex5Logx.dat');
y = load('ex5Logy.dat');
source('map_feature.m');

pos = find(y == 1);
neg = find(y == 0);

plot(x(pos, 1), x(pos, 2), '+', x(neg, 1), x(neg, 2), 'o')

x =  map_feature(x(:,1), x(:,2));

function h = sigmoid(x, theta)
        h = 1 ./ (1 + exp(-x  * theta));
endfunction

function J = LLReg(x,y,theta, lambda)
        J = 1/rows(y) .* (-y' * log(sigmoid(x, theta)) - (1-y)' * log(1 - sigmoid(x, theta))) + lambda/(2 * rows(y)) .* theta(2:end,:)' * theta(2:end,:);
endfunction

function dJ = GradientReg(x, y, theta, lambda)
        dJ = 1/rows(y) .* (x' * (sigmoid(x, theta) - y) + lambda * [0 ; theta(2:end,:)]);
endfunction

function H = HessianReg(x,y,theta,lambda)
        H = 1 ./ rows(y) .* ((x' * diag(sigmoid(x, theta) .* (1 - sigmoid(x, theta))) * x) + lambda .* diag([0, ones(size(x(1,2:end)))]));
endfunction

function NT = NMReg(x,y, lambda, tolerance)
        theta = zeros(size(x(1,:)))';
        %disp(theta)
        LL_old = 100000000;
        LL_new = LLReg(x,y,theta, lambda);
        while ((abs(LL_old - LL_new) > tolerance))
	LL_old = LL_new;
        theta = theta - HessianReg(x,y,theta,lambda)\GradientReg(x,y,theta,lambda);
	LL_new = LLReg(x,y,theta, lambda);
	disp(LL_new);
     endwhile
        NT = theta;
endfunction

disp(norm(NMReg(x,y,0,.0005)));
disp(norm(NMReg(x,y,1,.0005)));
disp(norm(NMReg(x,y,10,.0005)));
