function LinearRegression()

clear;
clc;

% 有两种方法进行回归:
% 1. normal equations正则方程
% 2. 梯度下降
% 当特征维度低的时候,使用正则方程比较简单,清晰.同时因为要求逆,所以有奇异矩阵的风险.
% 特征维度高的时候,使用梯度下降更加效率.另外不存在求逆的风险.

%一维线性系统
if 0
	x = load('./data/ex2x.dat');
	y = load('./data/ex2y.dat');
	x(:,2) = 1;
	
	%% 1. 正则方程
	W1 = x \ y;	
	figure(1);
	clf;
	plot(x(:,1),y,'o');
	hold on;
	plot(x(:,1), x*W1);
	
	
	%% 2. 梯度下降
	% 目标函数最小,不知道为什么要除m
	% L = (XW-Y)^2 / (2*m)
	
	W2 = zeros(2,1);
	MAX_ITR = 1000;
	m = length(x);
	% 简单起见,采用固定步长
	lambda = 0.07;
	
	for it = 1:MAX_ITR
		grad = x'*(x*W2-y) / m;
		W2 = W2 - lambda * grad;
	end
	
	figure(2)
	clf;
	plot(x(:,1),y,'o');
	hold on;
	plot(x(:,1), x*W2);
end

% 多维线性
if 1
	x = load('./data/ex3x.dat');
	y = load('./data/ex3y.dat');
	x(:,end+1) = 1;	
	
	%% 1. 正则方程
	W1 = x \ y;
	
	%% 2. 梯度
	% Loss函数 L = (XW-Y)^2 / (2*m)
	% 多维数据可能需要归一化
	meanx = mean(x);
	sigmax = std(x);
	x(:,1) = (x(:,1)-meanx(1)) / sigmax(1);
	x(:,2) = (x(:,2)-meanx(2)) / sigmax(2);
	
	
	% 不同的学习率(步长),影响收敛速度.
	MAXITER = 100;
	m = length(y);
	lambda = [0.01 0.03 0.1 0.3 1.3 1];
	style = {'b' 'r' 'g' 'k' 'b--' 'r--'};
	
	clf; hold on;
	for l = 1:length(lambda)
		W = zeros(size(x,2), 1);
		resi = zeros(MAXITER, 1);
		for it = 1:MAXITER
			resi(it) = sum((x*W-y).^2) / (2*m);
			grad = x' * (x*W - y) / m;
			W = W - lambda(l)*grad;
		end		
% 		plot(0:MAXITER-1, resi, style{l}, 'linewidth', 2);
		plot(0:49, resi(1:50), style{l}, 'linewidth', 2);
	end
	legend('0.01','0.03','0.1','0.3','1.3','1');	
end

end