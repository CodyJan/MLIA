function Regularization()

% 如果模型参数较多,而样本较少的话,容易产生过拟合
% 所以需要对参数进行惩罚,降低参数对模型的影响,
% 越小的参数说明模型越简单,越简单模型越不容易过拟合.



if 0
	% 线性回归
	% 损失函数
	% L = ((XW-Y)^2 + \lambda*W'W) / (2*m)
	% 用正则方程表示Regularization
	% W = (X'*X)^-1 * X' * Y          -->
	% W = (X'*X + \lambda diag(1,...,0))^-1 * X' * Y
	
	x = load('./ex5Data/ex5Linx.dat');
	y = load('./ex5Data/ex5Liny.dat');
	
	% 升维
	x = [x x.^2 x.^3 x.^4 x.^5];
	x(:,end+1) = 1;
	[m n] = size(x);
	
	clf;
	plot(x(:,1), y, 'o');
	hold on;
	
	style = {'g', 'b', 'r'};
	lam = [0 1 10];
	dig = eye(n);dig(end,end) = 0;
	xrange = linspace(min(x(:,1)),max(x(:,1)))';
	
	% 不同的正则化系数lam
	for i=1:3
		W(:,i) = inv(x'*x + lam(i)*dig) * x' * y;
		yrange = [xrange xrange.^2 xrange.^3 xrange.^4 xrange.^5 ones(100,1)]*W(:,i);
		plot(xrange, yrange, style{i});
	end
	legend('train data', '\lambda=0', '\lambda=1','\lambda=10');
end




if 1
	% Logistic回归
	% L = [-\sum(y*ln(p) + (1-y)*ln(1-p))  +  \lambda W'W/2] / m
	% 如果继续使用牛顿法,需要计算偏导:
	% 一阶偏导:
	% g1 = [-\sum (y_i - p_i)*x_i  +  \lambda W] / m
	% g1 = [X'(P - Y) + \lambda W] / m
	% 二阶偏导:
	% g2 = [\sum x^2 * e^{-XW} / (1+e^{-XW})^2  +  \lambda] / m
	% g2 = [X' * diag(p*(1-p)) * X   +  \lambda] / m
	
	x = load('./ex5Data/ex5Logx.dat');
	y = load('./ex5Data/ex5Logy.dat');
	
	figure(1);clf;
	plot(x(find(y),1),x(find(y),2),'o','MarkerFaceColor','b')
	hold on;
	plot(x(find(y==0),1),x(find(y==0),2),'r+')
	legend('y=1','y=0')
	
	% 升维
	addpath('./ex5Data');
	x = map_feature(x(:,1), x(:,2));
	[m n] = size(x);
	
	g = inline('1.0 ./ (1.0 + exp(-z))');
	MAXITER = 15;	
	dig = eye(n);dig(1,1) = 0;
	style = {'g', 'b', 'r'};
	lam = [0 1 10];
	
	figure(2);
	clf;hold on;
	for j=1:3
		W = zeros(n, 1);
		resid = zeros(MAXITER, 1);
		% 牛顿法
		for it = 1:MAXITER
			z = x*W;
			p = g(z);
			
			resid(it) = (sum(-y.*log(p)-(1-y).*log(1-p)) + lam(j)*W'*W/2)/m;
			grad = (x' * (p-y) + lam(j)*dig*W) / m;
			H = (x' * diag(p.*(1-p)) * x + lam(j)*dig) / m;
			
			W = W - H\grad;
		end
		
		figure(1);
		% 超平面位于p=0.5, 也就是XW=0
		u = linspace(-1, 1.5, 200);
		v = linspace(-1, 1.5, 200);		
		z = zeros(length(u), length(v));		
		for i = 1:length(u)
			for k = 1:length(v)
				z(i,k) = map_feature(u(i), v(k))*W;
			end
		end
		z = z';
		contour(u, v, z, [0, 0], style{j}, 'LineWidth', 2);	
		
		figure(2);
		plot(1:MAXITER, resid, style{j});
	end
	
	legend('\lambda=0', '\lambda=1', '\lambda=10');
end

end