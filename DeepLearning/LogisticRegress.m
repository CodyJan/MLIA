function LogisticRegress()


% 先将0-1转换成连续函数
% p = g(XW) = 1 / (1+e^{-XW})
% p是每个事件的概率
% 最大似然函数 l = \prod p^y * (1-p)^{1-y}
% 两边取对数
% ln(l) = \sum y*ln(p) + (1-y)*ln(1-p)
% 定义损失函数,最大似然->最小损失,所以需要负号
% L = -\sum( y*ln(p) + (1-y)*ln(1-p) ) / m
% 一阶偏导:
% g1 = -\sum (y_i - p_i)*x_i / m
% g1 = X' * (P - Y) / m
% 二阶偏导:
% g2 = \sum x^2 * e^{-XW} / (1+e^{-XW})^2 / m
% g2 = X' * diag(p*(1-p)) * X / m

x = load('./data/ex4x.dat');
y = load('./data/ex4y.dat');
x(:,end+1) = 1;
[m, n] = size(x);

% plot
pos = find(y);
neg = find(y==0);
clf
plot(x(pos,1), x(pos,2), '+');
hold on;
plot(x(neg,1), x(neg,2), 'o');

W = zeros(n, 1);
MAXITER = 7;
resi = zeros(MAXITER, 1);
g = inline('1.0 ./ (1.0 + exp(-z))');

% 牛顿法迭代
for it = 1:MAXITER
	z = x * W;
	p = g(z);	
	
	grad = x' * (p-y) / m;
	H = x' * diag(p.*(1-p)) * x / m;
	
	resi(it) = sum(y.*log(p) + (1-y).*log(1-p)) / (-m);
	
	W = W - H \ grad;
end

xtent = [min(x(:,1))-2  max(x(:,1))+2];
% 超平面位于p=0.5, 也就是XW=0
ytent = (xtent*W(1)+W(3)) / (-W(2));
plot(xtent, ytent);

end