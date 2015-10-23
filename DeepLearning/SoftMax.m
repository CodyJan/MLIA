
%% 简介
% 多分类问题，之前的Logistic是无法解决的，可以有两种解决方法：
%
% 1.SoftMax算法，相当于Logistic的推广。SoftMax在公式、推导、迭代上与Logistic有很大类似，并算法分类的结果是互斥、独立的，比如判断图像是数字几。
% 
% 2. 使用k次Logistic回归，也就是针对每类别回归一次。比较适合分类结果有关联的问题，比如图片属于哪个流派、风格。
%
%% 代价函数
% 先从最大似然说起
%
% $$L = \prod_{i=1}^{m}\prod_{j=1}^{k} p_{i,j}^{(y^{(i)}==j)}$$
%
% $$\log L = \sum_{i=1}^{m}\sum_{j=1}^{k} (y^{(i)}==j) \log p\left(y^{(i)}=j|x^{(i)};W\right)$$
%
% $$p\left(y^{(i)}=j|x^{(i)};W\right) = \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}}$$
%
% 所以SoftMax最小代价函数
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\sum_{j=1}^{k} (y^{(i)}==j) \log \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}}$$
% 
% 再看一下Logistic回归，即k=2的情况
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\left[ y^{(i)}\log \frac{e^{W_1x^{(i)}}}{e^{W_1x^{(i)}}+e^{W_2x^{(i)}}} + \left(1-y^{(i)}\right)\log \frac{e^{W_2x^{(i)}}}{e^{W_1x^{(i)}}+e^{W_2x^{(i)}}} \right]$$
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\left[ y^{(i)}\log \frac{1}{1+e^{-(W_1-W_2)x^{(i)}}} +\left(1-y^{(i)}\right)\log \left(1-\frac{1}{1+e^{-(W_1-W_2)x^{(i)}}}\right) \right]$$
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)}\log \frac{1}{1+e^{-Wx^{(i)}}} +\left(1-y^{(i)}\right)\log \left(1-\frac{1}{1+e^{-Wx^{(i)}}}\right)\right]$$
%
% 以上正好符合Logistic的公式，同时这也帮助理解了为什么SoftMax和Logistic采用两种不同的假设函数：
%
% $$p_{soft} = \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}}$$
%
% $$p_{log} = \frac{1}{1+e^{-Wx^{(i)}}}$$ 
%
%% 权重衰减
% SoftMax有一个特点，矩阵 $W$按行加减一个向量，并不会影响
% $p_{soft}$的结果，所以可能会使最小值的解不唯一，所以可以加一个权重衰减，防止上述问题。此权重衰减有点类似之前的正则项。
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\sum_{j=1}^{k} (y^{(i)}==j) \log \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}} + \frac{\lambda}{2}\sum_{i=1}^{k}\sum_{j=1}^{n}W_{ij}^2$$
%
%% 迭代更新
% 基本上可采用梯度下降、类牛顿两种方法，同时也决定了需要求一阶还是二阶偏导的问题。
%
% 一阶偏导：
% 
% $$\frac{\partial J}{\partial W} = -\frac{1}{m}\sum_{i=1}^{m} \left[x^{(i)}\left((y^{(i)}==j)-p_{ij}\right)\right] + \lambda W$$
%
% $$\frac{\partial J}{\partial W} = -\frac{1}{m}\left(Y-P\right)X^T + \lambda W$$
%
%% 程序代码
%

function SoftMax(imgs, labs, imgs_test, labs_test, maxiter)

K = 10;				% 总类别数
lambda = 1e-4;		% 衰减权重

if nargin < 5
	maxiter = 100;
	% MNIST Dataset: images and labels
	load('./data/softmax_data.mat');
	imgs = softmax_data.imgs;
	labs = softmax_data.labs;	
	imgs_test = softmax_data.imgs_test;
	labs_test = softmax_data.labs_test;	
	clear softmax_data;
end
labs(labs==0) = 10;
labs_test(labs_test==0) = 10;

[n m] = size(imgs);
wid = round(sqrt(n));
hei = round(n / wid);

X = imgs;
% 随机W矩阵
W = 0.005 * randn(K*n, 1);
% 形成类别的二维掩码，行是类别，列是样本
LABS = full(sparse(labs, 1:m, 1));


%%
% * L-BFGS算法求解SoftMax回归最优化问题
%
tic;
if 0
	addpath starter/minFunc
	options.Method = 'lbfgs';
	options.maxIter = maxiter;
	options.display = 'on';	
	[W, cost] = minFunc( @(p) cost_grad_func(p, X, LABS, lambda, true), W, options);	
else
	[W, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, X, LABS, lambda, p2), W, maxiter, 20, 0.55, 100);
end
toc


%%
% * 测试、评价
%
W = reshape(W, K, []);
[~, labs_pred] = max(W*imgs_test);
disp('test correct rate:');
sum(labs_pred' == labs_test) / length(labs_test)

end


%%
% * 代价-梯度函数
%
% $$J = J_0 + J_W$$
%
function [cost grad] = cost_grad_func(W, X, Y, lambda, calcgrad)

[n m] = size(X);
W = reshape(W, size(Y,1), []);
Z = W*X;
% Z = bsxfun(@minus, Z, max(Z, [], 1));
Z = exp(Z);
P = bsxfun(@rdivide, Z, sum(Z));
cost = -Y(:)'*log(P(:))/m + 0.5*lambda * W(:)'*W(:);

if calcgrad
	grad = -(Y - P)*X'/m + lambda * W;
	grad = grad(:);
end

end



%%
% * L-BFGS+Armijo实现的最优化算法
%
function [Wb cost] = mylbfgs(func, Wb, maxiter, maxmk, basemk, maxm)
% 1. 初值
[cost grad] = func(Wb, true);
eps = 1e-8;
Y = zeros(size(Wb,1), 0);
S = zeros(size(Wb,1), 0);
g = grad;
d = -g;
err = [cost];
sigma = 0.4;
c_old = cost;
g_old = g;
for k = 1:maxiter
	% 3. 线搜索, Armijo准则
	for mk = 0:maxmk
		alf = basemk^mk;
		[cost gg] = func(Wb+alf*d, true);
		if cost < c_old + sigma*alf*g'*d
			break;
		end
	end
	if mk>=maxmk
		disp('line search failed');
		alf = 1;
	end
	Wb = Wb + alf*d;
	
	
	% 2. 判断g_k
	g = gg;
	if g'*g < eps
		disp('g_k small enough');
		break;
	end
	
	
	% 缓存m组y和s
	y = g - g_old;
	s = alf*d;
	if size(Y,2) < maxm
		Y(:,end+1) = y;
		S(:,end+1) = s;
	else
		Y = [Y(:,2:end) y];
		S = [S(:,2:end) s];
	end
	
	% 4. two-loop求d_k+1
	H0 = (y'*s) / (y'*y);
	mcnt = size(Y,2);
	al = zeros(mcnt, 1);
	ro = zeros(mcnt, 1);
	for i = 1:mcnt
		ro(i) = 1 / (Y(:,i)'*S(:,i));
	end
	q = -g;
	for i = mcnt:-1:1
		al(i) = ro(i)*(S(:,i)'*q);
		q = q - al(i)*Y(:,i);
	end
	r = H0 * q;
	for i = 1:mcnt
		be = ro(i) * (Y(:,i)'*r);
		r = r + S(:,i)*(al(i) - be);
	end
	
	% 更新d
	d = r;
	g_old = g;
	c_old = cost;
	
	disp(sprintf('it:%d\t\tJ:%f\t\tstep:%f', k, cost, alf));
	err(end+1) = cost;
end
clf;
plot(err);
title('残差-迭代');
disp('done');
end