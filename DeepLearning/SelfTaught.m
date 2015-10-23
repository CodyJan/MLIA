
%% 简介
% 现实中的很多问题可以通过大量没有标记的样本去进行无监督的特征学习，然后利用这些特征方法在少量有标记的样本上进行有监督的学习。
%
% 联系实际问题的话，比如一个物体识别的问题，我们首先要决定采用何种特征，如何选择特征呢，采用无监督的特征学习（稀疏自编码），得到隐藏层的输出，既认为是最具有代表性的特征。
% 无监督学习输入层到隐藏层的转换过程，也就是特征的生成方法。
% 利用上述方法对有标记的样本提取出特征，再进行有监督的学习(SoftMax、线性回归等)，既能得到一个较好的识别方法。
%
% 有两种方法进行无监督学习，一种叫自学习，对无标记、有标记的数据没有明显要求；另一种叫半监督学习，要求无标记、有标记的数据符合同样的分布。也即是说前者不限制数据来源，后者要求数据都是固定的几类，对数据有一定的限制。
%
% 本例采用手写数字识别，从数字5-9的图像中进行无监督的特征学习，确定出特征提取方法；然后使用数字0-4的图像进行有监督的学习（调用SoftMax函数）；最后进行测试。
%
% 结果上来看，自学习算法能将准确率提高到98%，而如果不提取特征直接SoftMax训练只有96.7%
%
%% 程序代码
%
function SelfTaught()

clear;clc;clf;

rho = 0.1;							% 稀疏值
K = 10;								% 类别数
nodes = [28*28 200 28*28];			% 每层节点数
lambda = 3e-3;
beta = 3;


%%
% * 读取数据
%
load('./data/softmax_data.mat');
labset = find(softmax_data.labs <= 4);
unlabset = find(softmax_data.labs >=5);
% 无标记样本
train_udata = softmax_data.imgs(:, unlabset);
% 有标记样本
num = round(length(labset)/2);
train_ldata = softmax_data.imgs(:, labset(1:num));
train_labs = softmax_data.labs(labset(1:num));
% 测试样本
test_ldata = softmax_data.imgs(:, labset(num+1:end));
test_labs = softmax_data.labs(labset(num+1:end));
clear softmax_data;


%%
% * 产生随机的 $W,b$ ， $W$ 是在 $r=\pm\sqrt{6/(n_{in}+n_{out}+1)}$ 范围之间的伪随机数. 否则容易收敛到局部
%
r  = sqrt(6) / sqrt(nodes(1) + nodes(2) + 1);
Wb = [rand(2*nodes(1)*nodes(2),1)*2*r-r; zeros(nodes(1)+nodes(2), 1)];


%%
% * LBFGS算法，最优化求解无监督特征学习
%
tic;
if 0
	maxiter = 200;
	addpath starter/minFunc
	options.Method = 'lbfgs';
	options.maxIter = maxiter;
	options.display = 'on';
% 	[Wb, cost] = minFunc( @(p) cost_grad_func(p, train_udata, rho, lambda, beta, nodes(1), nodes(2), true), Wb, options);
	[Wb, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, train_udata, rho, lambda, beta, nodes(1), nodes(2), p2), Wb, maxiter, 20, 0.55, 50);
	save('./data/ulabWb.mat', 'Wb');
else
	load('./data/ulabWb.mat');
end
toc


%%
% * 显示无监督特征学习的 $W^{(1)}$ 信息
%
inwid = sqrt(nodes(1));
inhei = inwid;
hidwid = ceil(sqrt(nodes(2)));
hidhei = ceil(nodes(2) / hidwid);
im = zeros( hidhei * (inhei+1) + 1, hidwid * (inwid+1) + 1);
W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
for i = 1:nodes(2)
	c = mod((i-1), hidwid)+1;
	r = floor((i-1) / hidwid)+1;
	im( (r-1)*(inhei+1)+2:r*(inhei+1), (c-1)*(inwid+1)+2:c*(inwid+1) ) = (reshape(W1(i,:), inhei, []));
end
imshow(mat2gray(im));


%%
% * 利用上述特征生成方法，从有标记样本及测试样本中提取特征
%
W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
b1 = Wb(2*nodes(1)*nodes(2)+1:2*nodes(1)*nodes(2)+nodes(2));
z2 = bsxfun(@plus, W1*train_ldata, b1);
train_feat = sigmoid(z2);
z2 = bsxfun(@plus, W1*test_ldata, b1);
test_feat = sigmoid(z2);


%%
% * 调用SoftMax算法进行训练，将上述特征作为输入层，分类结果作为输出层。
% * 带入测试样本，进行评价。
%
SoftMax(train_feat, train_labs, test_feat, test_labs);

end




%%
% * 激活函数
%
function y = sigmoid(x)
	y = 1 ./ (1+exp(-x));
end
function y = sigmoidInv(x)
	ex = exp(-x);	
	y = ex ./ (1+ex).^2;
end


%%
% * 代价-梯度函数
%
% $$J = J_0 + J_W + J_{sp}$$
%
function [cost grad] = cost_grad_func(Wb, X, rho, lambda, beta, siz1, siz2, calcgrad)
[n m] = size(X);
siz = siz1*siz2;
W1 = reshape(Wb(1:siz), siz2, []);
W2 = reshape(Wb(siz+1:2*siz), siz1, []);
b1 = Wb(2*siz+1:2*siz+siz2);
b2 = Wb(2*siz+siz2+1:end);

z2 = bsxfun(@plus, W1*X, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2*a2, b2);
a3 = sigmoid(z3);

% 均方差项
Jc = sum(sum((a3-X).^2)) * 0.5 / m;
% 权重衰减项
Jw = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% 稀疏惩罚项
mrho = sum(a2,2) / m;
Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
% 总代价
cost = Jc + lambda*Jw + beta*Jsp;

if calcgrad
	d3 = -(X-a3).*sigmoidInv(z3);
	betrho = beta*(-rho./mrho + (1-rho)./(1-mrho));
	d2 = (W2' * d3 + repmat(betrho,1,m) ) .* sigmoidInv(z2);
	W1grad = ( d2*X') / m + lambda*W1;
	b1grad = ( sum(d2,2)) / m;
	W2grad = ( d3*a2') / m + lambda*W2;
	b2grad = ( sum(d3, 2)) / m;
	grad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];
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