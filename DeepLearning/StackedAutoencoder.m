
%% 简介
% 本例演示栈式自编码算法，是一个多层的网络结构，以识别手写数字为目标。
%
% 栈式自编码网络是多层稀疏自编码器组成的神经网络，前一层自编码器的输出作为后一层的输入，最后再加上一层SoftMax回归或其他回归输出分类结果。
% 由此来看，首先是依次逐层得到每层的自编码器参数，最后调用一次SoftMax回归。
%
% 每层的自编码器相对于对输入数据再进行一次“压缩”，而这些“压缩”的数据通过复原与输入数据是尽量相似的。
% 也就是栈式自编码器每层都需要执行一遍"SparseAutoencode"，并只保留 $W^{(1)}$ 信息。
%
% 本例是4层的网络，通过栈式自编码算法得到的正确率只有89%，并没有SoftMax直接回归出来的正确率高，这个原因可能是默认的系数各不一样，理论上来说4层网络应该比3层网络更好一点。
%
% 栈式自编码需要一个额外的“微调”来提高精度，所谓“微调”就是以上述的结果作为初始值，再将所有层串起来进行反向传播算法迭代，计算量较大。
%
%% 程序代码
%
function StackedAutoencoder()
return
nodes = [28*28 200 200 28*28];		% 每层节点数
rho = 0.1;							% 稀疏性
lambda = 3e-3;						% weight decay parameter       
beta = 3;							% weight of sparsity penalty term  
maxiter = 100;

%%
% * 读取数据
%
load('./data/softmax_data.mat');
train_imgs = softmax_data.imgs;
train_labs = softmax_data.labs;
train_labs(train_labs==0) = 10;
test_imgs = softmax_data.imgs_test;
test_labs = softmax_data.labs_test;
test_labs(test_labs==0) = 10;
clear softmax_data;


%%
% * 第二层的自编码器
%
r  = sqrt(6) / sqrt(nodes(1) + nodes(2) + 1);
Wb1 = [rand(2*nodes(1)*nodes(2),1)*2*r-r; zeros(nodes(1)+nodes(2), 1)];
tic;
if 0	
	[Wb1, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, train_imgs, rho, lambda, beta, nodes(1), nodes(2), p2), Wb1, maxiter, 20, 0.55, 50);
	save('./data/Wb1.mat', 'Wb1');
else
	load('./data/Wb1.mat');
end
toc
%%
% * 第二层的输出
%
W1 = reshape(Wb1(1:nodes(1)*nodes(2)), nodes(2), []);
b1 = Wb1(2*nodes(1)*nodes(2)+1:2*nodes(1)*nodes(2)+nodes(2));
z2 = bsxfun(@plus, W1*train_imgs, b1);
train_feat = sigmoid(z2);
z2 = bsxfun(@plus, W1*test_imgs, b1);
test_feat = sigmoid(z2);


%%
% * 第三层的自编码器
%
r  = sqrt(6) / sqrt(nodes(2) + nodes(3) + 1);
Wb2 = [rand(2*nodes(2)*nodes(3),1)*2*r-r; zeros(nodes(2)+nodes(3), 1)];
tic;
if 0
	[Wb2, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, train_feat, rho, lambda, beta, nodes(2), nodes(2), p2), Wb2, maxiter, 20, 0.55, 50);
	save('./data/Wb2.mat', 'Wb2');
else
	load('./data/Wb2.mat');
end
toc
%%
% * 第三层的输出
%
W1 = reshape(Wb2(1:nodes(2)*nodes(3)), nodes(3), []);
b1 = Wb2(2*nodes(2)*nodes(3)+1:2*nodes(2)*nodes(3)+nodes(3));
z2 = bsxfun(@plus, W1*train_feat, b1);
train_feat = sigmoid(z2);
z2 = bsxfun(@plus, W1*test_feat, b1);
test_feat = sigmoid(z2);


%%
% * 输出层调用SoftMax回归，并测试
%
SoftMax(train_feat, train_labs, test_feat, test_labs, maxiter);


%%
% * 将所有层串起来在进行一次反向传播算法
%

end


%=================================================================================================================%


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
% * 代价-梯度函数，自编码器
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