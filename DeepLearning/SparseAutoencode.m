
%% 简介
% 这个代码例子，展示了一个3层的网络,用来进行稀疏性自编码算法的验证,输入层输出层的节点数相同,隐藏层节点数少于输入层.
%
% 大致的意思是，将输入层的数据，压缩到中间层(稀疏性表达)，再复原到输出层，证明输入层的数据是可以稀疏表达的。
%
%% 代价函数
% 之前的例子中，只用考虑网络处理后的误差及抑制 $W$
% 过大。但这个例子还需要额外考虑稀疏性，也就是我们希望隐藏层能够达到一定程度的稀疏表达。所以代价函数如下：
%
% 均方差项：即输出层与输入层要尽量相似。
% 
% $$J_0 = \frac{1}{m}\sum_{i=1}^{m}\left(\frac{1}{2}\|h_{W,b}(x^{(i)})-y^{(i)}\|^2\right)$$
% 
% 正则项(权重衰减项)：即抑制参数 $W$过大，防止过度拟合。
%
% $$J_W = \frac{\lambda}{2}\sum_{l=1}^{n_l-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}} W_{ji}^{(l)}$$
%
% 稀疏惩罚项：即使隐藏层的激活值尽量小，从而达到稀疏性的假设。
%
% $$J_{sp} = \beta\sum_{j=1}^{s_2}\left(\rho\log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}\right)$$
%
% 最后：
%
% $$J = J_0 + J_W + J_{sp}$$
%
%% 梯度计算
% 在计算梯度的时候，需要充分考虑上述的代价函数的构成，然后利用反向传播的思路，依次求出对每层的 $W,b$ 的偏导。
%
% 反向传播的思路主要是利用后一层的残差，计算出前一层的偏导，依次从最后一层开始，直到第二层。与上述代价函数计算的过程正好相反，所以称之为反向传播算法。先看看每层残差的公式：
%
% $$\delta^{(n_l)} = -(y-a^{(n_l)})f'(z^{(n_l)})$$
%
% $$\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)} + \beta\left( -\frac{\rho}{\hat{\rho}} + \frac{1-\rho}{1-\hat{\rho}} \right)\right) f'(z^{(l)}) \qquad l=2,3,\cdots,n_l-1$$
%
% 对 $W,b$ 的偏导：
%
% $$\frac{\partial J}{\partial W^{(l)}} = \frac{1}{m}\delta^{(l+1)} \left(a^{(l)}\right)^T + \lambda W^{(l)}$$
%
% $$\frac{\partial J}{\partial b^{(l)}} = \frac{1}{m}\delta^{(l+1)}$$
%
%% L-BFGS算法
% 使用L-BFGS算法迭代更新参数 $W,b$ ，L-BFGS算法是性能较好的拟牛顿算法：一是二阶偏导，收敛速度快；二是不用求逆Hesse矩阵，计算量小；三是使用较少内存。
%
% L-BFGS算法的流程：
% 
% * 初值
%
% * 判断 $\|g_k\| \leq \epsilon$ ，满足则退出，否则继续。
%
% * 线搜索步长 $\alpha_k$ ，可以考虑Armijo准则
%
% $$f(x_k+\beta^md_k) \leq f(x_k) + \sigma \beta^m g_k^Td_k$$
%
% * two-loop算法求 $d_{k+1}$
%
%% 代码部分
%

function SparseAutoencode()


patnum = 10000;								% 样本数目
rho = 0.01;									% 稀疏值,通常较小
beta = 3;									% 稀疏值惩罚项权重
lambda = 1e-4;								% 权重惩罚项权重
nodes = [8^2 5^2 8^2];						% 每层节点数
maxiter = 100;								% 最大迭代次数
patsize = sqrt(nodes(1));


%%
% * 读取10张图片, 每张图片随机位置上获取一些patch图像
%
IMGS = load('./data/IMAGES.mat');
IMGS = IMGS.IMAGES;
[hei wid cnt] = size(IMGS);
imgs = zeros(patsize^2, patnum);
for i = 1:cnt
	for j = 1:patnum/cnt
		pos = randi([1, min(wid,hei)-patsize+1], 2, 1);
		imgs(:,(i-1)*patnum/cnt + j) = reshape(IMGS(pos(2):pos(2)+patsize-1, pos(1):pos(1)+patsize-1, i), [], 1);
	end
end


%%
% * 规范化：将原数据归0归1，使其符合正态分布 $\sim\mathcal{N}(0,3\sigma)$
%
imgs = bsxfun(@minus, imgs, mean(imgs));
pstd = 3 * std(imgs(:));
imgs = max(min(imgs, pstd), -pstd) / pstd;
% [-1,1] -> [0.1, 0.9]
imgs = (imgs + 1) * 0.4 + 0.1;


% 显示随机patch的部分原始信息
if 0
	A = imgs(:, randi(patnum, 200, 1));
	cols = ceil(sqrt(size(A,2)));
	rows = ceil(size(A,2) / cols);
	im = zeros(rows*(patsize+1)+1, cols*(patsize+1)+1);
	for i = 1:size(A,2)
		c = mod((i-1), cols)+1;
		r = floor((i-1) / cols)+1;
		im( (r-1)*(patsize+1)+2:r*(patsize+1), (c-1)*(patsize+1)+2:c*(patsize+1) ) = mat2gray(reshape(A(:, i), patsize, patsize));
	end
	clf; imshow(im);
end


%%
% * 产生随机的 $W,b$ ， $W$ 是在 $r=\pm\sqrt{6/(n_{in}+n_{out}+1)}$ 范围之间的伪随机数. 否则容易收敛到局部
%
r  = sqrt(6) / sqrt(nodes(1) + nodes(2) + 1);
Wb = [rand(2*nodes(1)*nodes(2),1)*2*r-r; zeros(nodes(1)+nodes(2), 1)];
%%
% * L-BFGS算法求解稀疏自编码最优化问题
%
tic;
if 0
	addpath starter/
	addpath starter/minFunc
	options.Method = 'lbfgs';
	options.maxIter = maxiter;
	options.display = 'on';	
	[Wb, cost, ~, ~, lams] = minFunc( @(p) sparseAutoencoderCost(p, nodes(1), nodes(2), lambda, rho, beta, imgs), Wb, options);
else
	[Wb, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, imgs, rho, lambda, beta, nodes(1), nodes(2), p2), Wb, maxiter, 20, 0.55, 100);
end
toc


%%
% * 显示W1信息
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
	im( (r-1)*(inhei+1)+2:r*(inhei+1), (c-1)*(inwid+1)+2:c*(inwid+1) ) = reshape(W1(i,:), inhei, []);
end
im = mat2gray(im);
im(1:inhei+1:end, :) = 0; im(:, 1:inwid+1:end) = 0;
imshow(im);

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
function [cost grad] = cost_grad_func(x, data, rho, lambda, beta, siz1, siz2, calcgrad)
[n m] = size(data);
siz = siz1*siz2;
W1 = reshape(x(1:siz), siz2, []);
W2 = reshape(x(siz+1:2*siz), siz1, []);
b1 = x(2*siz+1:2*siz+siz2);
b2 = x(2*siz+siz2+1:end);

z2 = bsxfun(@plus, W1*data, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2*a2, b2);
a3 = sigmoid(z3);

% 均方差项
Jc = sum(sum((a3-data).^2)) * 0.5 / m;
% 权重衰减项
Jw = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% 稀疏惩罚项
mrho = sum(a2,2) / m;
Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
% 总代价
cost = Jc + lambda*Jw + beta*Jsp;

if calcgrad
	d3 = -(data-a3).*sigmoidInv(z3);
	betrho = beta*(-rho./mrho + (1-rho)./(1-mrho));
	d2 = (W2' * d3 + repmat(betrho,1,m) ) .* sigmoidInv(z2);
	W1grad = ( d2*data') / m + lambda*W1;
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