
%% 简介
% 本例演示线性解码器，其输出层不使用Sigmoid激活函数，而是一个线性激活方程，甚至直接等于。使用不同的激活韩红素带来的影响是，代价函数和梯度函数的计算不同。
%
% 以直接等于为例，代价函数主要是最后一层变成：
%
% $$a^{(n_l)} = z^{(n_l)}$$
%
% 梯度函数变化，也主要体现在最后一层，并逐步往前影响
%
% $$\delta^{(n_l)} = -(y-a^{(n_l)})$$
% 
% $$\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) f'(z^{(l)}) \qquad l=2,3,\cdots,n_l-1$$
%
% 之所以有时候会选择线性解码器的原因是，有时候输入层不能保证缩放到[0,1]之间，如果输出层仍然经过Sigmoid激活函数的话就一定在[0,1]之间，就会产生矛盾。
%
% 总的来说，线性解码器相当于在稀疏自编码的输出层上做了一些改动，从而导致代价和梯度的计算略不一样。


%% 程序代码
%
function LinearDecoder
nch = 3;
nodes = [8*8*3, 400, 8*8*3];
rho = 0.035;
lambda = 3e-3;
beta = 5;
epsilon = 0.1;
maxiter = 200;


%% 
% * 读取、预处理数据
%
load ./data/stl_patches.mat

% 原始图像
figure(1); clf; 
subplot(1,3,1);
showColorInfo(patches(:, 1:100));

% ZCA 白化
[n m] = size(patches);
patches = bsxfun(@minus, patches, mean(patches, 1));
sigma = patches * patches' / m;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;
subplot(1,3,2);
showColorInfo(patches(:, 1:100));


%%
% * 随机初值
r  = sqrt(6) / sqrt(nodes(1) + nodes(2) + 1);
Wb = [rand(2*nodes(1)*nodes(2),1)*2*r-r; zeros(nodes(1)+nodes(2), 1)];


%%
% * 迭代求解参数
if 1
	figure(2);clf;
	tic;
	[Wb, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, patches, rho, lambda, beta, nodes(1), nodes(2), p2), Wb, maxiter, 20, 0.55, 100);
	toc
	save('./data/ldwb.mat', 'Wb');
else
	load('./data/ldwb.mat');
end

W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
figure(1);
subplot(1,3,3);
showColorInfo((W1*ZCAWhite)');
end


%=============================================================================================================%


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

a2 = sigmoid( bsxfun(@plus, W1*data, b1) );
a3 = W2 * a2 + repmat(b2, 1, m);
diff = data - a3;

% 均方差项
Jc = diff(:)'*diff(:) * 0.5 / m;
% 权重衰减项
Jw = 0.5 * (W1(:)'*W1(:) + W2(:)'*W2(:));
% 稀疏惩罚项
mrho = sum(a2,2) / m;
Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
% 总代价
cost = Jc + lambda*Jw + beta*Jsp;

if calcgrad
	betrho = beta*(-rho./mrho + (1-rho)./(1-mrho));	
	d3 = -diff;	
	d2 = bsxfun(@plus, W2'*d3, betrho) .* a2 .* (1-a2);
	
	W1grad = ( d2*data') / m + lambda*W1;
	b1grad = ( sum(d2,2)) / m;
	W2grad = ( d3*a2') / m + lambda*W2;
	b2grad = ( sum(d3, 2)) / m;
	grad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];
end
end


%%
% * 实现网络
function showColorInfo(A)
	A = A - mean(A(:));
	[n m] = size(A);	
	wid = round(sqrt(n/3));
	hei = n/3/wid;	
	gwid = ceil(sqrt(m));
	ghei = m / gwid;
	
	r = reshape(A(1:wid*hei,:), hei*wid, []);
	g = reshape(A(wid*hei+1:2*wid*hei, :), hei*wid, []);
	b = reshape(A(2*wid*hei+1:3*wid*hei, :), hei*wid, []);
	r = r ./ repmat(max(abs(r)), wid*hei, 1);
	g = g ./ repmat(max(abs(g)), wid*hei, 1);
	b = b ./ repmat(max(abs(b)), wid*hei, 1);	
	
	im = ones(ghei*(hei+1)+1, gwid*(wid+1)+1, 3);	
	for j = 1:ghei
		for i = 1:gwid
			im((j-1)*(hei+1)+2:j*(hei+1), (i-1)*(wid+1)+2:i*(wid+1), 1) = reshape(r(:, (j-1)*gwid+i), hei, wid);
			im((j-1)*(hei+1)+2:j*(hei+1), (i-1)*(wid+1)+2:i*(wid+1), 2) = reshape(g(:, (j-1)*gwid+i), hei, wid);
			im((j-1)*(hei+1)+2:j*(hei+1), (i-1)*(wid+1)+2:i*(wid+1), 3) = reshape(b(:, (j-1)*gwid+i), hei, wid);
		end
	end
	im = (im+1)/2;
	imagesc(im);
	axis equal;
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