
%% 介绍
% 一个3层的网络,用来进行稀疏性自编码算法的验证,输入层输出层的节点数相同,隐藏层节点数少于输入层.
%
% 大致的意思是,将输入层的数据,压缩到中间层(稀疏性表达),再复原到输出层.证明输入层的数据是可以稀疏表达的.

function SparseAutoencode()

clear all;
clc;
clf;

hidsize = 5;			% 隐藏层节点数目5*5
patsize = 8;			% 输入层节点数目8*8
patnum = 10000;			% 样本数目
rho = 0.01;				% 稀疏值,通常较小
beta = 3;				% 稀疏值惩罚项权重
lambda = 1e-4;			% 权重惩罚项权重

%% 预处理数据
% * 读取10张图片, 每张图片随机位置上获取一些patch图像
IMGS = load('./sparseae_exercise/IMAGES.mat');
IMGS = IMGS.IMAGES;
[hei wid cnt] = size(IMGS);
pats = zeros(patsize^2, patnum);
for i = 1:cnt	
	for j = 1:patnum/cnt
		pos = randi([1, min(wid,hei)-patsize+1], 2, 1);
		pats(:,(i-1)*patnum/cnt + j) = reshape(IMGS(pos(2):pos(2)+patsize-1, pos(1):pos(1)+patsize-1, i), [], 1);
	end
end

%%
% * 归一化: $\mathcal{N}(0,3\sigma)$
pats = bsxfun(@minus, pats, mean(pats));
pstd = 3 * std(pats(:));
pats = max(min(pats, pstd), -pstd) / pstd;
% [-1,1] -> [0.1, 0.9]
pats = (pats + 1) * 0.4 + 0.1;


%%
% * 显示随机patch的部分原始信息
A = pats(:, randi(patnum, 200, 1));
cols = ceil(sqrt(size(A,2)));
rows = ceil(size(A,2) / cols);
im = zeros(rows*(patsize+1)+1, cols*(patsize+1)+1);

for i = 1:size(A,2)
	c = mod((i-1), cols)+1;
	r = floor((i-1) / cols)+1;
	im( (r-1)*(patsize+1)+2:r*(patsize+1), (c-1)*(patsize+1)+2:c*(patsize+1) ) = mat2gray(reshape(A(:, i), patsize, patsize));
end
clf; imshow(im);


%% 
% * 产生随机的W和b, W是在 $r=\pm\sqrt{6/(n_{in}+n_{out}+1)}$ 范围之间的伪随机数. 否则容易收敛到局部
r  = sqrt(6) / sqrt(hidsize^2 + patsize^2 + 1);
W1 = rand(hidsize^2, patsize^2)*2*r - r;
W2 = rand(patsize^2, hidsize^2)*2*r - r;
b1 = zeros(hidsize^2, 1);
b2 = zeros(patsize^2, 1);




%% 计算三项代价（均方差、权重衰减、系数惩罚）、残差、梯度。
% * 前向算法, 依次计算每层每节点的激发值及各项代价
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

z2 = W1*pats + repmat(b1, 1, patnum);
a2 = sigmoid(z2);
z3 = W2*a2 + repmat(b2, 1, patnum);
a3 = sigmoid(z3);
% 均方差项
Jc = sum(sum((a3-pats).^2)) * 0.5 / patnum;
% 权重衰减项
Jw = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% 稀疏惩罚项
mrho = sum(a2,2) / patnum;
Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
% 总代价
J = Jc + lambda*Jw + beta*Jsp;


%%
% * 反向传导算法, 首先需要计算残差. 残差代表着每节点对J的影响, 也就是
%
% $$\delta_j^{(l)} = \partial J / \partial z_j^{(l)}$$
%
% 在一般情况下:
%
% $$\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \cdot f'(z^{(l)})$$
%
% 考虑稀疏惩罚: 
% 
% $$\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)} + \beta\left( -\frac{\rho}{\hat{\rho}} + \frac{1-\rho}{1-\hat{\rho}} \right)\right) \cdot f'(z^{(l)})$$
%
d3 = -(pats-a3).*sigmoidInv(z3);
betrho = beta*(-rho./mrho + (1-rho)./(1-mrho));
d2 = (W2' * d3 + repmat(betrho,1,patnum) ) .* sigmoidInv(z2);

% 梯度计算
W1grad = ( d2*pats') / patnum + lambda*W1;
b1grad = ( sum(d2,2)) / patnum;
W2grad = ( d3*a2') / patnum + lambda*W2;
b2grad = ( sum(d3, 2)) / patnum;
allgrad = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];
allWb = [W1(:); W2(:); b1(:); b2(:)];

%% 梯度检查
% * 检查上述理论梯度值是否符合数值梯度值
if 0
	eps = 1e-4;
	nlen = length(allgrad);	
	len1 = patsize^2 * hidsize^2;
	len2 = patsize^2 * hidsize^2;
	len3 = hidsize^2;
	len4 = patsize^2;
	allgradnum = zeros(nlen, 1);
	for j = 1:nlen
		dels = zeros(nlen, 1);
		dels(j) = eps;
		delW1 = reshape(dels(1:len1), hidsize^2, []);
		delW2 = reshape(dels(len1+1:len1+len2), patsize^2, []);
		delb1 = reshape(dels(len1+len2+1:len1+len2+len3), hidsize^2, []);
		delb2 = reshape(dels(len1+len2+len3+1:end), patsize^2, []);
		
		a2 = sigmoid( (W1+delW1)*pats + repmat((b1+delb1), 1, patnum) );
		a3 = sigmoid( (W2+delW2)*a2 + repmat((b2+delb2), 1, patnum) );
		Jc = sum(sum((a3-pats).^2)) * 0.5 / patnum;
		Jw = 0.5 * (sum(sum((W1+delW1).^2)) + sum(sum((W2+delW2).^2)));
		mrho = sum(a2,2) / patnum;
		Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
		J_0 = Jc + lambda*Jw + beta*Jsp;
		
		a2 = sigmoid( (W1-delW1)*pats + repmat((b1-delb1), 1, patnum) );
		a3 = sigmoid( (W2-delW2)*a2 + repmat((b2-delb2), 1, patnum) );
		Jc = sum(sum((a3-pats).^2)) * 0.5 / patnum;
		Jw = 0.5 * (sum(sum((W1-delW1).^2)) + sum(sum((W2-delW2).^2)));
		mrho = sum(a2,2) / patnum;
		Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
		J_1 = Jc + lambda*Jw + beta*Jsp;
		
		allgradnum(j) = (J_0 - J_1) / (2*eps);
	end

	disp( norm(allgrad-allgradnum) / norm(allgrad+allgradnum) );
end


%% L-BFGS最优化,寻找Wb
% * 使用minFunc函数
if 0
	addpath starter/
	addpath starter/minFunc
	options.Method = 'lbfgs';
	options.maxIter = 100;
	options.display = 'on';
	tic;
	[optWb, cost, ~, ~, lams] = minFunc( @(p) sparseAutoencoderCost(p, patsize^2, hidsize^2, lambda, rho, beta, pats), allWb, options);
	toc
end


%%
% * 自实现的L-BFGS算法
if 1
	tic;
	patlen = patsize^2;
	hidlen = hidsize^2;
	x = allWb;
	eps = 1e-4;
	g = allgrad;
	d = -g;
	
	m = 100;
	Y = zeros(size(x,1), 0);
	S = zeros(size(x,1), 0);
	res = [J];
	c1 = 1e-4;
	c2 = 0.9;
	g_old = g;
	J_old = J;
	lam = min(1,1/sum(abs(g)));	
	for k = 1:200
		% 缺少一步线搜索步长,目前使用固定步长1
		% ...
		%
		
		x = x + lam*d;
		W1 = reshape(x(1:patlen*hidlen), hidlen, []);
		W2 = reshape(x(patlen*hidlen+1:2*patlen*hidlen), patlen, []);
		b1 = reshape(x(2*patlen*hidlen+1:2*patlen*hidlen+hidlen), hidlen, []);
		b2 = reshape(x(2*patlen*hidlen+hidlen+1:end), patlen, []);
		
		z2 = W1*pats + repmat(b1, 1, patnum);
		a2 = sigmoid(z2);
		z3 = W2*a2 + repmat(b2, 1, patnum);
		a3 = sigmoid(z3);
		
		Jc = sum(sum((a3-pats).^2)) * 0.5 / patnum;
		Jw = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
		mrho = sum(a2,2) / patnum;
		Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
		J = Jc + lambda*Jw + beta*Jsp;	
		
		d3 = -(pats-a3).*sigmoidInv(z3);
		betrho = beta*(-rho./mrho + (1-rho)./(1-mrho));
		d2 = (W2' * d3 + repmat(betrho,1,patnum) ) .* sigmoidInv(z2);
		W1grad = ( d2*pats') / patnum + lambda*W1;
		b1grad = ( sum(d2,2)) / patnum;
		W2grad = ( d3*a2') / patnum + lambda*W2;
		b2grad = ( sum(d3, 2)) / patnum;
		g = [W1grad(:); W2grad(:); b1grad(:); b2grad(:)];	
		
		y = g - g_old;
		s = lam*d;
		
		% 缓存m组		
		if size(Y,2) < m
			Y(:,end+1) = y;
			S(:,end+1) = s;
		else
			Y = [Y(:,2:end) y];
			S = [S(:,2:end) s];
		end		
		
		% two-loop求d_k+1	
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
		
		d = r;
		lam = 1;		
		g_old = g;
		J_old = J;
		
		disp(sprintf('J:%f\t\tit:%d', J, k));
		res(end+1) = J;
	end
	toc
	clf;
	plot(res);
	title('残差-迭代');
	disp('done');
	optWb = x;
end


%%
% * 显示W1信息
im = zeros(hidsize*(patsize+1)+1, hidsize*(patsize+1)+1);
W1 = reshape(optWb(1:patsize^2*hidsize^2), hidsize^2, []);
W1 = W1';
for i = 1:hidsize^2
	c = mod((i-1), hidsize)+1;
	r = floor((i-1) / hidsize)+1;
	im( (r-1)*(patsize+1)+2:r*(patsize+1), (c-1)*(patsize+1)+2:c*(patsize+1) ) = mat2gray(reshape(W1(:,i), patsize, []));
end
clf; imshow(im);

end

function y = sigmoid(x)
	y = 1 ./ (1+exp(-x));
	y(isnan(y)) = 0;
end
function y = sigmoidInv(x)
	ex = exp(-x);	
	y = ex ./ (1+ex).^2;
	y(isnan(y)) = 0;
end
