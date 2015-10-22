
%% ���
% ����������ӣ�չʾ��һ��3�������,��������ϡ�����Ա����㷨����֤,����������Ľڵ�����ͬ,���ز�ڵ������������.
%
% ���µ���˼�ǣ������������ݣ�ѹ�����м��(ϡ���Ա��)���ٸ�ԭ������㣬֤�������������ǿ���ϡ����ġ�
%
%% ���ۺ���
% ֮ǰ�������У�ֻ�ÿ������紦���������� $W$
% ���󡣵�������ӻ���Ҫ���⿼��ϡ���ԣ�Ҳ��������ϣ�����ز��ܹ��ﵽһ���̶ȵ�ϡ������Դ��ۺ������£�
%
% ���������������������Ҫ�������ơ�
% 
% $$J_0 = \frac{1}{m}\sum_{i=1}^{m}\left(\frac{1}{2}\|h_{W,b}(x^{(i)})-y^{(i)}\|^2\right)$$
% 
% ������(Ȩ��˥����)�������Ʋ��� $W$���󣬷�ֹ������ϡ�
%
% $$J_W = \frac{\lambda}{2}\sum_{l=1}^{n_l-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}} W_{ji}^{(l)}$$
%
% ϡ��ͷ����ʹ���ز�ļ���ֵ����С���Ӷ��ﵽϡ���Եļ��衣
%
% $$J_{sp} = \beta\sum_{j=1}^{s_2}\left(\rho\log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}\right)$$
%
% ���
%
% $$J = J_0 + J_W + J_{sp}$$
%
%% �ݶȼ���
% �ڼ����ݶȵ�ʱ����Ҫ��ֿ��������Ĵ��ۺ����Ĺ��ɣ�Ȼ�����÷��򴫲���˼·�����������ÿ��� $W,b$ ��ƫ����
%
% ���򴫲���˼·��Ҫ�����ú�һ��Ĳв�����ǰһ���ƫ�������δ����һ�㿪ʼ��ֱ���ڶ��㡣���������ۺ�������Ĺ��������෴�����Գ�֮Ϊ���򴫲��㷨���ȿ���ÿ��в�Ĺ�ʽ��
%
% $$\delta^{(n_l)} = -(y-a^{(n_l)})f'(z^{(n_l)})$$
%
% $$\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)} + \beta\left( -\frac{\rho}{\hat{\rho}} + \frac{1-\rho}{1-\hat{\rho}} \right)\right) f'(z^{(l)}) \qquad l=2,3,\cdots,n_l-1$$
%
% �� $W,b$ ��ƫ����
%
% $$\frac{\partial J}{\partial W^{(l)}} = \frac{1}{m}\delta^{(l+1)} \left(a^{(l)}\right)^T + \lambda W^{(l)}$$
%
% $$\frac{\partial J}{\partial b^{(l)}} = \frac{1}{m}\delta^{(l+1)}$$
%
%% L-BFGS�㷨
% ʹ��L-BFGS�㷨�������²��� $W,b$ ��L-BFGS�㷨�����ܽϺõ���ţ���㷨��һ�Ƕ���ƫ���������ٶȿ죻���ǲ�������Hesse���󣬼�����С������ʹ�ý����ڴ档
%
% L-BFGS�㷨�����̣�
% 
% * ��ֵ
%
% * �ж� $\|g_k\| \leq \epsilon$ ���������˳������������
%
% * ���������� $\alpha_k$ �����Կ���Armijo׼��
%
% $$f(x_k+\beta^md_k) \leq f(x_k) + \sigma \beta^m g_k^Td_k$$
%
% * two-loop�㷨�� $d_{k+1}$
%
%% ���벿��
%

function SparseAutoencode()

clear all;
clc;
clf;

hidsize = 5;								% ���ز�ڵ���Ŀ5*5
patsize = 8;								% �����ڵ���Ŀ8*8
patnum = 10000;								% ������Ŀ
rho = 0.01;									% ϡ��ֵ,ͨ����С
beta = 3;									% ϡ��ֵ�ͷ���Ȩ��
lambda = 1e-4;								% Ȩ�سͷ���Ȩ��
nodes = [patsize^2 hidsize^2 patsize^2];	% ÿ��ڵ���


%%
% * ��ȡ10��ͼƬ, ÿ��ͼƬ���λ���ϻ�ȡһЩpatchͼ��
%
IMGS = load('./data/IMAGES.mat');
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
% * �淶������ԭ���ݹ�0��1��ʹ�������̬�ֲ� $\sim\mathcal{N}(0,3\sigma)$
%
pats = bsxfun(@minus, pats, mean(pats));
pstd = 3 * std(pats(:));
pats = max(min(pats, pstd), -pstd) / pstd;
% [-1,1] -> [0.1, 0.9]
pats = (pats + 1) * 0.4 + 0.1;


% ��ʾ���patch�Ĳ���ԭʼ��Ϣ
if 1
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
end


%%
% * ��������� $W,b$ �� $W$ ���� $r=\pm\sqrt{6/(n_{in}+n_{out}+1)}$ ��Χ֮���α�����. ���������������ֲ�
%
r  = sqrt(6) / sqrt(hidsize^2 + patsize^2 + 1);
Wb = [rand(2*nodes(1)*nodes(2),1)*2*r-r; zeros(nodes(1)+nodes(2), 1)];


%%
% * �����ֵ�Ĵ���ֵ���ݶ�ֵ
%
[cost grad] = cost_grad_func(Wb, pats, rho, lambda, beta, nodes(1), nodes(2), true);


%%
% * �ݶȼ�飬������������ݶ�ֵ�Ƿ������ֵ�ݶ�ֵ
%
if 0
	eps = 1e-4;
	nlen = length(grad);
	len1 = patsize^2 * hidsize^2;
	len2 = patsize^2 * hidsize^2;
	len3 = hidsize^2;
	len4 = patsize^2;
	grad_check = zeros(nlen, 1);
	W1 = reshape(Wb(1:len1), len3, []);
	W2 = reshape(Wb(len1+1:len1+len2), len4, []);
	b1 = reshape(Wb(len1+len2+1:len1+len2+len3), len3, []);
	b2 = reshape(Wb(len1+len2+len3+1:end), len4, []);
	for j = 1:nlen
		dels = zeros(nlen, 1);
		dels(j) = eps;
		delW1 = reshape(dels(1:len1), len3, []);
		delW2 = reshape(dels(len1+1:len1+len2), len4, []);
		delb1 = reshape(dels(len1+len2+1:len1+len2+len3), len3, []);
		delb2 = reshape(dels(len1+len2+len3+1:end), len4, []);
		
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
		
		grad_check(j) = (J_0 - J_1) / (2*eps);
		disp(sprintf('it:%d/%d',j,nlen));
	end

	disp( norm(grad-grad_check) / norm(grad+grad_check) );
end


%%
% * L-BFGS�㷨���ϡ���Ա������Ż�����
%
tic;
if 0
	addpath starter/
	addpath starter/minFunc
	options.Method = 'lbfgs';
	options.maxIter = 100;
	options.display = 'on';	
	[Wb, cost, ~, ~, lams] = minFunc( @(p) sparseAutoencoderCost(p, patsize^2, hidsize^2, lambda, rho, beta, pats), Wb, options);
else
	[Wb, cost] = mylbfgs( @(p1, p2) cost_grad_func(p1, pats, rho, lambda, beta, nodes(1), nodes(2), p2), Wb, 50, 20, 0.55, 50);
end
toc


%%
% * ��ʾW1��Ϣ
%
im = zeros(hidsize*(patsize+1)+1, hidsize*(patsize+1)+1);
W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
W1 = W1';
for i = 1:hidsize^2
	c = mod((i-1), hidsize)+1;
	r = floor((i-1) / hidsize)+1;
	im( (r-1)*(patsize+1)+2:r*(patsize+1), (c-1)*(patsize+1)+2:c*(patsize+1) ) = mat2gray(reshape(W1(:,i), patsize, []));
end
clf; imshow(im);

end


%%
% * �����
%
function y = sigmoid(x)
	y = 1 ./ (1+exp(-x));
end
function y = sigmoidInv(x)
	ex = exp(-x);	
	y = ex ./ (1+ex).^2;
end


%%
% * ����-�ݶȺ���
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

% ��������
Jc = sum(sum((a3-data).^2)) * 0.5 / m;
% Ȩ��˥����
Jw = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% ϡ��ͷ���
mrho = sum(a2,2) / m;
Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
% �ܴ���
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
% * L-BFGS+Armijoʵ�ֵ����Ż��㷨
%
function [Wb cost] = mylbfgs(func, Wb, maxiter, maxmk, basemk, maxm)
% 1. ��ֵ
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
	% 3. ������, Armijo׼��
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
	
	
	% 2. �ж�g_k
	g = gg;
	if g'*g < eps
		disp('g_k small enough');
		break;
	end
	
	
	% ����m��y��s
	y = g - g_old;
	s = alf*d;
	if size(Y,2) < maxm
		Y(:,end+1) = y;
		S(:,end+1) = s;
	else
		Y = [Y(:,2:end) y];
		S = [S(:,2:end) s];
	end
	
	% 4. two-loop��d_k+1
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
	
	% ����d
	d = r;
	g_old = g;
	c_old = cost;
	
	disp(sprintf('it:%d\t\tJ:%f\t\tstep:%f', k, cost, alf));
	err(end+1) = cost;
end
clf;
plot(err);
title('�в�-����');
disp('done');
end