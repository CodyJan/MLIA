
%% ���
% ��������⣬֮ǰ��Logistic���޷�����ģ����������ֽ��������
%
% 1.SoftMax�㷨���൱��Logistic���ƹ㡣SoftMax�ڹ�ʽ���Ƶ�����������Logistic�кܴ����ƣ����㷨����Ľ���ǻ��⡢�����ģ������ж�ͼ�������ּ���
% 
% 2. ʹ��k��Logistic�ع飬Ҳ�������ÿ���ع�һ�Ρ��Ƚ��ʺϷ������й��������⣬����ͼƬ�����ĸ����ɡ����
%
%% ���ۺ���
% �ȴ������Ȼ˵��
%
% $$L = \prod_{i=1}^{m}\prod_{j=1}^{k} p_{i,j}^{(y^{(i)}==j)}$$
%
% $$\log L = \sum_{i=1}^{m}\sum_{j=1}^{k} (y^{(i)}==j) \log p\left(y^{(i)}=j|x^{(i)};W\right)$$
%
% $$p\left(y^{(i)}=j|x^{(i)};W\right) = \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}}$$
%
% ����SoftMax��С���ۺ���
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\sum_{j=1}^{k} (y^{(i)}==j) \log \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}}$$
% 
% �ٿ�һ��Logistic�ع飬��k=2�����
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\left[ y^{(i)}\log \frac{e^{W_1x^{(i)}}}{e^{W_1x^{(i)}}+e^{W_2x^{(i)}}} + \left(1-y^{(i)}\right)\log \frac{e^{W_2x^{(i)}}}{e^{W_1x^{(i)}}+e^{W_2x^{(i)}}} \right]$$
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\left[ y^{(i)}\log \frac{1}{1+e^{-(W_1-W_2)x^{(i)}}} +\left(1-y^{(i)}\right)\log \left(1-\frac{1}{1+e^{-(W_1-W_2)x^{(i)}}}\right) \right]$$
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)}\log \frac{1}{1+e^{-Wx^{(i)}}} +\left(1-y^{(i)}\right)\log \left(1-\frac{1}{1+e^{-Wx^{(i)}}}\right)\right]$$
%
% �������÷���Logistic�Ĺ�ʽ��ͬʱ��Ҳ���������ΪʲôSoftMax��Logistic�������ֲ�ͬ�ļ��躯����
%
% $$p_{soft} = \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}}$$
%
% $$p_{log} = \frac{1}{1+e^{-Wx^{(i)}}}$$ 
%
%% Ȩ��˥��
% SoftMax��һ���ص㣬���� $W$���мӼ�һ��������������Ӱ��
% $p_{soft}$�Ľ�������Կ��ܻ�ʹ��Сֵ�ĽⲻΨһ�����Կ��Լ�һ��Ȩ��˥������ֹ�������⡣��Ȩ��˥���е�����֮ǰ�������
%
% $$J = -\frac{1}{m} \sum_{i=1}^{m}\sum_{j=1}^{k} (y^{(i)}==j) \log \frac{e^{W_jx^{(i)}}}{\sum_{j=1}^{k}e^{W_jx^{(i)}}} + \frac{\lambda}{2}\sum_{i=1}^{k}\sum_{j=1}^{n}W_{ij}^2$$
%
%% ��������
% �����Ͽɲ����ݶ��½�����ţ�����ַ�����ͬʱҲ��������Ҫ��һ�׻��Ƕ���ƫ�������⡣
%
% һ��ƫ����
% 
% $$\frac{\partial J}{\partial W} = -\frac{1}{m}\sum_{i=1}^{m} \left[x^{(i)}\left((y^{(i)}==j)-p_{ij}\right)\right] + \lambda W$$
%
% $$\frac{\partial J}{\partial W} = -\frac{1}{m}\left(Y-P\right)X^T + \lambda W$$
%
%% �������
%

function SoftMax()

clear;
clf;
clc;


K = 10;				% �������
lambda = 1e-4;		% ˥��Ȩ��

% MNIST Dataset: images and labels
load('./data/softmax_data.mat');
imgs = softmax_data.imgs;
labs = softmax_data.labs;
labs(labs==0) = 10;
imgs_test = softmax_data.imgs_test;
labs_test = softmax_data.labs_test;
labs_test(labs_test==0) = 10;
clear softmax_data;

[n m] = size(imgs);
wid = round(sqrt(n));
hei = round(n / wid);

X = imgs;
% ���W����
W = 0.005 * randn(K*n, 1);
% �γ����Ķ�ά���룬���������������
LABS = full(sparse(labs, 1:m, 1));
% ��ֵ������ۺ��ݶ�
[cost grad] = cost_grad_func(W, X, LABS, lambda, true);

if 0
%%
% * ʹ��minFunc����
	addpath starter/
	addpath starter/minFunc
	options.Method = 'lbfgs';
	options.maxIter = 50;
	options.display = 'on';
	tic;
	[W, cost, ~, ~, lams] = minFunc( @(p) cost_grad_func(p, X, LABS, lambda, true), W, options);
	toc	
else
%%
% * L-BFGS+Armijo

	% 1. ��ֵ
	eps = 1e-8;
	m = 100;
	Y = zeros(size(W,1), 0);
	S = zeros(size(W,1), 0);
	g = grad;
	d = -g;
	err = [cost];
	sigma = 0.4;
	c_old = cost;
	g_old = g;
	tic;
	for k = 1:50
		% 3. ������, Armijo׼��
		for mk = 0:20
			alf = 0.8^mk;
			[cost g] = cost_grad_func(W+alf*d, X, LABS, lambda, 1);
			if cost < c_old + sigma*alf*g'*d
				break;
			end
		end
		if mk>=20
			disp('line search failed');
			alf = 1;
		end
		W = W + alf*d;
		
		
		% 2. �ж�g_k
		if g'*g < eps
			disp('g_k small enough');
			break;
		end
		
		
		% ����m��y��s
		y = g - g_old;
		s = alf*d;
		if size(Y,2) < m
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
		
		disp(sprintf('J:%f\t\tit:%d', cost, k));
		err(end+1) = cost;
	end
	toc	
end


%%
% * ����
W = reshape(W, K, []);
[~, labs_pred] = max(W*imgs_test);
disp('correct rate:');
sum(labs_pred' == labs_test) / length(labs_test)

end



function [cost grad] = cost_grad_func(W, X, Y, lambda, wat)

% ���ۡ��ݶ�
[n m] = size(X);
W = reshape(W, size(Y,1), []);
Z = W*X;
% Z = bsxfun(@minus, Z, max(Z, [], 1));
Z = exp(Z);
P = bsxfun(@rdivide, Z, sum(Z));
cost = -Y(:)'*log(P(:))/m + 0.5*lambda * W(:)'*W(:);

if nargin > 4
	grad = -(Y - P)*X'/m + lambda * W;
	grad = grad(:);
end

end



