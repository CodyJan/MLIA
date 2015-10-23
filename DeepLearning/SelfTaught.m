
%% ���
% ��ʵ�еĺܶ��������ͨ������û�б�ǵ�����ȥ�����޼ල������ѧϰ��Ȼ��������Щ���������������б�ǵ������Ͻ����мල��ѧϰ��
%
% ��ϵʵ������Ļ�������һ������ʶ������⣬��������Ҫ�������ú������������ѡ�������أ������޼ල������ѧϰ��ϡ���Ա��룩���õ����ز�����������Ϊ������д����Ե�������
% �޼ලѧϰ����㵽���ز��ת�����̣�Ҳ�������������ɷ�����
% ���������������б�ǵ�������ȡ���������ٽ����мල��ѧϰ(SoftMax�����Իع��)�����ܵõ�һ���Ϻõ�ʶ�𷽷���
%
% �����ַ��������޼ලѧϰ��һ�ֽ���ѧϰ�����ޱ�ǡ��б�ǵ�����û������Ҫ����һ�ֽа�ලѧϰ��Ҫ���ޱ�ǡ��б�ǵ����ݷ���ͬ���ķֲ���Ҳ����˵ǰ�߲�����������Դ������Ҫ�����ݶ��ǹ̶��ļ��࣬��������һ�������ơ�
%
% ����������д����ʶ�𣬴�����5-9��ͼ���н����޼ල������ѧϰ��ȷ����������ȡ������Ȼ��ʹ������0-4��ͼ������мල��ѧϰ������SoftMax�������������в��ԡ�
%
% �������������ѧϰ�㷨�ܽ�׼ȷ����ߵ�98%�����������ȡ����ֱ��SoftMaxѵ��ֻ��96.7%
%
%% �������
%
function SelfTaught()

clear;clc;clf;

rho = 0.1;							% ϡ��ֵ
K = 10;								% �����
nodes = [28*28 200 28*28];			% ÿ��ڵ���
lambda = 3e-3;
beta = 3;


%%
% * ��ȡ����
%
load('./data/softmax_data.mat');
labset = find(softmax_data.labs <= 4);
unlabset = find(softmax_data.labs >=5);
% �ޱ������
train_udata = softmax_data.imgs(:, unlabset);
% �б������
num = round(length(labset)/2);
train_ldata = softmax_data.imgs(:, labset(1:num));
train_labs = softmax_data.labs(labset(1:num));
% ��������
test_ldata = softmax_data.imgs(:, labset(num+1:end));
test_labs = softmax_data.labs(labset(num+1:end));
clear softmax_data;


%%
% * ��������� $W,b$ �� $W$ ���� $r=\pm\sqrt{6/(n_{in}+n_{out}+1)}$ ��Χ֮���α�����. ���������������ֲ�
%
r  = sqrt(6) / sqrt(nodes(1) + nodes(2) + 1);
Wb = [rand(2*nodes(1)*nodes(2),1)*2*r-r; zeros(nodes(1)+nodes(2), 1)];


%%
% * LBFGS�㷨�����Ż�����޼ල����ѧϰ
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
% * ��ʾ�޼ල����ѧϰ�� $W^{(1)}$ ��Ϣ
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
% * ���������������ɷ��������б��������������������ȡ����
%
W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
b1 = Wb(2*nodes(1)*nodes(2)+1:2*nodes(1)*nodes(2)+nodes(2));
z2 = bsxfun(@plus, W1*train_ldata, b1);
train_feat = sigmoid(z2);
z2 = bsxfun(@plus, W1*test_ldata, b1);
test_feat = sigmoid(z2);


%%
% * ����SoftMax�㷨����ѵ����������������Ϊ����㣬��������Ϊ����㡣
% * ��������������������ۡ�
%
SoftMax(train_feat, train_labs, test_feat, test_labs);

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

% ��������
Jc = sum(sum((a3-X).^2)) * 0.5 / m;
% Ȩ��˥����
Jw = 0.5 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
% ϡ��ͷ���
mrho = sum(a2,2) / m;
Jsp = sum(rho*log(rho./mrho) + (1-rho)*log((1-rho)./(1-mrho)));
% �ܴ���
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