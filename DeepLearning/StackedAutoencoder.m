
%% ���
% ������ʾջʽ�Ա����㷨����һ����������ṹ����ʶ����д����ΪĿ�ꡣ
%
% ջʽ�Ա��������Ƕ��ϡ���Ա�������ɵ������磬ǰһ���Ա������������Ϊ��һ������룬����ټ���һ��SoftMax�ع�������ع������������
% �ɴ��������������������õ�ÿ����Ա�����������������һ��SoftMax�ع顣
%
% ÿ����Ա���������ڶ����������ٽ���һ�Ρ�ѹ����������Щ��ѹ����������ͨ����ԭ�����������Ǿ������Ƶġ�
% Ҳ����ջʽ�Ա�����ÿ�㶼��Ҫִ��һ��"SparseAutoencode"����ֻ���� $W^{(1)}$ ��Ϣ��
%
% ������4������磬ͨ��ջʽ�Ա����㷨�õ�����ȷ��ֻ��89%����û��SoftMaxֱ�ӻع��������ȷ�ʸߣ����ԭ�������Ĭ�ϵ�ϵ������һ������������˵4������Ӧ�ñ�3���������һ�㡣
%
% ջʽ�Ա�����Ҫһ������ġ�΢��������߾��ȣ���ν��΢���������������Ľ����Ϊ��ʼֵ���ٽ����в㴮�������з��򴫲��㷨�������������ϴ�
%
%% �������
%
function StackedAutoencoder()
return
nodes = [28*28 200 200 28*28];		% ÿ��ڵ���
rho = 0.1;							% ϡ����
lambda = 3e-3;						% weight decay parameter       
beta = 3;							% weight of sparsity penalty term  
maxiter = 100;

%%
% * ��ȡ����
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
% * �ڶ�����Ա�����
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
% * �ڶ�������
%
W1 = reshape(Wb1(1:nodes(1)*nodes(2)), nodes(2), []);
b1 = Wb1(2*nodes(1)*nodes(2)+1:2*nodes(1)*nodes(2)+nodes(2));
z2 = bsxfun(@plus, W1*train_imgs, b1);
train_feat = sigmoid(z2);
z2 = bsxfun(@plus, W1*test_imgs, b1);
test_feat = sigmoid(z2);


%%
% * ��������Ա�����
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
% * ����������
%
W1 = reshape(Wb2(1:nodes(2)*nodes(3)), nodes(3), []);
b1 = Wb2(2*nodes(2)*nodes(3)+1:2*nodes(2)*nodes(3)+nodes(3));
z2 = bsxfun(@plus, W1*train_feat, b1);
train_feat = sigmoid(z2);
z2 = bsxfun(@plus, W1*test_feat, b1);
test_feat = sigmoid(z2);


%%
% * ��������SoftMax�ع飬������
%
SoftMax(train_feat, train_labs, test_feat, test_labs, maxiter);


%%
% * �����в㴮�����ڽ���һ�η��򴫲��㷨
%

end


%=================================================================================================================%


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
% * ����-�ݶȺ������Ա�����
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