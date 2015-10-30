
%% ���
% ϡ����룬��ϡ�����Ա������Բ�һ��������ϡ���Ա��������Ե���������ṹ��Ȼ�����ģ����򴫲������Ȩ�ء�ƫ�Ʋ�����ϡ������൱��ֱ�ӹ���ڶ����ϵ�����趨���ۺ�����Ȼ������ϵ����
%
% * ���ۺ���
%
% $$J = \frac{\|As-x\|_2^2}{m} + \lambda\sqrt{s^2+\epsilon} + \gamma\|A\|_2^2$$
%
% $A$ ��Ȩ�ؾ����൱��һ�鳬�걸�Ļ������� $s$ �����������൱�ڻ�������ϵ������ $x$ ���������ݡ�
%
% ��һ�� $\frac{\|As-x\|_2^2}{m}$ ���ع����ڶ��� $\lambda\sqrt{s^2+\epsilon}$ ��ϡ��ͷ������� $\epsilon$ ��Ϊ��ƽ��L1��ʽ����Ϊ����0�㴦����΢��������
% $\gamma\|A\|_2^2$ ��Ȩ�سͷ���
%
% ϡ������Ŀ�ľ������ $A,s$ ������Ŀ�꺯���Ĳ���һ��͹������ֻ�ܽ�����������̶� $A$ ��� $s$���ٹ̶� $s$ ��� $A$��
%
% * ����ϡ�����
%
% �����������ۺ����ͽ�����������ܹ���� $A,s$������Ȩ�ؾ�����ӻ���ῴ�����Ƚ����������ϡ����������ͼ�����ڵ������������������������������Ƶġ�
%
% $$J = \frac{\|As-x\|_2^2}{m} + \lambda\sqrt{Vs^2+\epsilon} + \gamma\|A\|_2^2$$
%
% ���� $V$ �Ƿ���������������Щ�����ֵ�һ�飬 $V$ �����ľ�����Щ��������λ�õ�����ֵ�������� $V$
% ��ʱ�����ʹ��һ���ֲ���Χ�ڵ������ֳ�һ�飬�Ӷ��ﵽ�����ˡ���Ч�����������Ҫ���飬Ҳ���ǲ���Ҫ�����ԣ� $V$ ��Ϊ��λ��
%
% * �ݶȹ�ʽ
%
% $$\frac{\partial J}{\partial s} = \frac{2A^T(As-x)}{m} + \lambda\frac{V^T}{\sqrt{Vs^2+\epsilon}} \cdot s$$
%
% $$\frac{\partial J}{\partial A} = \frac{2(As-x)s^T}{m} + 2\gamma A$$
%
% * ע������
%
% 1. �����ȫ�����������ȥ������⣬�ǳ���ʱ�����еİ취�Ƿ����ε�����ÿ�����ѡ��С����������
%
% 2. �����ε���ʱ��ÿ�θ����� $A$����Ҫ���³�ʼ�� $s$ �ٽ������Ż�����������Ч����ܲ
%
% $$s \leftarrow A^Tx$$
%
% $$s_{(i,:)} \leftarrow \frac{s_{(i,:)}}{\|A_{(:,i)}\|^2}$$
%
% 3. $s$ ��Ҫ������ $A$ ֱ�����ͽ⡣
%
% $$A = xs^T(ss^T + m\gamma I)^{-1}$$
%
% 4. $s$ ��ò��ù����ݶȷ�������L-BFGS�����Ĳ�����
%
%% �������
%
function SparseCoding()
addpath('./starter/minFunc');
SKIP = 1;
patnum = 20000;
patnum_perbat = 2000;
poolDim = 4;
patwid = 8;
patsize = patwid^2;
gamma = 1e-2;
lambda = 5e-5;
epsilon = 1e-5;
featnum = 121;
maxiter = 20;	

images = load('./data/IMAGES.mat');
images = images.IMAGES;


%%
% * �������
patches = zeros(patwid^2, patnum);
[imghei imgwid imgnum] = size(images);
for i = 1:imgnum
	for j = 1:patnum / imgnum
		pos = randi([1, min(imgwid, imghei)-patwid+1], 2, 1);
		patches(:, (i-1)*patnum/imgnum + j) = reshape(images(pos(2):pos(2)+patwid-1, pos(1):pos(1)+patwid-1, i), [], 1);
	end
end
patches = bsxfun(@minus, patches, mean(patches));
% ����ֻ�ܹ�0�����ܹ�1
% pstd = 3 * std(patches(:));
% patches = max(min(patches, pstd), -pstd) / pstd;
% patches = (patches + 1) * 0.4 + 0.1;
display_network(patches(:, 1:100));


%%
% * ���Դ��ۺ��ݶȼ����Ƿ�׼ȷ
if SKIP < 1
	featnum = 5;
	patnum = 8;
	patches = patches(:, 1:patnum);
	V = eye(featnum);
	
	A = randn(patsize, featnum) * 0.005;
	s = randn(featnum, patnum) * 0.005;
	[cost, grad] = SparseCodingCostGrad(V, patches, A, s, gamma, lambda, epsilon, 1);
	
	% ���WeightGrad
	allcnt = numel(A);
	checkgrad = zeros(numel(grad),1);
	for k = 1:allcnt
		delW = A(:);
		delW(k) = delW(k)+1e-4;
		delW = reshape(delW, patsize, []);		
		diff = delW*s - patches;
		J0 = diff(:)'*diff(:) / patnum + gamma * delW(:)' * delW(:);
		
		delW = A(:);
		delW(k) = delW(k)-1e-4;
		delW = reshape(delW, patsize, []);
		diff = delW*s - patches;
		J1 = diff(:)'*diff(:) / patnum + gamma * delW(:)' * delW(:);
		checkgrad(k) = (J0-J1) / 2e-4;
	end
	if norm(checkgrad(:)-grad(:)) / norm(checkgrad(:)+grad(:)) > 1e-8
		disp('check weight grad failed');
	end
	
	% ���FeatureGrad
	epsilon = 1e-2;
	[cost, grad] = SparseCodingCostGrad(V, patches, A, s, gamma, lambda, epsilon, 2);
	allcnt = numel(s);
	checkgrad = zeros(numel(grad),1);
	for k = 1:allcnt
		delF = s(:);
		delF(k) = delF(k)+1e-4;
		delF = reshape(delF, featnum, []);		
		diff = A*delF - patches;
		norm1 = sqrt(delF.^2 + epsilon);
		J0 = diff(:)'*diff(:) / patnum + lambda * sum(norm1(:));
		
		delF = s(:);
		delF(k) = delF(k)-1e-4;
		delF = reshape(delF, featnum, []);		
		diff = A*delF - patches;
		norm1 = sqrt(delF.^2 + epsilon);
		J1 = diff(:)'*diff(:) / patnum + lambda * sum(norm1(:));		
		
 		checkgrad(k) = (J0-J1) / 2e-4;
	end
	if norm(checkgrad(:)-grad(:)) / norm(checkgrad(:)+grad(:))  > 1e-8
		disp('check weight grad failed');
	end
	
	% ���TopograhicFeature
	groupmat = rand(100, featnum);
	[cost, grad] = SparseCodingCostGrad(groupmat, patches, A, s, gamma, lambda, epsilon, 2);
	allcnt = numel(s);
	checkgrad = zeros(numel(grad),1);
	for k = 1:allcnt
		delF = s(:);
		delF(k) = delF(k)+1e-4;
		delF = reshape(delF, featnum, []);		
		diff = A*delF - patches;
		norm1 = sqrt(groupmat*delF.^2 + epsilon);
		J0 = diff(:)'*diff(:) / patnum + lambda * sum(norm1(:));
		
		delF = s(:);
		delF(k) = delF(k)-1e-4;
		delF = reshape(delF, featnum, []);		
		diff = A*delF - patches;
		norm1 = sqrt(groupmat*delF.^2 + epsilon);
		J1 = diff(:)'*diff(:) / patnum + lambda * sum(norm1(:));		
		
		checkgrad(k) = (J0-J1) / 2e-4;
	end
	if norm(checkgrad(:)-grad(:)) / norm(checkgrad(:)+grad(:))  > 1e-8
		disp('check weight grad failed');
	end
end


%%
% * �������Ż�
A = rand(patsize, featnum);
s = rand(featnum, patnum_perbat);
donutDim = floor(sqrt(featnum));


% ���ɷ���������ڵ�dxd����
V = zeros(featnum, featnum);
for k = 0:featnum-1
	% ���������Եߵ�����
	r = mod(k, donutDim);
	c = floor(k / donutDim);
	for j = r:r+poolDim-1
		for i = c:c+poolDim-1
			V( k+1, mod(j,donutDim)*donutDim+mod(i,donutDim)+1 ) = 1;
		end
	end
end


% ��ʾ�������
clf;
for k = 1:featnum
	c = mod((k-1), donutDim) + 1;
	r = floor((k-1) / donutDim) + 1;
	subplot(donutDim, donutDim, k);
	sV = reshape(V(k,:), donutDim, donutDim);
	imshow(sV);
end


if isequal(questdlg('Initialize grouping matrix for topographic or non-topographic sparse coding?', 'Topographic/non-topographic?', 'Non-topographic', 'Topographic', 'Non-topographic'), 'Non-topographic')
    V = eye(featnum);
end

indices = randperm(patnum, patnum_perbat);
subpatches = patches(:, indices); 

eyemat = patnum_perbat * gamma * eye(featnum);

for it = 1:200
	
	% ��ǰ�Ĵ���ֵ
	resid = A * s - subpatches;
	Jres = resid(:)'*resid(:) / patnum_perbat;	
	norm1 = sqrt( V*s.^2 + epsilon );
	Jsp = lambda * sum(norm1(:));	
	Jw = gamma * A(:)'*A(:);
	disp(sprintf('  %4d  %10.4f %10.4f %10.4f ', it, Jres, Jsp, Jw));
	
	
	% ȷ��һ���µ��������
	indices = randperm(patnum, patnum_perbat);
	subpatches = patches(:, indices);
	
	% ��ʼ��s
	s = A' * subpatches;
	normWM = sum(A .^ 2)';
	s = bsxfun(@rdivide, s, normWM);	
	
	% �̶�A���������s
	options.maxIter = maxiter;
	options.Method = 'cg';
	options.display = 'off';
	options.verbose = 0;
	[s, cost] = minFunc( @(x) SparseCodingCostGrad(V, subpatches, A, x, gamma, lambda, epsilon, 2), s(:), options);
% 	[s, cost] = mylbfgs( @(p1, p2) SparseCodingCostGrad(V, subpatches, A, p1, gamma, lambda, epsilon, p2), s(:), maxiter, 20, 0.5, 100);	
	s = reshape(s, featnum, []);
		
	% �̶�s�����A������Ҫ������A���������ͽ�		
	A = (subpatches * s') / (s*s' + eyemat);

	% ���A�ļ����Ƿ���ȷ��Ҳ����grad�Ƿ��㹻С
% 	[cost grad] = SparseCodingCostGrad(V, subpatches, A, s, gamma, lambda, epsilon, 1);
% 	norm(grad)
	
	% ��ʾA		
	display_network(A);
	drawnow;
% 	pause;
end

end


%=============================================================================================================%


% ��ʾ���patch�Ĳ���ԭʼ��Ϣ
function display_network(imgs)
cols = ceil(sqrt(size(imgs,2)));
rows = ceil(size(imgs,2) / cols);
imgwid = round(sqrt(size(imgs,1)));
imghei = round(size(imgs,1)/imgwid);
im = zeros(rows*(imgwid+1)+1, cols*(imgwid+1)+1);
for i = 1:size(imgs,2)
	c = mod((i-1), cols)+1;
	r = floor((i-1) / cols)+1;
	tmp = mat2gray(reshape(imgs(:, i), imghei, imgwid));
	im( (r-1)*(imgwid+1)+2:r*(imgwid+1), (c-1)*(imgwid+1)+2:c*(imgwid+1) ) = tmp;
end
clf; imagesc(im); 
axis equal;
end


%%
% * ����-�ݶȺ���
%
% $$J = J_0 + J_W + J_{sp}$$
%
function [cost grad] = SparseCodingCostGrad(V, X, A, s, gamma, lambda, epsilon, calcgrad)

m = size(X,2);
s = reshape(s, [], m);
diff = A*s - X;
J0 = diff(:)'*diff(:) / m;
% Jw = A(:)' * A(:);
norm1 = sqrt(V*s.^2 + epsilon);
Jsp = sum(norm1(:));
% cost = J0 + gamma*Jw + lambda*Jsp;
cost = J0 + lambda*Jsp;

if calcgrad == 1
	grad = 2*(A*s-X)*s'/m + 2*gamma*A;
elseif calcgrad == 2
	grad = 2*A'*(A*s-X) /m + lambda*(V'*(1./norm1).*s);
end
grad = grad(:);

end


%%
% * L-BFGS+Armijoʵ�ֵ����Ż��㷨
%
function [Wb cost] = mylbfgs(func, Wb, maxiter, maxmk, basemk, maxm)
% 1. ��ֵ
[cost grad] = func(Wb, 2);
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
		[cost gg] = func(Wb+alf*d, 2);
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
	
% 	disp(sprintf('it:%d\t\tJ:%f\t\tstep:%f', k, cost, alf));
	err(end+1) = cost;
end
clf;
plot(err);
title('�в�-����');
% disp('done');
end


