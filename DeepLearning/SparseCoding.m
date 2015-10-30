
%% 简介
% 稀疏编码，与稀疏性自编码器略不一样。不像稀疏自编码有明显的三层网络结构，然后逐层的（反向传播）求解权重、偏移参数。稀疏编码相当于直接构造第二层各系数，设定代价函数，然后求解各系数。
%
% * 代价函数
%
% $$J = \frac{\|As-x\|_2^2}{m} + \lambda\sqrt{s^2+\epsilon} + \gamma\|A\|_2^2$$
%
% $A$ 是权重矩阵，相当于一组超完备的基向量， $s$ 是特征矩阵，相当于基向量的系数矩阵， $x$ 是输入数据。
%
% 第一项 $\frac{\|As-x\|_2^2}{m}$ 是重构误差；第二项 $\lambda\sqrt{s^2+\epsilon}$ 是稀疏惩罚，加入 $\epsilon$ 是为了平滑L1范式，因为其在0点处不可微；第三项
% $\gamma\|A\|_2^2$ 是权重惩罚。
%
% 稀疏编码的目的就是求解 $A,s$ ，由于目标函数的不是一个凸函数，只能交替迭代，即固定 $A$ 求解 $s$，再固定 $s$ 求解 $A$。
%
% * 拓扑稀疏编码
%
% 按照上述代价函数和交替迭代，是能够求出 $A,s$，但是权重矩阵可视化后会看起来比较随机。拓扑稀疏编码就是试图让相邻的特征向量（基向量）看起来是相似的。
%
% $$J = \frac{\|As-x\|_2^2}{m} + \lambda\sqrt{Vs^2+\epsilon} + \gamma\|A\|_2^2$$
%
% 其中 $V$ 是分组矩阵，其决定将哪些特征分到一组， $V$ 里面存的就是这些特征所在位置的索引值，在生成 $V$
% 的时候可以使得一个局部范围内的特征分成一组，从而达到“拓扑”的效果。如果不需要分组，也就是不需要拓扑性， $V$ 可为单位阵。
%
% * 梯度公式
%
% $$\frac{\partial J}{\partial s} = \frac{2A^T(As-x)}{m} + \lambda\frac{V^T}{\sqrt{Vs^2+\epsilon}} \cdot s$$
%
% $$\frac{\partial J}{\partial A} = \frac{2(As-x)s^T}{m} + 2\gamma A$$
%
% * 注意事项
%
% 1. 如果将全部样本载入进去迭代求解，非常耗时，可行的办法是分批次迭代，每次随机选择小部分样本。
%
% 2. 分批次迭代时，每次更新完 $A$后，需要重新初始化 $s$ 再进行最优化，否则收敛效果会很差。
%
% $$s \leftarrow A^Tx$$
%
% $$s_{(i,:)} \leftarrow \frac{s_{(i,:)}}{\|A_{(:,i)}\|^2}$$
%
% 3. $s$ 需要迭代， $A$ 直接完型解。
%
% $$A = xs^T(ss^T + m\gamma I)^{-1}$$
%
% 4. $s$ 最好采用共轭梯度法迭代，L-BFGS迭代的不理想
%
%% 程序代码
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
% * 随机采样
patches = zeros(patwid^2, patnum);
[imghei imgwid imgnum] = size(images);
for i = 1:imgnum
	for j = 1:patnum / imgnum
		pos = randi([1, min(imgwid, imghei)-patwid+1], 2, 1);
		patches(:, (i-1)*patnum/imgnum + j) = reshape(images(pos(2):pos(2)+patwid-1, pos(1):pos(1)+patwid-1, i), [], 1);
	end
end
patches = bsxfun(@minus, patches, mean(patches));
% 此例只能归0，不能归1
% pstd = 3 * std(patches(:));
% patches = max(min(patches, pstd), -pstd) / pstd;
% patches = (patches + 1) * 0.4 + 0.1;
display_network(patches(:, 1:100));


%%
% * 测试代价和梯度计算是否准确
if SKIP < 1
	featnum = 5;
	patnum = 8;
	patches = patches(:, 1:patnum);
	V = eye(featnum);
	
	A = randn(patsize, featnum) * 0.005;
	s = randn(featnum, patnum) * 0.005;
	[cost, grad] = SparseCodingCostGrad(V, patches, A, s, gamma, lambda, epsilon, 1);
	
	% 检查WeightGrad
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
	
	% 检查FeatureGrad
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
	
	% 检查TopograhicFeature
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
% * 迭代最优化
A = rand(patsize, featnum);
s = rand(featnum, patnum_perbat);
donutDim = floor(sqrt(featnum));


% 生成分组矩阵，相邻的dxd区域
V = zeros(featnum, featnum);
for k = 0:featnum-1
	% 列主序，所以颠倒行列
	r = mod(k, donutDim);
	c = floor(k / donutDim);
	for j = r:r+poolDim-1
		for i = c:c+poolDim-1
			V( k+1, mod(j,donutDim)*donutDim+mod(i,donutDim)+1 ) = 1;
		end
	end
end


% 显示分组矩阵
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
	
	% 当前的代价值
	resid = A * s - subpatches;
	Jres = resid(:)'*resid(:) / patnum_perbat;	
	norm1 = sqrt( V*s.^2 + epsilon );
	Jsp = lambda * sum(norm1(:));	
	Jw = gamma * A(:)'*A(:);
	disp(sprintf('  %4d  %10.4f %10.4f %10.4f ', it, Jres, Jsp, Jw));
	
	
	% 确定一组新的随机样本
	indices = randperm(patnum, patnum_perbat);
	subpatches = patches(:, indices);
	
	% 初始化s
	s = A' * subpatches;
	normWM = sum(A .^ 2)';
	s = bsxfun(@rdivide, s, normWM);	
	
	% 固定A，迭代求解s
	options.maxIter = maxiter;
	options.Method = 'cg';
	options.display = 'off';
	options.verbose = 0;
	[s, cost] = minFunc( @(x) SparseCodingCostGrad(V, subpatches, A, x, gamma, lambda, epsilon, 2), s(:), options);
% 	[s, cost] = mylbfgs( @(p1, p2) SparseCodingCostGrad(V, subpatches, A, p1, gamma, lambda, epsilon, p2), s(:), maxiter, 20, 0.5, 100);	
	s = reshape(s, featnum, []);
		
	% 固定s，求解A。不需要迭代，A可以是完型解		
	A = (subpatches * s') / (s*s' + eyemat);

	% 检查A的计算是否正确，也就是grad是否足够小
% 	[cost grad] = SparseCodingCostGrad(V, subpatches, A, s, gamma, lambda, epsilon, 1);
% 	norm(grad)
	
	% 显示A		
	display_network(A);
	drawnow;
% 	pause;
end

end


%=============================================================================================================%


% 显示随机patch的部分原始信息
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
% * 代价-梯度函数
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
% * L-BFGS+Armijo实现的最优化算法
%
function [Wb cost] = mylbfgs(func, Wb, maxiter, maxmk, basemk, maxm)
% 1. 初值
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
	% 3. 线搜索, Armijo准则
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
	
% 	disp(sprintf('it:%d\t\tJ:%f\t\tstep:%f', k, cost, alf));
	err(end+1) = cost;
end
clf;
plot(err);
title('残差-迭代');
% disp('done');
end


