function PCA_Whitening()


%% PCA
% 从数据中提取最主要的几个方向，作为主成分向量。将数据投射到这些主成分上,即能得到压缩的、降维的数据；反之反投射后能还原成与原数据近似的效果。 
%
% $$X_{n\times m}$$
%
% 原始数据用矩阵，代表n个特征，m个样本。
% 
% * 规范化
%
% $$X_{:,i} = X_{:,i} - \frac{1}{n}\sum_{j=1}^{n}X_{j,i}$$
%
% 归0是必须的，因为希望结果不受平均值的影响；但归1却不一定必须，因为比如自然图片本身具备统计意义上的方差一致性（如果样本足够丰富的话），另外一个角度来看，PCA具有缩放不变形。
%
% * 协方差矩阵
% 
% $$\Sigma = \frac{1}{m}XX^T$$
%
% * 特征向量
% 特征向量有两种方法获得:
% 
% $$\Sigma = USV^T$$
%
% $$\Sigma P= P\Lambda$$
%
% svd()和eig()方法对对称阵 $\Sigma$ 的结果是一样的,唯一不同的是svd()的特征值是排序的,而eig()是没有的.
% 
% * 投影与反投影
%
% $$Xp = U^TX$$
%
% $$X' = UU^TX$$
%
% * 白化
% 
% 白化的目的就是降低输入的冗余性；更正式的说，我们希望通过白化过程使得学习算法的输入具有如下性质：
%
% (i)特征之间相关性较低；
%
% (ii)所有特征具有相同的方差。
%
% $$X_{PCAWhite}(i,:)=\frac{Xp(i,:)}{\sqrt{\lambda_i}}$$
%
% $$X_{ZCAWhite} = UX_{PCAWhite}$$
%
% 或者说,之前归0归1是对每个样本的特征间做的预处理,这里是对每个特征的样本间做的处理.

clear;
clf;
clc;

if 0
	% 二维数组演示
	subplot(2,2,1);
	title('raw data');
	hold on;
	x = load('./pca_2d/pcaData.txt');
	[n m] = size(x);
	plot(x(1,:), x(2,:), 'o');
	[U S V] = svd(x*x'/m);
	% [U S V] = svd(x*x')
	% [U S V] = svd(x);
	% [U S] = eig(x*x')
	plot([0 U(1,1)], [0 U(2,1)], 'g');
	plot([0 U(1,2)], [0 U(2,2)], 'r');
	
	
	subplot(2,2,2);
	title('down dim');
	hold on;
	xp = U(:,1)'*x;
	xr = U(:,1)*xp;
	plot(x(1,:), x(2,:), 'o');
	plot(xr(1,:), xr(2,:), 'r*');
	
	
	subplot(2,2,3);
	title('PCAwhite');
	hold on;
	epsilon = 1e-5;
	xw = diag(1./sqrt(diag(S)+epsilon))*U'*x;
	xp = U'*x;
	plot(xp(1,:),xp(2,:),'o')
	plot(xw(1,:), xw(2,:), 'r*');
	
	
	subplot(2,2,4);
	title('ZCA');
	hold on;
	xr = U*diag(1./sqrt(diag(S)+epsilon))*U'*x;
	plot(x(1,:), x(2,:), 'o');
	plot(xr(1,:), xr(2,:), 'r*');
	
end








if 1
	% 图像演示
	patsize = 12;
	patnum = 10000;
	epsilon = 0.1;
	randsamp = randi(patnum, 90, 1);
	
	load('./pca_exercise/IMAGES_RAW');
	IMGS = IMAGESr;
	[hei wid cnt] = size(IMGS);	
	pats = zeros(patsize^2, patnum);
	for i = 1:cnt
		for j = 1:patnum/cnt			
			pos = randi([1, min(wid,hei)-patsize+1], 2, 1);
			pats(:,(i-1)*patnum/cnt + j) = reshape(IMGS(pos(2):pos(2)+patsize-1, pos(1):pos(1)+patsize-1, i), [], 1);
		end
	end
		
	
	% 规范化
	X = bsxfun(@minus, pats, mean(pats));
	
	
	% 投影反投影
	[n m] = size(X);	
	[U S V] = svd(X*X'/m);
	Xp = U'*X;
	Xr = U*Xp;
	
	
% 	% 显示投影后向量的协方差矩阵
% 	figure(1); clf;
% 	covmat = Xp*Xp'/m;
% 	imagesc(covmat);
% 	title('投影后的Xp的协方差矩阵');	
	
	
	
	% 显示原数据
	figure(2);clf;
	subplot(1,3,1);	
	showsomething(X(:, randsamp));
	title('原数据');
	
	
	% 截取90%特征值
	subplot(1,3,2);		
	sumval = sum(diag(S)) * 0.90;
	sumtmp = 0;
	for k = 1:n
		sumtmp = sumtmp + S(k,k);
		if sumtmp >= sumval
			break;
		end
	end		
	Xp = U(:,1:k)'*X;
	Xr = U(:,1:k) * Xp;
	showsomething(Xr(:, randsamp));	
	title('90%PCA');
	
	
	% 截取99%特征值
	subplot(1,3,3);		
	sumval = sum(diag(S)) * 0.99;
	sumtmp = 0;
	for k = 1:n
		sumtmp = sumtmp + S(k,k);
		if sumtmp >= sumval
			break;
		end
	end		
	Xp = U(:,1:k)'*X;
	Xr = U(:,1:k) * Xp;
	showsomething(Xr(:, randsamp));
	title('99%PCA');
	
	
	
	% 白化
	figure(3); clf;
	Xp = diag(1./sqrt(diag(S)+epsilon))' * U' * X;
	Xr = U*Xp;
	subplot(1,2,1);
	showsomething(Xp(:, randsamp));
	title('PCA Whitening');
	
	subplot(1,2,2);
	showsomething(Xr(:, randsamp));
	title('ZCA Whitening');
	
end

end


function showsomething(A)

patsize = sqrt(size(A,1));
cols = ceil(sqrt(size(A,2)));
rows = ceil(size(A,2) / cols);
im = zeros(rows*(patsize+1)+1, cols*(patsize+1)+1);
for i = 1:size(A,2)
	c = mod((i-1), cols)+1;
	r = floor((i-1) / cols)+1;
	im( (r-1)*(patsize+1)+2:r*(patsize+1), (c-1)*(patsize+1)+2:c*(patsize+1) ) = mat2gray(reshape(A(:, i), patsize, patsize));
end
imshow(im);

end