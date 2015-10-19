function PCA_Whitening()


%% PCA
% ����������ȡ����Ҫ�ļ���������Ϊ���ɷ�������������Ͷ�䵽��Щ���ɷ���,���ܵõ�ѹ���ġ���ά�����ݣ���֮��Ͷ����ܻ�ԭ����ԭ���ݽ��Ƶ�Ч���� 
%
% $$X_{n\times m}$$
%
% ԭʼ�����þ��󣬴���n��������m��������
% 
% * �淶��
%
% $$X_{:,i} = X_{:,i} - \frac{1}{n}\sum_{j=1}^{n}X_{j,i}$$
%
% ��0�Ǳ���ģ���Ϊϣ���������ƽ��ֵ��Ӱ�죻����1ȴ��һ�����룬��Ϊ������ȻͼƬ����߱�ͳ�������ϵķ���һ���ԣ���������㹻�ḻ�Ļ���������һ���Ƕ�������PCA�������Ų����Ρ�
%
% * Э�������
% 
% $$\Sigma = \frac{1}{m}XX^T$$
%
% * ��������
% �������������ַ������:
% 
% $$\Sigma = USV^T$$
%
% $$\Sigma P= P\Lambda$$
%
% svd()��eig()�����ԶԳ��� $\Sigma$ �Ľ����һ����,Ψһ��ͬ����svd()������ֵ�������,��eig()��û�е�.
% 
% * ͶӰ�뷴ͶӰ
%
% $$Xp = U^TX$$
%
% $$X' = UU^TX$$
%
% * �׻�
% 
% �׻���Ŀ�ľ��ǽ�������������ԣ�����ʽ��˵������ϣ��ͨ���׻�����ʹ��ѧϰ�㷨����������������ʣ�
%
% (i)����֮������Խϵͣ�
%
% (ii)��������������ͬ�ķ��
%
% $$X_{PCAWhite}(i,:)=\frac{Xp(i,:)}{\sqrt{\lambda_i}}$$
%
% $$X_{ZCAWhite} = UX_{PCAWhite}$$
%
% ����˵,֮ǰ��0��1�Ƕ�ÿ������������������Ԥ����,�����Ƕ�ÿ�����������������Ĵ���.

clear;
clf;
clc;

if 0
	% ��ά������ʾ
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
	% ͼ����ʾ
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
		
	
	% �淶��
	X = bsxfun(@minus, pats, mean(pats));
	
	
	% ͶӰ��ͶӰ
	[n m] = size(X);	
	[U S V] = svd(X*X'/m);
	Xp = U'*X;
	Xr = U*Xp;
	
	
% 	% ��ʾͶӰ��������Э�������
% 	figure(1); clf;
% 	covmat = Xp*Xp'/m;
% 	imagesc(covmat);
% 	title('ͶӰ���Xp��Э�������');	
	
	
	
	% ��ʾԭ����
	figure(2);clf;
	subplot(1,3,1);	
	showsomething(X(:, randsamp));
	title('ԭ����');
	
	
	% ��ȡ90%����ֵ
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
	
	
	% ��ȡ99%����ֵ
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
	
	
	
	% �׻�
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