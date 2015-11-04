
%% ���
% �����ɷַ�����ICA���൱��ϡ������һ�����Σ�ϡ��������ǻ�ȡһ�����걸�� $A$
% ��Ҳ������ζ�������������ģ�����һ�������޹ء�ICA����ҪѰ��һ���׼������ $W$ ��
%
% * Ԥ����
%
% ��Ҫ���й�0��ZCA�׻�����ZCA�׻�ʹ�� $\epsilon=0$ ��
%
% * ���ۺ���
%
% ICA��Ҫ�������㣺 ����Ҫ����ϡ�裻 $W$ �Ǳ�׼��������
%
% $$J = \|W^TWx-x\|_2^2 + \|Wx\|_1$$
%
% ǰһ�����ع����ۣ���һ����ϡ������ǵ�ϡ����������΢��������Ҫ��һЩƽ������
%
% $$J = \frac{1}{m}\sum\sum \left(W^TWx-x\right)^2 + \frac{1}{m}\sum\sum \sqrt{(Wx)^2+\epsilon}$$
%
% ��ϡ�������ʵ�ǳ����ƣ��൱�� $A=W^T,s=Wx$��ֻ����ICA����һЩ���ơ�
%
% * �ݶȼ���
%
% $$\frac{\partial J}{\partial W} = \frac{2}{m}\left(W\left(W^TWx-x\right)x^T + Wx\left(W^TWx-x\right)^T\right) + \frac{1}{m}\left(\frac{1}{\sqrt{(Wx)^2+\epsilon}}\bullet Wx\right) x^T$$
%
% * ����ICA
%
% ������ϡ��������ƣ�ֻ��Ҫ�滻���ۺ����ڶ���Ϊ $\frac{1}{m}\sum\sum \sqrt{V(Wx)^2+\epsilon}$��
%
%% �������
%
function ICA()

SKIP = 1;
patnum = 20000;
featnum = 121;
patwid = 8;
patsize = patwid * patwid * 3;
epsilon = 1e-6;							% L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon)


%%
% * ��ȡ��������ʾ����
patches = load('./data/stlSampledPatches.mat');
patches = patches.patches(:, 1:patnum);
showColorInfo(patches(:,1:100));


%%
% * ��0�� ZCA�׻�
patches = patches / 255;
patches = bsxfun(@minus, patches, mean(patches, 2));
sigma = patches * patches';
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s))) * u';
% ZCAWhite = u * diag(1 ./ sqrt(diag(s) + 0.1)) * u';
patches = ZCAWhite * patches;


%%
% * �����ۡ��ݶȼ���
if SKIP < 1
	featnum = 8;
	patnum = 100;	
	patsize = 192;
	patches = patches(1:patsize, 1:patnum);
	W = rand(featnum, patsize);
	
	% �ݶȼ��
	[cost, grad] = orthonormalICACost(W, patsize, featnum, patches, epsilon);
	allcnt = numel(W);
	checkgrad = zeros(numel(grad),1);
	for k = 1:allcnt
		delW = W(:);
		delW(k) = delW(k)+1e-4;
		delW = reshape(delW, [], patsize);
		Wx = delW*patches;
		WWx = delW'*Wx - patches;
		Wx_l1 = sqrt(Wx.^2+epsilon);
		J0 = sum(Wx_l1(:))/patnum +  WWx(:)'*WWx(:)/patnum;
		
		
		
		delW = W(:);
		delW(k) = delW(k)-1e-4;
		delW = reshape(delW, [], patsize);
		Wx = delW*patches;
		WWx = delW'*Wx - patches;
		Wx_l1 = sqrt(Wx.^2+epsilon);
		J1 = sum(Wx_l1(:))/patnum +  WWx(:)'*WWx(:)/patnum;
		checkgrad(k) = (J0-J1) / 2e-4;
	end
	if norm(checkgrad(:)-grad(:)) / norm(checkgrad(:)+grad(:)) > 1e-8
		disp('check weight grad failed');
	end
end

%%
% * �ݶȵ���

% ��ʼ����
alpha = 0.5;
t = 0.02;
lastCost = 1e40;
eyef = eye(featnum);

W = rand(featnum, patsize);
[cost grad] = orthonormalICACost(W, patsize, featnum, patches, epsilon);

% Ϊ�˱�֤WΪ������������ʹ������Ŀɱ�ѧϰ�ʵ��ݶ��½�����
tic();
for iteration = 1:20000	
	
	newCost = Inf;
	gtg = grad(:)'*grad(:);
	
	cnt = 1;
	while 1
		
		cW = W - alpha * grad;			
		cW = (cW*cW')^(-0.5) * cW;
		
		% ���cW�Ƿ�����
% 		temp = cW * cW' - eyef;
% 		assert(sum(temp(:).^2) < 1e-23, 'considerWeightMatrix does not satisfy WW^T = I. Check your projection again');
		
		[newCost, newGrad] = orthonormalICACost(cW, patsize, featnum, patches, epsilon);
		if newCost > lastCost - alpha * t * gtg
			t = 0.8 * t;
		else
			break;
		end
		cnt = cnt + 1;
	end
	
	lastCost = newCost;
	W = cW;
	
	fprintf('  %9d  %10.4f  %10.4f  %d\n', iteration, newCost, t, cnt);
	
	t = 1.2 * t;
	
	cost = newCost;
	grad = newGrad;
	
	% ��ʾ
	if mod(iteration, 100) == 0		
		showColorInfo(W');
		drawnow;
	end	
end
toc

showColorInfo(W');
end




%%
% * ��ʾ����
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
% * ICA���ۡ��ݶ�
function [cost, grad] = orthonormalICACost(W, patsize, featnum, patches, epsilon)
	m = size(patches, 2);
	
	Wx = W*patches;
	WWx = W'*Wx - patches;
	Wx_l1 = sqrt(Wx.^2+epsilon);
	cost = (sum(Wx_l1(:)) +  WWx(:)'*WWx(:) ) / m;	
		
	grad = ( (2*W*WWx+Wx./Wx_l1)*patches' + 2*Wx*WWx' )/m;
% 	grad = (( W*WWx*patches' + Wx*WWx' )*2 + (Wx./Wx_l1)*patches' )/ m;	
end

