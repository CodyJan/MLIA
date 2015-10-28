
%% 简介
% 使用卷积核的神经网络。当输入图像尺寸较大，冗余数据较多，导致计算非常困难。根据人眼识别图像的经验，可使用卷积核+池化将一定区域的原图像转成一个精简的特征，再用这个特征进行训练。
%
% * 计算卷积核：在实施上，首先要确定卷积核，卷积核的计算过程相当于是一次压缩的过程，先从原始图片上获取一定数目的8*8*3的图像块，使用稀疏自编码器得到权重、偏移参数
% $W,b$ 。由于需要做ZCA白化处理，所以最后需要采用线性解码器进行训练，可以调用之前的LinearDecoder函数。
% 
% * 卷积处理：利用LinearDecoder训练得到 $W^{(1)}, b^{(1)}$，对所有原始图片逐块的进行卷积处理，在卷积之前，还需要考虑归0和白化的预处理，可以使用前一步计算出的均值和ZCAWhitening参数。
%
% * 池化处理：经过卷积后，数据量虽然有降低，但是仍然较大，还需要进一步的进行池化操作。
% 所谓池化既是在一个更大的互不重叠的范围内（卷积时是有重叠的）获取一个平均的/最大的特征值作为该pool的代表。这样的池化是会具有一定范围的平移不变性。
%
% * 回归、测试：池化后的特征已经大大降低了训练的数据量，利用这些特征进行SoftMax或其他回归。最后测试数据显示，正确率只有80.28%。
%
%% 程序代码

function CNN()

hidsize = 400;
patwid = 8;
nodes = [patwid^2*3, hidsize, patwid^2*3];
poolsize = 19;
SKIP = 4;


%%
% * 读取之前的线性解码器的训练结果
[Wb meanpat ZCAWh] = LinearDecoder();
W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
b1 = Wb(2*nodes(1)*nodes(2)+1:2*nodes(1)*nodes(2)+nodes(2));
WWh = W1*ZCAWh;

figure(1);clf;
showColorInfo(WWh');


%%
% * 数据读取
load('./data/stltrainSubset.mat');
[imgwid, ~, nch, m] = size(trainImages);

%%
% * 测试卷积和池化计算是否正确
if SKIP<1
	imgs = trainImages(:, :, :, 1:8);
	
	% 卷积计算	
	convfeats = cnnConvolve(imgs, hidsize, patwid, WWh, b1, meanpat);	
	% 检查卷积
	for k = 1:1000
		km = randi([1, 8]);
		j = randi([1, patwid^2 - patwid + 1]);
		i = randi([1, patwid^2 - patwid + 1]);		
		patch = imgs(j:j+patwid-1, i:i+patwid-1, :, km);
		feats = sigmoid( bsxfun(@plus, W1*ZCAWh*(patch(:) - meanpat), b1) );
		
		if norm(feats - convfeats(:, km, j, i)) > 1e-8
			disp('convolved feature check failed');
			break;
		end
	end
	
	
	% 池化
	poolfeats = cnnPooling(convfeats, imgwid, patwid, poolsize);	
	% 池化检测
	testmat = reshape(1:64, 8, 8);
	expect = [mean(mean(testmat(1:4, 1:4))) mean(mean(testmat(1:4, 5:8))); ...
			  mean(mean(testmat(5:8, 1:4))) mean(mean(testmat(5:8, 5:8))); ];		  
	testmat = reshape(testmat, 1, 1, 8, 8);	
	
	poolsize = 4;
	pooltest = zeros(1, 1, 8/poolsize, 8/poolsize);
	[~,~,pr,pc] = size(pooltest);	
	for r = 1:pr
		for c = 1:pc
			pool = testmat(1, 1, (r-1)*poolsize+1:r*poolsize, (c-1)*poolsize+1:c*poolsize);
			pooltest(1, 1, r, c) = mean(pool(:));
		end
	end
	
	if norm(expect(:) - pooltest(:)) > 1e-8
		disp('pooling failed');
	end
end


%%
% * 卷积和池化，提取训练数据的特征
if SKIP<2
	substep = 50;
	steps = m / substep;
	poolfeats = zeros(hidsize, m, floor((imgwid-patwid+1)/poolsize), floor((imgwid-patwid+1)/poolsize));
	
	h = waitbar(0,'Please wait...'); tic;
	for k = 1 : steps
		imgs = trainImages(:,:,:,(k-1)*substep+1:k*substep);
		convfeats = cnnConvolve(imgs, hidsize, patwid, WWh, b1, meanpat);
		poolfeats(:, (k-1)*substep+1:k*substep, :, :) = cnnPooling(convfeats, imgwid, patwid, poolsize);
		waitbar( k/steps, h, sprintf('convolve and pooling: %d/%d', k, steps));
	end
	close(h); toc
	
	save('./data/pfeats.mat', 'poolfeats', '-v7.3');
else
	load('./data/pfeats.mat');
end


%%
% * 卷积和池化，提取测试数据的特征
load('./data/stlTestSubset.mat');
if SKIP<4	
	[imgwid, ~, nch, m] = size(testImages);
	substep = 50;
	steps = m / substep;
	testpoolfeats = zeros(hidsize, m, floor((imgwid-patwid+1)/poolsize), floor((imgwid-patwid+1)/poolsize));
	
	h = waitbar(0,'Please wait...'); tic;
	for k = 1 : steps
		imgs = testImages(:,:,:,(k-1)*substep+1:k*substep);
		convfeats = cnnConvolve(imgs, hidsize, patwid, WWh, b1, meanpat);
		testpoolfeats(:, (k-1)*substep+1:k*substep, :, :) = cnnPooling(convfeats, imgwid, patwid, poolsize);
		waitbar( k/steps, h, sprintf('convolve and pooling: %d/%d', k, steps));
	end
	close(h); toc
	
	save('./data/testpfeats.mat', 'testpoolfeats', '-v7.3');
else
	load('./data/testpfeats.mat');
end


%%
% * SoftMax回归并测试
trainX = permute(poolfeats, [1 3 4 2]);
trainX = reshape(trainX, numel(poolfeats) / size(poolfeats,2), size(poolfeats,2));
testX = permute(testpoolfeats, [1 3 4 2]);
testX = reshape(testX, numel(testpoolfeats) / size(testpoolfeats,2), size(testpoolfeats,2));
SoftMax(trainX, trainLabels, testX, testLabels, 200);

end


%=============================================================================================================%


%%
% * 激活函数
%
function y = sigmoid(x)
	y = 1 ./ (1+exp(-x));
end


%%
% * 显示网络
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
% * 卷积计算
function convfeats = cnnConvolve(imgs, hidsize, patwid, WWh, b1, meanpat)
[imgwid, ~, nch, m] = size(imgs);
convfeats = zeros(hidsize, m, imgwid-patwid+1, imgwid-patwid+1);
for km = 1:m
	for j = 1:imgwid-patwid+1
		for i = 1:imgwid-patwid+1
			subim = imgs(j:j+patwid-1, i:i+patwid-1, :, km);
			convfeats(:, km, j, i) = sigmoid(WWh * (subim(:) - meanpat) + b1);
		end
	end
end
end


%%
% * 池化计算
function poolfeats = cnnPooling(convfeats, imgwid, patwid, poolsize)
[hidsize, m, ~, ~] = size(convfeats);
poolfeats = zeros(hidsize, m, floor((imgwid-patwid+1)/poolsize), floor((imgwid-patwid+1)/poolsize));
[~,~,pr,pc] = size(poolfeats);
for f = 1:hidsize
	for km = 1:m
		for r = 1:pr
			for c = 1:pc
				pool = convfeats(f, km, (r-1)*poolsize+1:r*poolsize, (c-1)*poolsize+1:c*poolsize);
				poolfeats(f, km, r, c) = mean(pool(:));
			end
		end
	end
end
end