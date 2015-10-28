
%% ���
% ʹ�þ���˵������硣������ͼ��ߴ�ϴ��������ݽ϶࣬���¼���ǳ����ѡ���������ʶ��ͼ��ľ��飬��ʹ�þ����+�ػ���һ�������ԭͼ��ת��һ����������������������������ѵ����
%
% * �������ˣ���ʵʩ�ϣ�����Ҫȷ������ˣ�����˵ļ�������൱����һ��ѹ���Ĺ��̣��ȴ�ԭʼͼƬ�ϻ�ȡһ����Ŀ��8*8*3��ͼ��飬ʹ��ϡ���Ա������õ�Ȩ�ء�ƫ�Ʋ���
% $W,b$ ��������Ҫ��ZCA�׻��������������Ҫ�������Խ���������ѵ�������Ե���֮ǰ��LinearDecoder������
% 
% * �����������LinearDecoderѵ���õ� $W^{(1)}, b^{(1)}$��������ԭʼͼƬ���Ľ��о�������ھ��֮ǰ������Ҫ���ǹ�0�Ͱ׻���Ԥ��������ʹ��ǰһ��������ľ�ֵ��ZCAWhitening������
%
% * �ػ����������������������Ȼ�н��ͣ�������Ȼ�ϴ󣬻���Ҫ��һ���Ľ��гػ�������
% ��ν�ػ�������һ������Ļ����ص��ķ�Χ�ڣ����ʱ�����ص��ģ���ȡһ��ƽ����/��������ֵ��Ϊ��pool�Ĵ��������ĳػ��ǻ����һ����Χ��ƽ�Ʋ����ԡ�
%
% * �ع顢���ԣ��ػ���������Ѿ���󽵵���ѵ������������������Щ��������SoftMax�������ع顣������������ʾ����ȷ��ֻ��80.28%��
%
%% �������

function CNN()

hidsize = 400;
patwid = 8;
nodes = [patwid^2*3, hidsize, patwid^2*3];
poolsize = 19;
SKIP = 4;


%%
% * ��ȡ֮ǰ�����Խ�������ѵ�����
[Wb meanpat ZCAWh] = LinearDecoder();
W1 = reshape(Wb(1:nodes(1)*nodes(2)), nodes(2), []);
b1 = Wb(2*nodes(1)*nodes(2)+1:2*nodes(1)*nodes(2)+nodes(2));
WWh = W1*ZCAWh;

figure(1);clf;
showColorInfo(WWh');


%%
% * ���ݶ�ȡ
load('./data/stltrainSubset.mat');
[imgwid, ~, nch, m] = size(trainImages);

%%
% * ���Ծ���ͳػ������Ƿ���ȷ
if SKIP<1
	imgs = trainImages(:, :, :, 1:8);
	
	% �������	
	convfeats = cnnConvolve(imgs, hidsize, patwid, WWh, b1, meanpat);	
	% �����
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
	
	
	% �ػ�
	poolfeats = cnnPooling(convfeats, imgwid, patwid, poolsize);	
	% �ػ����
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
% * ����ͳػ�����ȡѵ�����ݵ�����
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
% * ����ͳػ�����ȡ�������ݵ�����
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
% * SoftMax�ع鲢����
trainX = permute(poolfeats, [1 3 4 2]);
trainX = reshape(trainX, numel(poolfeats) / size(poolfeats,2), size(poolfeats,2));
testX = permute(testpoolfeats, [1 3 4 2]);
testX = reshape(testX, numel(testpoolfeats) / size(testpoolfeats,2), size(testpoolfeats,2));
SoftMax(trainX, trainLabels, testX, testLabels, 200);

end


%=============================================================================================================%


%%
% * �����
%
function y = sigmoid(x)
	y = 1 ./ (1+exp(-x));
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
% * �������
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
% * �ػ�����
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