function LinearRegression()

clear;
clc;

% �����ַ������лع�:
% 1. normal equations���򷽳�
% 2. �ݶ��½�
% ������ά�ȵ͵�ʱ��,ʹ�����򷽳̱Ƚϼ�,����.ͬʱ��ΪҪ����,�������������ķ���.
% ����ά�ȸߵ�ʱ��,ʹ���ݶ��½�����Ч��.���ⲻ��������ķ���.

%һά����ϵͳ
if 0
	x = load('./data/ex2x.dat');
	y = load('./data/ex2y.dat');
	x(:,2) = 1;
	
	%% 1. ���򷽳�
	W1 = x \ y;	
	figure(1);
	clf;
	plot(x(:,1),y,'o');
	hold on;
	plot(x(:,1), x*W1);
	
	
	%% 2. �ݶ��½�
	% Ŀ�꺯����С,��֪��ΪʲôҪ��m
	% L = (XW-Y)^2 / (2*m)
	
	W2 = zeros(2,1);
	MAX_ITR = 1000;
	m = length(x);
	% �����,���ù̶�����
	lambda = 0.07;
	
	for it = 1:MAX_ITR
		grad = x'*(x*W2-y) / m;
		W2 = W2 - lambda * grad;
	end
	
	figure(2)
	clf;
	plot(x(:,1),y,'o');
	hold on;
	plot(x(:,1), x*W2);
end

% ��ά����
if 1
	x = load('./data/ex3x.dat');
	y = load('./data/ex3y.dat');
	x(:,end+1) = 1;	
	
	%% 1. ���򷽳�
	W1 = x \ y;
	
	%% 2. �ݶ�
	% Loss���� L = (XW-Y)^2 / (2*m)
	% ��ά���ݿ�����Ҫ��һ��
	meanx = mean(x);
	sigmax = std(x);
	x(:,1) = (x(:,1)-meanx(1)) / sigmax(1);
	x(:,2) = (x(:,2)-meanx(2)) / sigmax(2);
	
	
	% ��ͬ��ѧϰ��(����),Ӱ�������ٶ�.
	MAXITER = 100;
	m = length(y);
	lambda = [0.01 0.03 0.1 0.3 1.3 1];
	style = {'b' 'r' 'g' 'k' 'b--' 'r--'};
	
	clf; hold on;
	for l = 1:length(lambda)
		W = zeros(size(x,2), 1);
		resi = zeros(MAXITER, 1);
		for it = 1:MAXITER
			resi(it) = sum((x*W-y).^2) / (2*m);
			grad = x' * (x*W - y) / m;
			W = W - lambda(l)*grad;
		end		
% 		plot(0:MAXITER-1, resi, style{l}, 'linewidth', 2);
		plot(0:49, resi(1:50), style{l}, 'linewidth', 2);
	end
	legend('0.01','0.03','0.1','0.3','1.3','1');	
end

end