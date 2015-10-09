function NativeBayes()

%{
���ر�Ҷ˹����Ѱ����������ʣ������������ʷֲ��������������ļ���,������Ȼ������׼ȷ�ԣ����ǲ���ʹ���ر�Ҷ˹��
������ʣ�
P(Y=c_k)
�������ʣ�
P(X=x|Y=c_k) = \prod P(X^(j)=x^(j) | Y=c_k)
������ʣ�
P(Y=c_k|X=x) = \frac {P(Y=c_k)P(X=x|Y=c_k)} {\sum P(Y=c_k) \prod P(X^(j)=x^(j)|Y=c_k}

��ĸ��c_k�޹أ�������������ʣ�
y = argmin P(Y=c_k) P(X=x|Y=c_k)
%}


%% 1. ѵ��
ndocs = 700;
ntokens = 2500;

% ��һ�� �ĵ����
% �ڶ��� �������
% ������ �ĵ��ڵ��ʳ��ִ���
M = load('./ex6DataPrepared/train-features.txt');

% A = sparse(i,j,s,m,n)
% ������i,j,s�ϲ���һ��mxn���󣬲�A(i(k),j(k)) = s(k)
% �������A���д����ĵ���ţ��д�������ţ�������ֵ����õ����ڸ��ĵ����ֵĴ���
spmat = sparse(M(:,1), M(:,2), M(:,3), ndocs, ntokens);
flmat = full(spmat);

% ������ݣ��Ƿ��������ʼ�
labs = load('./ex6DataPrepared/train-labels.txt');
yes_ind = find(labs);
no_ind = find(labs==0);

% ÿ���ĵ��ĵ�������
words_doc = sum(flmat, 2);

% �����ʼ��ĵ�������
words_yes = sum(words_doc(yes_ind));
words_no = sum(words_doc(no_ind));


%% 1.1 �������
prob_yes = length(yes_ind) / ndocs;
prob_no = 1 - prob_yes;

%% 1.2 ��������
% Ϊ��ֹ�еĸ���Ϊ0��Ӱ����������㣬������Ҫ����������˹ƽ����\lambdaͨ��=1
prob_words_yes = (sum(flmat(yes_ind,:))+1) ./ (ntokens + words_yes);
prob_words_no = (sum(flmat(no_ind,:))+1) ./ (ntokens + words_no);





%% 2. ����
N = load('./ex6DataPrepared/test-features.txt');
spmat = sparse(N(:,1), N(:,2), N(:,3));
flmat = full(spmat);

[ndocs ntokens] = size(flmat);

a = flmat*(log(prob_words_yes))' + log(prob_yes);
b = flmat*(log(prob_words_no))' + log(prob_no);
out = a > b;

labs = load('./ex6DataPrepared/test-labels.txt');

end