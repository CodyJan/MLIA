function NativeBayes()

%{
朴素贝叶斯就是寻找最大后验概率，其中条件概率分布做了条件独立的假设,这样虽然牺牲了准确性，但是才能使朴素贝叶斯简化
先验概率：
P(Y=c_k)
条件概率：
P(X=x|Y=c_k) = \prod P(X^(j)=x^(j) | Y=c_k)
后验概率：
P(Y=c_k|X=x) = \frac {P(Y=c_k)P(X=x|Y=c_k)} {\sum P(Y=c_k) \prod P(X^(j)=x^(j)|Y=c_k}

分母与c_k无关，所以最大后验概率：
y = argmin P(Y=c_k) P(X=x|Y=c_k)
%}


%% 1. 训练
ndocs = 700;
ntokens = 2500;

% 第一列 文档序号
% 第二列 单词序号
% 第三列 文档内单词出现次数
M = load('./ex6DataPrepared/train-features.txt');

% A = sparse(i,j,s,m,n)
% 将向量i,j,s合并成一个mxn矩阵，并A(i(k),j(k)) = s(k)
% 所以最后A的行代表文档序号，列代表单词序号，具体数值代表该单词在该文档出现的次数
spmat = sparse(M(:,1), M(:,2), M(:,3), ndocs, ntokens);
flmat = full(spmat);

% 类别数据，是否是垃圾邮件
labs = load('./ex6DataPrepared/train-labels.txt');
yes_ind = find(labs);
no_ind = find(labs==0);

% 每个文档的单词数量
words_doc = sum(flmat, 2);

% 垃圾邮件的单词数量
words_yes = sum(words_doc(yes_ind));
words_no = sum(words_doc(no_ind));


%% 1.1 先验概率
prob_yes = length(yes_ind) / ndocs;
prob_no = 1 - prob_yes;

%% 1.2 条件概率
% 为防止有的概率为0，影响最大后验计算，所以需要加上拉普拉斯平滑，\lambda通常=1
prob_words_yes = (sum(flmat(yes_ind,:))+1) ./ (ntokens + words_yes);
prob_words_no = (sum(flmat(no_ind,:))+1) ./ (ntokens + words_no);





%% 2. 测试
N = load('./ex6DataPrepared/test-features.txt');
spmat = sparse(N(:,1), N(:,2), N(:,3));
flmat = full(spmat);

[ndocs ntokens] = size(flmat);

a = flmat*(log(prob_words_yes))' + log(prob_yes);
b = flmat*(log(prob_words_no))' + log(prob_no);
out = a > b;

labs = load('./ex6DataPrepared/test-labels.txt');

end