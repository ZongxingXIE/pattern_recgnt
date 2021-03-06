clc;

data = xlsread('Cryotherapy.xlsx');
% sex,	age,	Time,	Number_of_Warts,	Type,	Area,	Result_of_Treatment
% the 7th column is the class
X = data(:,1:6); %attributes
Y = data(:,7);   %class result

num_feature = size(X,2);
% ==============================
% % 
% % normalization
% % 
data_normlz = normaliz(data,num_feature);
% 
data = data_normlz;
% ==============================

% ==============================
% % pca
% % 
data_pca = PCA(data, num_feature, 0.05);
% 
data = data_pca;
% ===============================


% % ==============================
% % divide into trainset and testset
% 
test_ratio = 0.15;
siz = size(data);
train_set = data(1:round((1-test_ratio)*siz),:);


test_set = data(round((1-test_ratio)*siz)+1:siz, :);

% ==============================

% % accuracy
[row,col] = size(test_set);
correct = 0;
total = 0;
for i=1:row
    test = test_set(i,:);
    result = knn_algo(train_set,test,5);
    if(result) 
        correct=correct+1;
    end
    total=total+1;            
end
acc = correct/total




                    
