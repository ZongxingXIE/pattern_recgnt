function vote_result = knn_algo(Train, predict, k)
% Train is the train dataset
% predict is the test vector
% k is the number of nearest neighbors used to vote
[row,col] = size(Train);

eucl_dist_class = [];

test = predict(1,1:col-1);

diff = Train(:,1:col-1) - test;
for i=1:row
%     
     eucl_dist_class = [eucl_dist_class; sqrt(sum(diff(i,:).^2)),Train(i,col)];
%     
end

[~,inx] = sort(eucl_dist_class(:,1));
sorted = eucl_dist_class(inx,:);
% k = 5;
% count the voting
vote_ratio = sum(sorted(1:k,2))/k;
if vote_ratio>0.5
    vote = 1;
else
    vote = 0;
end

vote_result = predict(1,col)== (vote);