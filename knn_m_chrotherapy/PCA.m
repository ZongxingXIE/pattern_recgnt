function data_pca = PCA(data, num_feature, threshold)

corr = corrcoef(data)
corr_abs = abs(corr(num_feature+1,1:num_feature));
[~,inx] = sort(corr_abs,'descend');
inx_pc = [];
for i=1:num_feature
 if corr_abs(inx(i)) > threshold %% the threshold to reduce the irrelative component
     inx_pc = [inx_pc inx(i)];
 end
end
% inx_pc
data_pca = data(:,[inx_pc num_feature+1]);
size(data_pca);