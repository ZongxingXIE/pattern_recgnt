function data_nmlz = normaliz(data, num_feature)
min_ = [];
max_ = [];
for j=1:num_feature
% %     scale to [0,1]
%     min_(j) = min(data(:,j));
%     max_(j) = max(data(:,j));
%     data_scaled = (data(:,j)-min_(j))/(max_(j)-min_(j));
    
% %     normalizing
    mean_(j) = mean(data(:,j));
    var_(j) = var(data(:,j));
    data_scaled = (data(:,j)-mean_(j))/(var_(j));

    data_nmlz(:,j) = data_scaled/norm(data_scaled);
end

% return data_nmlz;
data_nmlz(:,j+1)=data(:,j+1);



