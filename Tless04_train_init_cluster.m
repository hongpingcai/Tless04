%% Added by Hongping Cai
% Tless04_train_init_centres.m
% Generate the centres with the cluster labels in the Tless-train set
%

%%
addpath('../Tless02');
Tless02_init;
cluster_method = 'kmeans';
cluster_K = 15;
feat_type = 'caffenet';
feat_blob = 'fc6';
feat_dim  = 128;
train_dataset = 'imagenet';%'tless';%
caffe_input_w = 227; 
caffe_input_h = 227;
be_show = 0;
fix_layer = 5;
lamda = 0.2;

dir_Tless04_train = fullfile(dir_Tless04,['train']);

%% load the training image and label
txt_train_list = fullfile(dir_Tless01,'train_val/train.txt');
[train_im_files, train_labels] = textread(txt_train_list,'%s %d');
n_im_train = length(train_labels);

ite = 1;
%% generate the training features
mat_train_feat = fullfile(dir_Tless04_train, ['train_feat.mat']);
if ~exist(mat_train_feat,'file')
error('mat_train_feat does not exist. Run Tless04_train_init.m first');
end;
    fprintf(1,'** Load the training features frm %s....\n',mat_train_feat);
    load(mat_train_feat,'train_feats');


%% STEP3: clustering
fprintf(1, 'STEP3: clustering...\n');
mat_cluster = fullfile(dir_Tless04_train,[cluster_method '_' int2str(cluster_K) ...
    '_init.mat']);
if ~exist(mat_cluster,'file')
    rng(1); % For reproducibility
    [ids_cluster,centres_cluster,sumd,D] = kmeans(train_feats,cluster_K,'MaxIter',1000,...
        'start','cluster','Display','final','Replicates',10);
  
    % find the cluster centre images
    ids_centre = zeros(1, cluster_K);
    for i=1:cluster_K
        ids_cur = find(ids_cluster == i);
        dis_cur = D(ids_cur,i);
        [v,d] = min(dis_cur);
        ids_centre(i) = ids_cur(d);
    end;
    save(mat_cluster,'ids_cluster','D','centres_cluster','ids_centre');
else
    fprintf(1,'** Load clustering file: %s ....\n',mat_cluster);
    load(mat_cluster);
end;

%% STEP4: clustering performance
disp('Cluster on Tless-train set, init performance.');
fprintf(1,'STEP4: clustering performance\n');
theta_group_purity = 0.8;
[ACC] = eval_cluster1(ids_cluster, train_labels);%
fprintf(1,'** ACC: %.4f\n',ACC);
[rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, train_labels, theta_group_purity);
fprintf(1,'** Obj-wise: rec: %.4f, pre: %.4f, tp:%d\n',rec,pre,tp);
fprintf(1,'** Frm-wise: acc_fm: %.4f, tp_fm: %d\n', acc_fm,tp_fm);



