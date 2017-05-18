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

dir_Tless04_train = fullfile(dir_Tless04,['train_lamda' int2str(lamda)]);
if ~exist(dir_Tless04_train,'dir')
mkdir(dir_Tless04_train);
end;

%% load the training image and label
txt_train_list = fullfile(dir_Tless01,'train_val/train.txt');
[train_im_files, train_labels] = textread(txt_train_list,'%s %d');
n_im_train = length(train_labels);

ite = 1;
%% generate the training features
mat_train_feat = fullfile(dir_Tless04_train, ['train_feat_ite' int2str(ite) '.mat']);
if ~exist(mat_train_feat,'file')
    disp('** Generate the training features....');
    net_prototxt = fullfile(dir_Tless04,'caffenet-prototxt','deploy.prototxt');
    net_caffemodel = fullfile(dir_Tless04,'caffenet-model',...
        ['fix' int2str(fix_layer) '-tless-caffenet_ite' int2str(ite-1) '_iter_5000.caffemodel']);
    mat_mean = fullfile(dir_Tless02,'ilsvrc_2012_mean_227.mat');
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu in this demo
    caffe.set_device(gpu_id);
    net = caffe.Net(net_prototxt, net_caffemodel,'test');
    train_feats = zeros(n_im_train,4096);
    for i=1:n_im_train
        if mod(i,50)==1
            fprintf(1,'%d ',i);
        end;
        im = imread(train_im_files{i});
        input_data = {prepare_image(im,caffe_input_w,caffe_input_h,mat_mean)};
        scores = net.forward(input_data);
        cur_feat = net.blobs(feat_blob).get_data();
        train_feats(i,:) = cur_feat';%%%%%%%%
    end;
    fprintf(1,'\n  Save features into %s\n',mat_train_feat);
    caffe.reset_all();
    save(mat_train_feat,'train_feats');
else
    fprintf(1,'** Load the training features frm %s....\n',mat_train_feat);
    load(mat_train_feat,'train_feats');
end;

ids_centre = zeros(1,cluster_K);
for i=1:cluster_K
    ids_cur_cluster = find(train_labels==i-1);
    cur_feats = train_feats(ids_cur_cluster,:);
    mean_feats = sum(cur_feats)./size(cur_feats,1);
    dis_cur   = pdist2(cur_feats,mean_feats);
    [v,d] = min(dis_cur);
    ids_centre(i) = ids_cur_cluster(d);
end;

%% write the centres into txt file
txt_centres = fullfile(dir_Tless04_train,['centres_ite' int2str(ite) '.txt']);
fprintf(1,'** Write the centresinto %s ....\n',txt_centres);
fid = fopen(txt_centres,'w');
for i=1:cluster_K    
    fprintf(fid,'%s %d\n',train_im_files{ids_centre(i)}, i-1); %% 0,1,....
    fprintf(1,'%s %d\n',train_im_files{ids_centre(i)}, i-1); %% 0,1,....
    if be_show
        figure(1);clf;
        imshow(train_im_files{ids_centre(i)});
        title(['C' int2str(i) '_centre image']);
        pause;
    end;
end;
fclose(fid);


%% STEP3: clustering
fprintf(1, 'STEP3: clustering...\n');
mat_cluster = fullfile(dir_Tless04_train,['fix' int2str(fix_layer) '_' cluster_method '_' int2str(cluster_K) ...
    '_ite' int2str(ite) '.mat']);
if ~exist(mat_cluster,'file')
    rng(1); % For reproducibility
%     [ids_cluster_,centres_cluster_,sumd_,D_] = kmeans(train_feats,cluster_K,'MaxIter',1000,...
%         'start',old_centre_feats,'Display','final');
    [ids_cluster,centres_cluster,sumd,D] = kmeans(train_feats,cluster_K,'MaxIter',1000,...
        'start','cluster','Display','final','Replicates',10);
%     if (sum(sumd_)-sum(sumd))/sum(sumd)<0.0001
%         fprintf(1,'The clustering from the initial of previous centres is adopted.\n');
%         ids_cluster = ids_cluster_;
%         centres_cluster = centres_cluster_;
%         sumd = sumd_;
%         D = D_ ;
%     else
%         fprintf(1,'The clustering from start-cluster is adopted.\n');
%     end;
%     
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
disp('Cluster on Tless-train set.');
fprintf(1,'STEP4: clustering performance(lamda=%.1f,ite=%d,fix_layer=%d)...\n',lamda,ite,fix_layer);
theta_group_purity = 0.8;
[ACC] = eval_cluster1(ids_cluster, train_labels);%
fprintf(1,'** ACC: %.4f\n',ACC);
[rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, train_labels, theta_group_purity);
fprintf(1,'** Obj-wise: rec: %.4f, pre: %.4f, tp:%d\n',rec,pre,tp);
fprintf(1,'** Frm-wise: acc_fm: %.4f, tp_fm: %d\n', acc_fm,tp_fm);



