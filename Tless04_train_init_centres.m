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
feat_blob = 'fc7';
feat_dim  = 128;
train_dataset = 'imagenet';%'tless';%
caffe_input_w = 227; 
caffe_input_h = 227;
be_show = 1;

dir_Tless04_train = fullfile(dir_Tless04,'train');

%% load the training image and label
txt_train_list = fullfile(dir_Tless01,'train_val/train.txt');
[train_im_files, train_label] = textread(txt_train_list,'%s %d');
n_im_train = length(train_label);

%% generate the training features
switch lower(train_dataset)
    case 'tless'
        net_prototxt = fullfile(dir_Tless01,'caffenet-prototxt','deploy.prototxt');
        net_caffemodel = fullfile(dir_Tless01,'caffenet-model',...
            ['m' int2str(opt_split_trainval) '_Tless-caffenet_iter_10000.caffemodel']);
    case 'imagenet'
        net_prototxt = fullfile(dir_DATA,'Hongping/model-caffenet/deploy.prototxt');
        net_caffemodel = fullfile(dir_DATA,'Hongping/model-caffenet/bvlc_reference_caffenet.caffemodel');
    otherwise
        error('No such train_dataset.');
end;
mat_train_feat = fullfile(dir_Tless04_train, ['train_feat.mat']);
if ~exist(mat_train_feat,'file')
    disp('** Generate the training features....');
    mat_mean = '/media/deepthought/DATA/Hongping/Tless02/ilsvrc_2012_mean_227.mat';
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
    ids_cur_cluster = find(train_label==i-1);
    cur_feats = train_feats(ids_cur_cluster,:);
    mean_feats = sum(cur_feats)./size(cur_feats,1);
    dis_cur   = pdist2(cur_feats,mean_feats);
    [v,d] = min(dis_cur);
    ids_centre(i) = ids_cur_cluster(d);
end;

%% write the centres into txt file
txt_centres = fullfile(dir_Tless04_train,['centres_init.txt']);
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


