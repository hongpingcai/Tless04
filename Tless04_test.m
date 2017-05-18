%% added by Hongping Cai, 04/05/2017
% Tless04_test.m
%
%INPUT:
% 
%OUTPUT:
% Tless04/test_shuffle/kmeans_15_ite*.mat
% Tless04/test_shuffle/centres_ite*.txt
% Tless04/test_shuffle/test_feats_ite*.mat

%%

addpath('../Tless02');
Tless02_init;
lamda = 0.2;%%%%%%%%%%%%%%%%%%%%%%

dir_test_shuffle = fullfile(dir_Tless04,['test_shuffle_lamda' int2str(lamda*10)]);
if ~exist(dir_test_shuffle,'dir')
mkdir(dir_test_shuffle)
end;

cluster_method = 'kmeans';
cluster_K = 15;
feat_type = 'caffenet';
feat_blob = 'fc6';
feat_dim  = 4096;
caffe_input_w = 227; 
caffe_input_h = 227;
fix_layer = 5; %%%%%%%%%%%%%%%%%%%%%%
be_show = 0;

mat_ids_shuffle = fullfile(dir_Tless03,'test_shuffle','ids_shuffle.mat');
if ~exist(mat_ids_shuffle,'file')
    error(sprintf('%s does not exist. Pls run Tless03_shuffle_test.m first.',mat_ids_shuffle));
end;
load(mat_ids_shuffle,'ids_shuffle','labels_shuffle','im_files_shuffle');

for ite=1:1
    fprintf(1,'****** ite = %d *******\n',ite);
    
    %% SPTE1: generate the test features 
    mat_test_feat = fullfile(dir_test_shuffle, ['fix' int2str(fix_layer) '_test_feat_ite' int2str(ite) '.mat']);
    fprintf(1,'SPTE1: generate the test features...\n');
    if ~exist(mat_test_feat,'file')
        net_prototxt = fullfile(dir_Tless04,'caffenet-prototxt','deploy.prototxt');
        net_caffemodel = fullfile(dir_Tless04,'caffenet-model',...
            ['fix' int2str(fix_layer) '-tless-caffenet_ite' int2str(ite-1) '_iter_5000.caffemodel']);
        mat_mean = fullfile(dir_Tless02,'ilsvrc_2012_mean_227.mat');
        disp('** Generate the testing features....');
        caffe.set_mode_gpu();
        gpu_id = 0;  % we will use the first gpu in this demo
        caffe.set_device(gpu_id);
        net = caffe.Net(net_prototxt, net_caffemodel,'test');
        
        mat_test_patch = fullfile(dir_Tless01,'mode0','test_patch.mat');
        load(mat_test_patch,'patches');        
        %mat_test_label = fullfile(dir_Tless01,'mode0','test_label.mat');
        patches = patches(ids_shuffle,:,:,:);
        
        
        n_im_test = size(patches,1);
        test_feats = zeros(n_im_test,4096);
        for i=1:n_im_test
            if mod(i,50)==1
                fprintf(1,'%d ',i);
            end;
            im = squeeze(patches(i,:,:,:));
            input_data = {prepare_image(im,caffe_input_w,caffe_input_h,mat_mean)};
            scores = net.forward(input_data);
            cur_feat = net.blobs(feat_blob).get_data();
            test_feats(i,:) = cur_feat';%%%%%%%%
        end;
        fprintf(1,'\n  Save features into %s\n',mat_test_feat);
        caffe.reset_all();
        save(mat_test_feat,'test_feats');
    else
        fprintf(1,'** Load the test features frm %s....\n',mat_test_feat);
        load(mat_test_feat,'test_feats');
    end;
    
    %% STEP2: generate the centre image features
    fprintf(1,'SPTE2: generate the centre image features.\n');
    if ite==1
        mat_old_cluster = fullfile(dir_Tless03,'test_shuffle',[cluster_method '_' int2str(cluster_K) ...
        '_ite' int2str(ite-1) '.mat']);
    else
        mat_old_cluster = fullfile(dir_test_shuffle,['fix' int2str(fix_layer) '_' cluster_method '_' int2str(cluster_K) ...
        '_ite' int2str(ite-1) '.mat']);
    end;
    load(mat_old_cluster,'ids_centre');
    old_centre_feats = test_feats(ids_centre,:);
    old_centre_ids = ids_centre;
    clear ids_centre;
    
    %% STEP3: clustering
    fprintf(1, 'STEP3: clustering...\n');
    mat_cluster = fullfile(dir_test_shuffle,['fix' int2str(fix_layer) '_' cluster_method '_' int2str(cluster_K) ...
        '_ite' int2str(ite) '.mat']);
    if ~exist(mat_cluster,'file')
        rng(1); % For reproducibility
        [ids_cluster_,centres_cluster_,sumd_,D_] = kmeans(test_feats,cluster_K,'MaxIter',1000,...
            'start',old_centre_feats,'Display','final');
        [ids_cluster,centres_cluster,sumd,D] = kmeans(test_feats,cluster_K,'MaxIter',1000,...
            'start','cluster','Display','final','Replicates',10);
        if (sum(sumd_)-sum(sumd))/sum(sumd)<0.0001
            fprintf(1,'The clustering from the initial of previous centres is adopted.\n');
            ids_cluster = ids_cluster_;
            centres_cluster = centres_cluster_;
            sumd = sumd_;
            D = D_ ;
        else
            fprintf(1,'The clustering from start-cluster is adopted.\n');
        end;
        
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
    fprintf(1,'STEP4: clustering performance(lamda=%.1f,ite=%d,fix_layer=%d)...\n',lamda,ite,fix_layer);
    theta_group_purity = 0.8;
    [ACC] = eval_cluster1(ids_cluster, labels_shuffle);%
    fprintf(1,'** ACC: %.4f\n',ACC);
    [rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, labels_shuffle, theta_group_purity);
    fprintf(1,'** Obj-wise: rec: %.4f, pre: %.4f, tp:%d\n',rec,pre,tp);
    fprintf(1,'** Frm-wise: acc_fm: %.4f, tp_fm: %d\n', acc_fm,tp_fm);
    
    new_centre_feats = test_feats(ids_centre,:);
    new_centre_ids = ids_centre;
    dis_centres = pdist2(old_centre_feats,new_centre_feats);
    [assignment,cost] = munkres(dis_centres);    
    if be_show
        im_all = zeros(64*2,64*15,3,'uint8');
        for k=1:cluster_K
            im_all(1:64,(k-1)*64+1:k*64,:)   = imresize(imread(im_files_shuffle{old_centre_ids(k)}),[64 64]);
            im_all(65:end,(k-1)*64+1:k*64,:) = imresize(imread(im_files_shuffle{new_centre_ids(assignment(k))}),[64 64]);
        end;
        figure(1);clf;
        imshow(im_all);
        title(sprintf('Up: centres of ite=%d, Down: centres of ite=%d',ite-1,ite));
        fig_name = fullfile(dir_Tless04, 'fig',sprintf('lamda%d-fix%d-centres_ite%d-ite%d.png',lamda*10,fix_layer,ite-1,ite));
        if ~exist(fig_name)
            export_fig(fig_name);
        end;
    end
        
    %% STEP5: write the new centres into txt file
    txt_centres = fullfile(dir_test_shuffle,['fix' int2str(fix_layer) '_centres_ite' int2str(ite) '.txt']);
    fprintf(1,'STEP5: Write the centres into %s ....\n',txt_centres);
    if ~exist(txt_centres,'file')
        fid = fopen(txt_centres,'w');
        for i=1:cluster_K
            fprintf(fid,'%s %d\n',im_files_shuffle{ids_centre(i)}, i-1); %% 0,1,....
            fprintf(1,'%s %d\n',im_files_shuffle{ids_centre(i)}, i-1);
            if be_show
                figure(2);clf;
                imshow(im_files_shuffle{ids_centre(i)});
                tmp_ = findstr(im_files_shuffle{ids_centre(i)},'/');
                title(['C' int2str(i) '_centre image (' ...
                    im_files_shuffle{ids_centre(i)}(tmp_(end-1)+1:end) ')']);
                pause;
            end;
        end;
        fclose(fid);
    end;
    if cost<1e4%%%%%%%%
        fprintf(1,'STOP.');
        break;
    end;
end;

