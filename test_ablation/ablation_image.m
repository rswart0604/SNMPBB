[cdata, ~, alphadata] = imread("/Users/ryanswart/Projects/SNMPBB/ablation_test/painting.png");
% [cdata, ~, alphadata] = imread("/Users/ryanswart/Projects/SNMPBB/ablation_test/painting2.jpg");
% [cdata, ~, alphadata] = imread("/Users/ryanswart/Projects/SNMPBB/ablation_test/largeimage.png");
painting = double(cdata);

% largeimage
% painting = painting(1:2000,1:2000,:);
% cdata = cdata(1:2000,1:2000,:);


% painting2
% painting = painting(1:1154,1:1154,:);
% cdata = cdata(1:1154,1:1154,:);

% painting
painting = painting(1:816,1:816,:);
cdata = cdata(1:816,1:816,:);
r = 50;

painting_one = painting(:,:,1);
painting_one = tril(painting_one) + triu(painting_one', 1);
% sym_weight = 2 * norm(painting_one, 'fro') / sqrt(2000*25);
[W1, H1, output1, ~] = Graph_SNMPBB_modified(painting_one, r,'verbose', 1, 'graph_reg', -1, 'do_preprocess',false);
[H1_, output_anls1, acc_anls] = symnmf_anls(painting_one, r);

painting_2 = painting(:,:,2);
painting_2 = tril(painting_2) + triu(painting_2', 1);
% sym_weight = 2 * norm(painting_2, 'fro') / sqrt(2000*25);
[W2, H2, output2, ~] = Graph_SNMPBB_modified(painting_2, r,'verbose', 1, 'graph_reg', -1, 'do_preprocess',false);
[H2_, output_anls2, acc_anls] = symnmf_anls(painting_2, r);


painting_3 = painting(:,:,3);
painting_3 = tril(painting_3) + triu(painting_3', 1);
sym_weight = 2 * norm(painting_3, 'fro') / sqrt(2000*25);
[W3, H3, output3, ~] = Graph_SNMPBB_modified(painting_3, r,'verbose', 1, 'graph_reg', -1, 'do_preprocess',false);
[H3_, output_anls3, acc_anls] = symnmf_anls(painting_3, r);

figure(1);
m = uint8(cat(3,W1*H1,W2*H2,W3*H3));
imshow(m)

figure(2);
m2 = uint8(cat(3,H1_*H1_',H2_*H2_',H3_*H3_'));
imshow(m2)


figure(3); plot(output1.total_time, output1.relres); hold on; plot(output_anls1.total_time, output_anls1.relres); hold off
figure(4); plot(output2.total_time, output2.relres); hold on; plot(output_anls2.total_time, output_anls2.relres); hold off
figure(5); plot(output3.total_time, output3.relres); hold on; plot(output_anls3.total_time, output_anls3.relres); hold off

original_image = uint8(cat(3,painting_one,painting_2,painting_3));
figure(6); imshow(original_image)