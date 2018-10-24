clear all;
cd /home/can/projeler/tft
setup_tft
cd /home/can/projeler/tft_image_completion

corrupted_image = imread('/home/can/projeler/tft_image_completion/corrupted_image.png');
I_corr = rgb2gray(corrupted_image);
I2_corr = sparse(double(I_corr));
imwrite(I_corr,'corrupted_image_grayscale.png')
%I2_corr( I2_corr==0 ) = 0.0001;

i_index = Index(size(I_corr,1));
j_index = Index(size(I_corr,2));
k_index = Index(10);

iter_num = 100

X = Tensor(i_index, j_index);
X.data = I2_corr;

A=Tensor(i_index, k_index);
B=Tensor(j_index, k_index);

A.data = rand(i_index.cardinality, k_index.cardinality);
B.data = rand(j_index.cardinality, k_index.cardinality);

pre_process();

p = [1];
phi = [1];

factorization_model = {X, {A, B}};

model = TFModel(factorization_model, p, phi);
config = TFEngineConfig(model, iter_num);
engine = TFDefaultEngine(config, 'gtp_mex');
engine.factorize();
plot(engine.beta_divergence');
check_divergence(engine.beta_divergence);

pred = squeeze(A.data) * squeeze(B.data)';

original_image = imread('/home/can/projeler/tft_image_completion/original.jpg');
I_orig = rgb2gray(original_image);
imwrite(I_orig,'original_grayscale.png')

sqrt( sum(sum( (pred - double(I_orig)).^2 ) )/prod(size(I_orig)))

imwrite(pred/100,'AtimesB_normalized.png')

I3 = I_corr;
I3(I3==0) = pred(I3==0);
imwrite(I3, ['corrupted_with_pred_k' num2str(k_index.cardinality) '_iter' num2str(iter_num) '.png'] )

sqrt( sum(sum( (double(I3) - double(I_orig)).^2 ) )/prod(size(I_orig)))