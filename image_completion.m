clear all;
cd /home/can/projeler/tft
setup_tft
cd /home/can/projeler/image_completion

corrupted_image = imread('/home/can/projeler/image_completion/corrupted_image.png');
I_corr = rgb2gray(corrupted_image);
I2_corr = double(I_corr);
I2_corr( I2_corr==0 ) = 0.0001;

i_index = Index(size(I2_corr,1));
j_index = Index(size(I2_corr,2));
k_index = Index(10);

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
config = TFEngineConfig(model, 200);
engine = TFDefaultEngine(config, 'gtp');
engine.factorize();
plot(engine.beta_divergence');
check_divergence(engine.beta_divergence);

pred = squeeze(A.data) * squeeze(B.data)';

original_image = imread('/home/can/projeler/image_completion/original.jpg');
I_orig = rgb2gray(original_image);

sqrt( sum(sum( (pred - double(I_orig)).^2 ) )/prod(size(I_orig)))

imwrite(pred/100,'prediction.jpg')

I3 = I_corr;
I3(I3==0) = pred(I3==0);
imwrite(I3, 'prediction.png' )