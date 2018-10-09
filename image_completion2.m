clear all;
cd /home/can/projeler/tft
setup_tft
cd /home/can/projeler/image_completion

corrupted_image = imread('/home/can/projeler/image_completion/corrupted_image.png');
I2_corr = double(corrupted_image);
I2_corr( I2_corr==0 ) = 0.0001;

i_index = Index(size(I2_corr,1));
j_index = Index(size(I2_corr,2));
k_index = Index(size(I2_corr,3));
r_index = Index(10);


X = Tensor(i_index, j_index, k_index);
X.data = I2_corr;

A=Tensor(i_index, r_index);
B=Tensor(j_index, r_index);
C=Tensor(k_index, r_index);

A.data = rand(i_index.cardinality, r_index.cardinality);
B.data = rand(j_index.cardinality, r_index.cardinality);
C.data = rand(k_index.cardinality, r_index.cardinality);

pre_process();

p = [1];
phi = [1];

factorization_model = {X, {A, B, C}};

model = TFModel(factorization_model, p, phi);
config = TFEngineConfig(model, 2000);
engine = TFDefaultEngine(config, 'gtp');
engine.factorize();
plot(engine.beta_divergence');
check_divergence(engine.beta_divergence);

pred = create_tensor( cellfun( @(index) index.id, X.indices ), 'zeros' );
gtp(pred, A, B, C);

original_image = imread('/home/can/projeler/image_completion/original.jpg');

sqrt( sum(sum( (pred.data - double(original_image)).^2 ) )/prod(size(original_image)))

imwrite(pred.data,'prediction1.png')

I3 = corrupted_image;
I3(I3==0) = pred.data(I3==0);
imwrite(I3, 'prediction2.png' )