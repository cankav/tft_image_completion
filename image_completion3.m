clear all;
cd /home/can/projeler/tft
setup_tft
cd /home/can/projeler/tft_image_completion

corrupted_image = imread('/home/can/projeler/tft_image_completion/corrupted_image.png');
I2_corr = double(corrupted_image);
I2_corr( I2_corr==0 ) = 0.0001;

r_index = Index(255);
g_index = Index(255);
b_index = Index(255);
i_index = Index(size(I2_corr,1));
j_index = Index(size(I2_corr,2));
r_index = Index(10);

X = Tensor(r_index, g_index, ...
               b_index, i_index, ...
               j_index);
X.data = zeros(r_index.cardinality, g_index.cardinality, ...
               b_index.cardinality, i_index.cardinality, ...
               j_index.cardinality);

for i=1:size(I2_corr,1)
    for j=1:size(I2_corr,2)
        for c=1:3
            rgb_values = I2_corr(i, j, :);
            X.data( rgb_values(1), rgb_values(2), rgb_values(3), i, j ) = 1;
        end
    end
end

A=Tensor(r_index, r_index);
B=Tensor(g_index, r_index);
C=Tensor(b_index, r_index);
D=Tensor(i_index, r_index);
E=Tensor(j_index, r_index);

A.data = rand(r_index.cardinality, r_index.cardinality);
B.data = rand(g_index.cardinality, r_index.cardinality);
C.data = rand(b_index.cardinality, r_index.cardinality);
D.data = rand(i_index.cardinality, r_index.cardinality);
E.data = rand(j_index.cardinality, r_index.cardinality);

pre_process();

p = [1];
phi = [1];

factorization_model = {X, {A, B, C, D, E}};

model = TFModel(factorization_model, p, phi);
config = TFEngineConfig(model, 200);
engine = TFDefaultEngine(config, 'gtp');
engine.factorize();
plot(engine.beta_divergence');
check_divergence(engine.beta_divergence);

pred = create_tensor( cellfun( @(index) index.id, X.indices ), 'zeros' );
gtp(pred, A, B, C, D, E);

original_image = imread('/home/can/projeler/tft_image_completion/original.jpg');
orig_image_data = zeros(r_index.cardinality, g_index.cardinality, ...
                        b_index.cardinality, i_index.cardinality, ...
                        j_index.cardinality);

for i=1:size(I2_corr,1)
    for j=1:size(I2_corr,2)
        for c=1:3
            rgb_values = original_image(i, j, :);
            orig_image_data( rgb_values(1), rgb_values(2), rgb_values(3), i, j ) = 1;
        end
    end
end


sqrt( sum(sum( (pred.data - double(orig_image_data)).^2 ) )/prod(size(orig_image_data)))


%imwrite(pred.data,'prediction1.png')

%I3 = corrupted_image;
%I3(I3==0) = pred.data(I3==0);
%imwrite(I3, 'prediction2.png' )