# NoiseTypePrediction
<p> Simple and short project made to predict 4 different types of noise in images (including no noise): gaussian, speckle, salt and pepper, poisson (or shot noise).
</p>
<p> The dataset can be found here: https://www.kaggle.com/datasets/gpiosenka/100-bird-species, but the model was only trained on 10000 images. Also, it appears that some of the images in the dataset are already noisy, which may have biased the training, and the metrics.
</p>
<p> We compute the DWT and energy of each images that we feed into the network. We also add additional information to the fully connected layers, including: skewness, kurtosis, entropy, correlation.
</p>
<p> Here are the results training with the "UNetFC3" model on SNR levels [10 - 15 - 20], which was obtained in 29 epochs.
</p>

| Type of noise | Recall | Precision | F1 Score |
| ------------- | ------ | --------- | -------- |
| None          | 0.9941 | 0.9882    | 0.9911   |
| Gaussian      | 0.9751 | 0.9416    | 0.9580   |
| Speckle       | 0.9088 | 0.9573    | 0.9324   |
| Salt & Pepper | 1.0000 | 1.0000    | 1.0000   |
| Poisson       | 0.9866 | 0.9768    | 0.9817   |
