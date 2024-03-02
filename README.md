# NoiseTypePrediction
<p> Simple and short project made to predict 4 different types of noise in images (including no noise): gaussian, speckle, salt and pepper, poisson (or shot noise).
</p>
<p> The dataset can be found here: https://www.kaggle.com/datasets/gpiosenka/100-bird-species, but the model was only trained on 10000 images. Also, it appears that some of the images in the dataset are already noisy, which may have biased the training, and the metrics.
</p>
<p> We compute the DWT and energy of each images that we feed into the network. We also add additional information to the fully connected layers, including: skewness, kurtosis, entropy, correlation.
</p>
<p> Here are the results training with the "UNetFC" model, which was obtained in 29 epochs.
</p>
| Type of noise | Recall | Precision | F1 Score |
| ------------- | ------ | --------- | -------- |
| None          | 0.82   | 0.76      | 0.79     |
| Gaussian      | 0.89   | 0.79      | 0.84     |
| Speckle       | 0.66   | 0.78      | 0.71     |
| Salt & Pepper | 1.00   | 1.00      | 1.00     |
| Poisson       | 0.67   | 0.71      | 0.69     |