# Neural Network Model to Classify Images(MNIST)
## EDA

## Models
```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dense(10) # linear activation function
])
```

## Make Predictions
We can test the model's accuracy on few images from test dataset.  
But since our model is using the default 'linear activation function' we have to attach a softmax layer to convert the logits to probabilities, which are easier to interpret.

3https://www.kaggle.com/code/satishgunjal/neural-network-model-to-classify-images
