import numpy as np

# Gaussian noise
def add_gsnoise(Xsample, sigma = 0.05):
  Xsample_Gn = Xsample + np.random.normal(0, sigma, Xsample.shape)
  # Clip the pixel values to be between 0 and 255.
  Xsample_Gn = np.clip(Xsample_Gn, 0, 1)
  return Xsample_Gn.astype(np.float32)

# Impulsive noise
def add_psnoise(Xsample, level = 0.05):
  buffer = []
  for i in range(Xsample.shape[0]):
    sample = Xsample[i,0,:,:]
    # Get the image size (number of pixels in the image).
    img_size = sample.size

    # Set the percentage of pixels that should contain noise
    noise_percentage = level  # Setting to 10%

    # Determine the size of the noise based on the noise precentage
    noise_size = int(noise_percentage*img_size)

    # Randomly select indices for adding noise.
    random_indices = np.random.choice(img_size, noise_size)

    # Create a copy of the original image that serves as a template for the noised image.
    Xsample_In = sample.copy()

    # Create a noise list with random placements of min and max values of the image pixels.
    noise = np.random.choice([sample.min(), sample.max()], noise_size)

    # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
    Xsample_In.flat[random_indices] = noise
    buffer.append(Xsample_In.reshape(1, 1, 28, 28))
  return np.concatenate(buffer,axis=0)