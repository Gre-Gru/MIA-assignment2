import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import diff_ops


def load_image(filename):
    image = plt.imread(filename)
    # input image is color png depicting grayscale, just use first plane from here on
    image = image[:, :, 1].astype(np.float64)
    # after reading and converting to float, image is between 0 and 1, we want to compute within the range 0 and 255
    image *= 255.0
    #print(image.shape)
    #print('image: min = ', np.min(image), ' max = ', np.max(image))
    return image

def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    
    return np.sqrt(mse)

def laplace_operator(u):
    laplacian_x = diff_ops.dx_forward(diff_ops.dx_backward(u))
    laplacian_y = diff_ops.dy_forward(diff_ops.dy_backward(u))
    return laplacian_x + laplacian_y

def tikhonov_regularization(image, noisy_image, lamb, iterations, learn_rate, tol):
    denoised_image = noisy_image.copy()
    prev_image = noisy_image.copy()

    for i in range(iterations):
        laplacian = laplace_operator(denoised_image)
        gradient = -laplacian + lamb * (denoised_image - noisy_image)
        denoised_image -= learn_rate * gradient
        rmse_step = rmse(image, denoised_image)

        if (i + 1) % 100 == 0:
            current_rmse = rmse(image, denoised_image)
            print(f"Iteration {i + 1}/{iterations}, RMSE: {current_rmse}")

        difference = np.linalg.norm(denoised_image - prev_image) / np.linalg.norm(prev_image)

        if difference < tol:
            print(f"Convergence at iteration {i + 1} reached. Difference is {difference:.2e}. Lambda {lamb}, Iteration {i}, RMSE: {rmse_step}")
            break
        
        prev_image = denoised_image.copy()

    return denoised_image

def g_f(image, K):
   
  grad_x = diff_ops.dx_forward(image)
  grad_y = diff_ops.dy_forward(image)

  grad_mag = np.sqrt(grad_x**2 + grad_y**2)

  diffusion_coeff = 1 / (1 + (grad_mag / K)**2)

#   plt.imshow(diffusion_coeff, cmap='gray')
#   plt.title(f'K: {K}')
#   plt.show()

  return diffusion_coeff

def Perona_Malik(image, noisy_image, lamb, iterations, learn_rate, tol, K):

    denoised_image = noisy_image.copy()
    prev_image = noisy_image.copy()

    for i in range(iterations):

        grad_u_x = diff_ops.dx_forward(denoised_image) - diff_ops.dx_backward(denoised_image)
        grad_u_y = diff_ops.dy_forward(denoised_image) - diff_ops.dy_backward(denoised_image)

        div_g_grad_u_x = diff_ops.dx_backward(g_f(denoised_image, K) * grad_u_x)
        div_g_grad_u_y = diff_ops.dy_backward(g_f(denoised_image, K) * grad_u_y)

        divergence = (div_g_grad_u_x + div_g_grad_u_y) 

        denoised_image -= learn_rate * (-divergence + lamb * (denoised_image - noisy_image))
        rmse_step = rmse(image, denoised_image)

        if (i + 1) % 100 == 0:
            current_rmse = rmse(image, denoised_image)
            print(f"Iteration {i + 1}/{iterations}, RMSE: {current_rmse}, K: {K}")

        difference = np.linalg.norm(denoised_image - prev_image) / np.linalg.norm(prev_image)

        if difference < tol:
            print(f"Convergence at iteration {i + 1} reached. Difference is {difference:.2e}. Lambda {lamb}, Iteration {i}, RMSE: {rmse_step}, K: {K}")
            break
        
        prev_image = denoised_image.copy()

    return denoised_image


   
def main():
    filename = 'CTThoraxSlice256.png'

    # load image
    image = load_image(filename)
    # add noise with std. dev. 20 to input image
    img_noise = 20 * np.random.randn(image.shape[0], image.shape[1]) + image
    # clip noised image to [0,255]
    img_noise[img_noise > 255] = 255
    img_noise[img_noise < 0] = 0

    plt.imshow(img_noise, cmap='gray')
    plt.title('Noisy image')
    plt.savefig('Noisy.png')
    plt.show()

    # sigma of 3 for spatial smoothing, leads to strong blurring
    blurred = ndimage.gaussian_filter(image, sigma=3)

    print('RMSE (blurred):', rmse(image, blurred))
    plt.imshow(blurred, cmap='gray')
    plt.title('Gaussian blur')
    plt.savefig('Gauss.png')
    plt.show()

    iterations = 5000
    learn_rate = 0.1
    tolerance = 1e-4 
    lambdas = np.arange(0.01, 1.05, 0.05)

    best_lambda = None
    best_rmse = float('inf')
    best_image = None
    

    for i, lamb in enumerate(lambdas):
        denoised_image = tikhonov_regularization(image, img_noise, lamb, iterations, learn_rate, tolerance)
    
        # plt.imshow(denoised_image, cmap='gray')
        # plt.show()

        current_rmse = rmse(image, denoised_image)

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_lambda = lamb
            best_image = denoised_image

    print(f"Best RMSE Tikhonov {best_rmse}, best Lambda {best_lambda}")
    plt.imshow(best_image, cmap='gray')
    plt.title(f'Tikhonov; best Lmabda: {best_lambda}')
    plt.savefig('Best Tikhonov.png')
    plt.show()
    
    K = np.arange(0, 101, 5)
    K = K[1:]

    best_lambda = None
    best_rmse = float('inf')
    best_image = None
    best_K = None

    for i, Ks in enumerate(K):
        for i, lamb in enumerate(lambdas):
            denoised_image = Perona_Malik(image, img_noise, lamb, iterations, learn_rate, tolerance, Ks)
    
            # plt.imshow(denoised_image, cmap='gray')
            # plt.title(f'K: {Ks}, Lmabda: {lamb}')
            # plt.show()

            current_rmse = rmse(image, denoised_image)
        
          
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_lambda = lamb
                best_image = denoised_image
                best_K = Ks
    
    print(f"Best RMSE Perona-Malik {best_rmse}, best Lambda {best_lambda}, best K {best_K}")
    plt.imshow(best_image, cmap='gray')
    plt.title(f'Perona-Malik; best Lmabda: {best_lambda}, best K: {best_K}')
    plt.savefig('Best Perona-Malik.png')
    plt.show()
    # for i in K:
    #     grad = g_f(img_noise,i)

    grad_x_fwdiff = diff_ops.dx_forward(image)
    grad_y_fwdiff = diff_ops.dy_forward(image)
    
    fig, axarr = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)

    axarr[0, 0].imshow(image, cmap=plt.cm.gray)
    axarr[0, 0].set_title('Original Image')
    axarr[0, 1].imshow(blurred, cmap=plt.cm.gray)
    axarr[0, 1].set_title('Gaussian Blur')
    axarr[1, 0].imshow(grad_x_fwdiff, cmap=plt.cm.gray)
    axarr[1, 0].set_title('Original Image X Gradient')
    axarr[1, 1].imshow(grad_y_fwdiff, cmap=plt.cm.gray)
    axarr[1, 1].set_title('Original Image Y Gradient')

    plt.show()






if __name__ == "__main__":
    main()
