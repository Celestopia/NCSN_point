import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
from torch.utils.data import DataLoader
from torchvision.models import inception_v3




def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def transform_image(images:torch.Tensor, use_imagenet_norm=True):
    """
    Transform images to shape (batch_size, 3, 299, 299) and normalize pixel values.
    
    Args:
        images (torch.Tensor): Images of shape (batch_size, channels, height, width). Typically (batch_size, 3, 32, 32).
    
    Returns:
        transformed_images (torch.Tensor): Batch of transformed images of shape (batch_size, 3, 299, 299).
    """
    assert images.ndim == 4, "Images should be of shape (batch_size, channels, height, width)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    transformed_images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False) # Shape: (batch_size, 3, 299, 299)
    
    if use_imagenet_norm:
        device = images.device
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
        imagenet_std=torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
        transformed_images = (transformed_images - imagenet_mean) / imagenet_std
    
    return transformed_images


def extract_features(images:torch.Tensor, model, device):
    """
    Extract features from images using Inception v3
    
    Args:
        images (torch.Tensor): Batch of images of shape (batch_size, 3, H, W).
        model (nn.Module): Pretrained Inception v3 model.
        device (torch.device): Computation device.
    
    Returns:
        features (np.ndarray): Feature matrix of shape (n_test_samples, 2048).
    """
    dataloader = DataLoader(images, batch_size=128, shuffle=False)

    model.eval()
    features = []
    with torch.no_grad(): # Disable gradient calculation
        for images in dataloader:
            images = transform_image(images, use_imagenet_norm=True)
            images = images.to(device) # Shape: (batch_size, 3, 299, 299)
            feature = model(images).detach().cpu().numpy() # Shape: (batch_size, 2048)
            features.append(feature)
    return np.concatenate(features, axis=0) # Shape: (n_test_samples, 2048)


def sample_fid(real_features, gen_features, eps=1e-6):
    """
    Calculate Frechet Inception Distance between two groups of samples (features).
    
    Args:
        real_features (np.ndarray): Real feature matrix of shape (N, d). For Inception v3 model, d=2048.
        gen_features (np.ndarray): Generated feature matrix of shape (N, d). For Inception v3 model, d=2048.
        eps (float): Small epsilon value for numerical stability of covariance calculation.

    Returns:
        fid (float): Frechet Inception Distance (FID).
    """
    assert real_features.shape[1] == gen_features.shape[1]
    
    # Calculate sample mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[-1]) * eps
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False) + np.eye(gen_features.shape[-1]) * eps
    
    return frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)



def sample_inception_score(dataloader, model, device, n_splits=10):
    """
    Calculate Inception Score for generated images
    
    Args:
        dataloader (DataLoader): Loader for generated images. Each batch is of shape (batch_size, 3, H, W).
        model (nn.Module): Pretrained Inception v3 model.
        device (torch.device): Computation device.
        n_splits (int): Number of splits for score calculation.
    
    Returns:
        (mean score, standard deviation) (tuple):
    """
    # Get softmax probabilities for all images
    probs = []
    model.eval()
    with torch.no_grad():
        for images in tqdm.tqdm(dataloader, desc="Function <sample_inception_score> processing images"):
            images = transform_image(images, use_imagenet_norm=True) # (batch_size, 3, 299, 299)
            images = images.to(device)
            logits = model(images) # Shape: (batch_size, 1000)
            preds = nn.functional.softmax(logits, dim=1)
            probs.append(preds.cpu().detach().numpy())
    probs = np.concatenate(probs, axis=0) # Shape: (n_samples, d)
    split_size = len(probs) // n_splits
    
    scores = []
    for i in range(n_splits): # Calculate score for each split
        subset = probs[i*split_size : (i+1)*split_size] # Split data into chunks
        py = np.mean(subset, axis=0) # Shape: (d,); marginal probability p(y)
        
        # Calculate KL divergence for each sample
        kl = subset * (np.log(subset) - np.log(py))
        kl = np.sum(kl, axis=1)

        average_kl = np.mean(kl) # Scalar
        is_score = np.exp(average_kl) # Scalar
        scores.append(is_score)
    
    return np.mean(scores), np.std(scores)





def fast_2sample_fid(gen_samples:torch.Tensor, real_samples:torch.Tensor, model=None, device=None):
    """
    Calculate Frechet Inception Distance between two groups of samples (images).
    
    Args:
        real_samples (torch.Tensor): Real images of shape (N, 3, H, W).
        gen_samples (torch.Tensor): Generated images of shape (N, 3, H, W).
        model (nn.Module): Pretrained Inception v3 model. If None, use the default Inception v3 model.
        device (torch.device): Computation device. If None, use CUDA if available.
    
    Returns:
        fid (float): Frechet Inception Distance (FID).
    """
    assert gen_samples.ndim == real_samples.ndim == 4, "Images should be of shape (batch_size, channels, height, width)"
    assert gen_samples.shape[1] == real_samples.shape[1] == 3, "Images should have 3 channels"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        model = inception_v3(pretrained=True, transform_input=False, aux_logits=True).to(device) # Must set aux_logits=True for pretrained 
        model.fc = nn.Identity() # Take the output of the penultimate layer
        print("Loaded Inception v3 model.")
    real_samples = real_samples.to(device)
    gen_samples = gen_samples.to(device)
    model.eval()
    
    real_features = extract_features(real_samples, model, device) # Shape: (n_real_samples, 2048)
    gen_features = extract_features(gen_samples, model, device) # Shape: (n_gen_samples, 2048)
    
    return sample_fid(real_features, gen_features)



def fast_sample_inception_score(gen_samples:torch.Tensor, model=None, device=None, n_splits=10):
    """
    Calculate Inception Score for generated images.
    
    Args:
        gen_samples (torch.Tensor): Generated images of shape (N, 3, H, W).
        model (nn.Module): Pretrained Inception v3 model. If None, use default Inception v3 model.
        device (torch.device): Computation device. If None, use CUDA if available.
        n_splits (int): Number of splits for score calculation.
    
    Returns:
        (mean score, standard deviation) (tuple):
    """
    assert gen_samples.ndim == 4, "Images should be of shape (batch_size, channels, height, width)"
    assert gen_samples.shape[1] == 3, "Images should have 3 channels"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        model = inception_v3(pretrained=True, transform_input=False, aux_logits=True).to(device) # Must set aux_logits=True for pretrained 
        print("Loaded Inception v3 model.")
    gen_samples = gen_samples.to(device)
    model.eval()
    
    dataloader = DataLoader(gen_samples, batch_size=128, shuffle=False)
    return sample_inception_score(dataloader, model, device, n_splits=n_splits)



def fast_get_fid_and_is(gen_samples, real_samples, inception_v3_model=None, device=None, num_workers=0):
    """
    Args:
        gen_samples (torch.Tensor): generated samples of shape (n_samples, 3, H, W).
        real_samples (torch.Tensor): real samples of shape (n_samples, 3, H, W).
        inception_v3_model: inception v3 model, loaded from torchvision.models.
        device: device to run the model on.
        num_workers: number of workers to use for data loading.
    
    Returns:
        (fid, is_mean, is_std) (tuple):
        - Frechet Inception Distance between the generated and real samples.
        - Inception Score of the generated samples.
        - Inception Score standard deviation of the generated samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inception_v3_model is None:
        inception_v3_model = inception_v3(pretrained=True, transform_input=False, aux_logits=True).to(device)
    inception_v3_model.eval()

    # Extract features
    gen_loader = DataLoader(gen_samples, batch_size=64, shuffle=False, num_workers=num_workers)
    gen_features = extract_features(gen_samples, inception_v3_model, device) # (n_gen_samples, 2048)
    real_features = extract_features(real_samples, inception_v3_model, device) # (n_real_samples, 2048)

    # Calculate FID
    fid = sample_fid(real_features, gen_features)
    is_mean, is_std = sample_inception_score(gen_loader, inception_v3_model, device=device, n_splits=10)

    return fid, is_mean, is_std





if __name__ == '__main__':
    os.environ['TORCH_HOME'] = './model_weights' # set the path to save inception v3 model weights.
    real_samples = torch.randn(1000, 3, 32, 32)
    gen_samples = torch.randn(1000, 3, 32, 32)
    #fid = fast_2sample_fid(real_samples, gen_samples, device=torch.device("cuda"))
    #print(fid)
    #is_mean, is_std = fast_sample_inception_score(gen_samples, device=torch.device("cuda"))
    #print(is_mean, is_std)
    #sample_path = r"E:\PythonProjects\NCSN\ncsnv2\images\samples_300000_0.pth"
    #all_generated_samples = torch.load(sample_path)
    #print(all_generated_samples.shape)
    #print(fast_sample_inception_score(all_generated_samples, device=torch.device("cuda")))
    print(fast_get_fid_and_is(gen_samples, real_samples))






