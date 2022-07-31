import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
from torch.autograd import Variable
import numpy as np
# import cv2
import collections
from utils.parameters import crop_size

from collections import OrderedDict
from scipy.ndimage import affine_transform

from torch.distributions.utils import _standard_normal

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')
ExpertSimTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert is_sim')

def featureExtractor():
    '''Creates a CNN module used for feature extraction'''
    return nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(1, 16, kernel_size=7)),
        ('relu0', nn.ReLU(True)),
        ('pool0', nn.MaxPool2d(2)),
        ('conv1', nn.Conv2d(16, 32, kernel_size=7)),
        ('relu1', nn.ReLU(True)),
        ('pool1', nn.MaxPool2d(2)),
        ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
        ('relu2', nn.ReLU(True)),
        ('pool2', nn.MaxPool2d(2))
    ]))

# def rotate(tensor, rad):
#   """
#   rotate the input tensor with the given rad
#   Args:
#     tensor: 1 x d x d image tensor
#     rad: degree in rad
#
#   Returns: 1 x d x d image tensor after rotation
#
#   """
#   img = transforms.ToPILImage()(tensor)
#   angle = 180./np.pi * rad
#   img = TF.rotate(img, angle)
#   return transforms.ToTensor()(img)

class TransformationMatrix(nn.Module):
    def __init__(self):
        super(TransformationMatrix, self).__init__()

        self.scale = torch.eye(3,3)
        self.rotation = torch.eye(3,3)
        self.translation = torch.eye(3,3)

    def forward(self, scale, rotation, translation):
        scale_matrix = self.scale.repeat(scale.size(0), 1, 1)
        rotation_matrix = self.rotation.repeat(rotation.size(0), 1, 1)
        translation_matrix = self.translation.repeat(translation.size(0), 1, 1)

        scale_matrix[:,0,0] = scale[:,0]
        scale_matrix[:,1,1] = scale[:,1]

        rotation_matrix[:,0,0] = torch.cos(rotation)
        rotation_matrix[:,0,1] = -torch.sin(rotation)
        rotation_matrix[:,1,0] = torch.sin(rotation)
        rotation_matrix[:,1,1] = torch.cos(rotation)

        translation_matrix[:,0,2] = translation[:,0]
        translation_matrix[:,1,2] = translation[:,1]

        return torch.bmm(translation_matrix, torch.bmm(rotation_matrix, scale_matrix))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        self.rotation = self.rotation.to(*args, **kwargs)
        self.translation = self.translation.to(*args, **kwargs)
        return self

class WeightedHuberLoss(nn.Module):
    ''' Compute weighted Huber loss for use with Pioritized Expereince Replay '''
    def __init__(self):
        super(WeightedHuberLoss, self).__init__()

    def forward(self, input, target, weights, mask):
        batch_size = input.size(0)
        batch_loss = (torch.abs(input - target) < 1).float() * (input - target)**2 + \
                     (torch.abs(input - target) >= 1).float() * (torch.abs(input - target) - 0.5)
        batch_loss *= mask
        weighted_batch_loss = weights * batch_loss.view(batch_size, -1).sum(dim=1)
        weighted_loss = weighted_batch_loss.sum() / batch_size

        return weighted_loss

def clip(tensor, min, max):
    '''
    Clip the given tensor to the min and max values given

    Args:
      - tensor: PyTorch tensor to clip
      - min: List of min values to clip to
      - max: List of max values to clip to

    Returns: PyTorch tensor like given tensor clipped to bounds
    '''
    clipped_tensor = torch.zeros_like(tensor)
    for i in range(len(min)):
        clipped_tensor[:,i] = torch.max(torch.min(tensor[:,i], torch.tensor(max[i])), torch.tensor(min[i]))
    return clipped_tensor

def argmax2d(tensor):
    '''
    Find the index of the maximum value in a 2d tensor.

    Args:
      - tensor: PyTorch tensor of size (n x 1 x d x d)

    Returns: nx2 PyTorch tensor containing indexes of max values
    '''
    n = tensor.size(0)
    d = tensor.size(2)
    m = tensor.view(n, -1).argmax(1)
    return torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1)

def argmax3d(tensor):
    n = tensor.size(0)
    c = tensor.size(1)
    d = tensor.size(2)
    m = tensor.contiguous().view(n, -1).argmax(1)
    return torch.cat(((m//(d*d)).view(-1, 1), ((m%(d*d))//d).view(-1, 1), ((m%(d*d))%d).view(-1, 1)), dim=1)

def argmax4d(tensor):
    n = tensor.size(0)
    c1 = tensor.size(1)
    c2 = tensor.size(2)
    c3 = tensor.size(3)
    c4 = tensor.size(4)
    m = tensor.reshape(n, -1).argmax(1)

    d0 = (m//(c4*c3*c2)).reshape(-1, 1)
    d1 = ((m%(c4*c3*c2))//(c4*c3)).reshape(-1, 1)
    d2 = (((m%(c4*c3*c2))%(c4*c3))//c4).reshape(-1, 1)
    d3 = (((m%(c4*c3*c2))%(c4*c3))%c4).reshape(-1, 1)

    return torch.cat((d0, d1, d2, d3), dim=1)

def argmax5d(tensor):
    n = tensor.size(0)
    c1 = tensor.size(1)
    c2 = tensor.size(2)
    c3 = tensor.size(3)
    c4 = tensor.size(4)
    c5 = tensor.size(5)
    m = tensor.reshape(n, -1).argmax(1)

    d0 = (m//(c5*c4*c3*c2)).reshape(-1, 1)
    d1 = ((m%(c5*c4*c3*c2))//(c5*c4*c3)).reshape(-1, 1)
    d2 = (((m%(c5*c4*c3*c2))%(c5*c4*c3))//(c5*c4)).reshape(-1, 1)
    d3 = ((((m%(c5*c4*c3*c2))%(c5*c4*c3))%(c5*c4))//c5).reshape(-1, 1)
    d4 = ((((m%(c5*c4*c3*c2))%(c5*c4*c3))%(c5*c4))%c5).reshape(-1, 1)

    return torch.cat((d0, d1, d2, d3, d4), dim=1)

def softUpdate(target_net, source_net, tau):
    '''
    Move target  net to source net a small amount

    Args:
      - target_net: net to copy weights into
      - source_net: net to copy weights from
      - tau: Amount to update weights
    '''
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def hardUpdate(target_net, source_net):
    '''
    Copy all weights from source net to target net

    Args:
      - target_net: net to copy weights into
      - source_net: net to copy weights from
    '''
    target_net.load_state_dict(source_net.state_dict())

def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

def randomCrop(imgs, out=64):
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = torch.empty((n, c, out, out), dtype=imgs.dtype).to(imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def centerCrop(imgs, out=64):
    n, c, h, w = imgs.shape
    top = (h - out) // 2
    left = (w - out) // 2

    imgs = imgs[:, :, top:top + out, left:left + out]
    return imgs

def bbox(img, threshold=0.011):
    rows = np.any(img>threshold, axis=1)
    cols = np.any(img>threshold, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def perturb(current_image, next_image, dxy, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    transformed_current_image = []
    for i in range(current_image.shape[0]):
        transformed_current_image.append(affine_transform(current_image[i], np.linalg.inv(transform), mode='nearest', order=1))
    transformed_current_image = np.stack(transformed_current_image)
    # current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
    if next_image is not None:
        transformed_next_image = []
        for i in range(next_image.shape[0]):
            transformed_next_image.append(affine_transform(next_image[i], np.linalg.inv(transform), mode='nearest', order=1))
        transformed_next_image = np.stack(transformed_next_image)
    else:
        transformed_next_image = None
    return transformed_current_image, transformed_next_image, rotated_dxy, transform_params

def perturbPixel(current_image, next_image, pixels, set_theta_zero=False, set_trans_zero=False):
  image_size = current_image.shape[-2:]
  if np.any(np.array(pixels) > image_size[0]-5) or np.any(np.array(pixels) < 5):
      set_theta_zero = True

  # Compute random rigid transform.
  while True:
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
      theta = 0.
    if set_trans_zero:
      trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    # Ensure pixels remain in the image after transform.
    is_valid = True
    new_pixels = []
    new_rounded_pixels = []
    for pixel in pixels:
      pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

      rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
      rounded_pixel = np.flip(rounded_pixel)

      pixel = (transform @ pixel)[:2].squeeze()
      pixel = np.flip(pixel)

      in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[
        1] < image_size[1]
      in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

      is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(
        pixel >= 0) and in_fov_rounded and in_fov

      new_pixels.append(pixel)
      new_rounded_pixels.append(rounded_pixel)
    if is_valid:
      break

  # Apply rigid transform to image and pixel labels.
  transformed_current_image = []
  for i in range(current_image.shape[0]):
      transformed_current_image.append(
          affine_transform(current_image[i], transform, mode='nearest', order=1))
  transformed_current_image = np.stack(transformed_current_image)
  if next_image is not None:
      transformed_next_image = []
      for i in range(next_image.shape[0]):
          transformed_next_image.append(
              affine_transform(next_image[i], transform, mode='nearest', order=1))
      transformed_next_image = np.stack(transformed_next_image)
  else:
      transformed_next_image = None

  return transformed_current_image, transformed_next_image, new_rounded_pixels[0], transform_params


  # current_image = cv2.warpAffine(
  #     current_image,
  #     transform[:2, :], (image_size[1], image_size[0]),
  #     flags=cv2.INTER_NEAREST)
  # if next_image is not None:
  #     next_image = cv2.warpAffine(
  #         next_image,
  #         transform[:2, :], (image_size[1], image_size[0]),
  #         flags=cv2.INTER_NEAREST)
  #
  # return current_image, next_image, new_rounded_pixels[0], transform_params

def perturbVec(current_state, next_state, dxy, set_theta_zero=False, set_trans_zero=False):
    assert not set_theta_zero
    assert set_trans_zero

    aug_current_state = current_state.copy()
    aug_next_state = next_state.copy()

    n_pose = (current_state.shape[0] - 1) // 4

    theta = np.random.random() * 2 * np.pi - np.pi
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    for i in range(n_pose):
        aug_current_state[1+i*4: 1+i*4+2] = rot.dot(current_state[1+i*4: 1+i*4+2])
        aug_next_state[1+i*4: 1+i*4+2] = rot.dot(next_state[1+i*4: 1+i*4+2])

        scaled_current_theta = current_state[1+i*4+3]
        unscaled_current_theta = (scaled_current_theta+1) * np.pi
        unscaled_aug_current_theta = unscaled_current_theta + theta
        if unscaled_aug_current_theta > np.pi:
            unscaled_aug_current_theta -= 2* np.pi
        if unscaled_aug_current_theta < -np.pi:
            unscaled_aug_current_theta += 2* np.pi
        aug_current_state[1 + i * 4 + 3] = 2 * (unscaled_aug_current_theta - -np.pi) / (2*np.pi) - 1

        scaled_next_theta = next_state[1+i*4+3]
        unscaled_next_theta = (scaled_next_theta+1) * np.pi
        unscaled_aug_next_theta = unscaled_next_theta + theta
        if unscaled_aug_next_theta > np.pi:
            unscaled_aug_next_theta -= 2* np.pi
        if unscaled_aug_next_theta < -np.pi:
            unscaled_aug_next_theta += 2* np.pi
        aug_next_state[1 + i * 4 + 3] = 2 * (unscaled_aug_next_theta - -np.pi) / (2*np.pi) - 1


        # aug_current_state[1+i*4+3] = (current_state[1+i*4+3]+1) * np.pi + theta
        # if aug_current_state[1+i*4+3] > np.pi:
        #     aug_current_state[1 + i * 4 + 3] -= 2* np.pi
        # if aug_current_state[1+i*4+3] < -np.pi:
        #     aug_current_state[1 + i * 4 + 3] += 2* np.pi
        # aug_next_state[1+i*4+3] = next_state[1+i*4+3] + theta

    return aug_current_state, aug_next_state, rotated_dxy, theta

def augmentDQNTransitionC4(d):
    t1_map = np.array([6, 3, 0,
                       7, 4, 1,
                       8, 5, 2])
    t2_map = np.array([8, 7, 6,
                       5, 4, 3,
                       2, 1, 0])
    t3_map = np.array([2, 5, 8,
                       1, 4, 7,
                       0, 3, 6])
    current_image = d.obs[0].copy()
    next_image = d.next_obs[0].copy()
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta_id = np.random.randint(0, 4)
    theta = theta_id * np.pi/2
    trans = [0., 0.]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    transform = get_image_transform(theta, trans, pivot)

    # Apply rigid transform to image and pixel labels.
    current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
    if next_image is not None:
        next_image = affine_transform(next_image, np.linalg.inv(transform), mode='nearest', order=1)

    action = d.action.copy()
    if theta_id == 1:
        action[1] = t1_map[action[1]]
    elif theta_id == 2:
        action[1] = t2_map[action[1]]
    elif theta_id == 3:
        action[1] = t3_map[action[1]]
    obs = current_image.reshape(1, *current_image.shape)
    next_obs = next_image.reshape(1, *next_image.shape)
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionSO2(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs.copy(),
                                                   d.next_obs.copy(),
                                                   d.action[1:3].copy(),
                                                   set_trans_zero=True)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionSO2Pixel(d):
    import matplotlib.pyplot as plt
    obs, next_obs, dxy, transform_params = perturbPixel(d.obs.copy(),
                                                   d.next_obs.copy(),
                                                   [d.action[1:3].copy()],
                                                   set_trans_zero=True)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)


def augmentTransitionSE2(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy())
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    action = d.action.clone()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)


def augmentTransitionTranslate(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy(),
                                                   set_theta_zero=True)
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionCnVec(d):
    obs, next_obs, dxy, transform_params = perturbVec(d.obs.copy(),
                                                      d.next_obs.copy(),
                                                      d.action[1:3].copy(),
                                                      set_trans_zero=True)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionShift(d):
    obs = d.obs[0]
    next_obs = d.next_obs[0]
    heightmap_size = obs.shape[-1]
    padded_obs = np.pad(obs, [4, 4], mode='edge')
    padded_next_obs = np.pad(next_obs, [4, 4], mode='edge')
    mag_x = np.random.randint(8)
    mag_y = np.random.randint(8)
    obs = padded_obs[mag_x:mag_x + heightmap_size, mag_y:mag_y + heightmap_size]
    next_obs = padded_next_obs[mag_x:mag_x + heightmap_size, mag_y:mag_y + heightmap_size]
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionCrop(d):
    obs = d.obs
    next_obs = d.next_obs
    heightmap_size = obs.shape[-1]

    crop_max = heightmap_size - crop_size + 1
    w1 = np.random.randint(0, crop_max)
    h1 = np.random.randint(0, crop_max)
    obs = obs[:, w1:w1 + crop_size, h1:h1 + crop_size]
    next_obs = next_obs[:, w1:w1 + crop_size, h1:h1 + crop_size]
    # obs = obs.reshape(1, *obs.shape)
    # next_obs = next_obs.reshape(1, *next_obs.shape)
    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)


def augmentTransition(d, aug_type):
    if aug_type == 'se2':
        return augmentTransitionSE2(d)
    elif aug_type == 'so2':
        return augmentTransitionSO2(d)
    elif aug_type == 'so2_pixel':
        return augmentTransitionSO2Pixel(d)
    elif aug_type == 't':
        return augmentTransitionTranslate(d)
    elif aug_type == 'dqn_c4':
        return augmentDQNTransitionC4(d)
    elif aug_type == 'cn_vec':
        return augmentTransitionCnVec(d)
    elif aug_type == 'shift':
        return augmentTransitionShift(d)
    elif aug_type == 'crop':
        return augmentTransitionCrop(d)
    else:
        raise NotImplementedError

def normalizeTransition(d: ExpertTransition):
    obs = np.clip(d.obs, 0, 0.32)
    obs = obs/0.4*255
    obs = obs.astype(np.uint8)

    next_obs = np.clip(d.next_obs, 0, 0.32)
    next_obs = next_obs/0.4*255
    next_obs = next_obs.astype(np.uint8)

    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state, next_obs, d.done, d.step_left, d.expert)

def augmentBuffer(buffer, aug_t, aug_n):
    aug_list = []
    for i, d in enumerate(buffer):
        for _ in range(aug_n):
            aug_list.append(augmentTransition(d, aug_t))
    for d in aug_list:
        buffer.add(d)


# code for this function from: https://github.com/facebookresearch/drqv2/blob/21e9048bf59e15f1018b49b850f727ed7b1e210d/utils.py#L105
class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
