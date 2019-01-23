"""
Helps load images and labels (segmentations) from a filesystem path, under which:
- images are expected to be found under "leftImg8bit/{phase}/*/resize*.png"
- segmentation labels are to be found under "gtFine/{phase}/*/resize*labelTrainIds.png"

and 'phase' is one of 'train', 'test' or 'val'.

Validates existence of expected directories and files.

Helps merge together multiple datasets into one conceptual one.

Calculates and caches the mean image from the training set.

"""
from glob import glob
import os
import math
import numpy as np
import random

from multiprocessing import Pool
from numpy_image_utils import fit_image
from PIL import Image
from statistics import mode
import pdb

class GTADataset_VOC:

    def __init__(self, voc_path_info, *, cache_dir, calc_val_mean):
        """
        :param paths: List of paths to load datasets from. Each item may optionally be a tuple, the 2nd item
            indicating the maximum number of training images to load, the 3rd indicating the slug (or None to
            base on path)
            e.g ['/path/to/dataset', ('/path/to/other/dataset', 1000, 'dataset-slug')]
        :param cache_dir: '/path/to/store/cached/mean/images'
        :param calc_val_mean: whether to calculate the mean image of the validation dataset
        """
        if not os.path.isdir(cache_dir):
            raise ValueError('{} is not a directory.'.format(self.cache_dir))
        self.cache_dir = cache_dir

        ##
        #def normalize_path_image(path_tuple):
        #    if not isinstance(path_tuple, (list, tuple)):
        #        return path_tuple, None, None
        #    if not len(path_tuple) == 3:
        #        raise ValueError("Expected path_tuple to be of length 3 but is: {}".format(path_tuple))
        #    return path_tuple
        #    ##
        #paths = [normalize_path_image(p) for p in paths]
        #datasets = [load_dataset(path, max_train_images, slug) for path, max_train_images, slug in paths]
        path = voc_path_info[0][0]
        if not os.path.isdir(path):
            raise ValueError("Expected {} to be a directory".format(path))
            ##
        #input_fname_pattern_real = '*.jpg'
        #voc_imgs = os.path.join(voc_path, 'VOC2012/JPEGImages')
        #dataset_voc = sorted(glob(os.path.join(voc_imgs, input_fname_pattern_real)))
        #dataset_voc = [fn for fn in dataset_voc  if 'aug' not in fn]
        datasets = [load_dataset(voc_path, max_train_images, slug) for voc_path, max_train_images, slug in voc_path_info]
        ##
        self.train_image_channel_mean = load_or_calculate_image_channel_mean(datasets=datasets, cache_dir=cache_dir)
        ##
        if calc_val_mean:
            self.val_image_channel_mean = load_or_calculate_image_channel_mean(datasets=datasets, cache_dir=cache_dir, segment='val')
        else:
            self.val_image_channel_mean = None
            ##
        #merged_dataset = merge_datasets(datasets)
        #self.slug = merged_dataset['slug']
        #self.train = merged_dataset['train']
        #self.val = merged_dataset['val']
        #pdb.set_trace()
        self.slug = datasets[0]['slug']
        self.train = datasets[0]['train']
        self.val = datasets[0]['val']


def load_dataset(voc_path, max_train_images=None, slug=None):
    """
    Loads a dataset from a directory according to conventions of pascal voc.

    The slug of the returned dataset is based on the path and max train images provided. For instance,
    if the `path` is '/path/to/my-dataset' and `max_train_images` is 100, the slug is 'my-dataset+100'.
    This is useful for constructing a unique cache name.

    :param path: /path/to/dataset
    :param max_train_images:
    :param slug:
    :return: {'slug': 'slug-based-on-path',
              'train': {'images': ['/path/to/image.png', ...], 'labels': ['path/to/label.png', ...],
              'val': {'images': ['/path/to/image.png', ...], 'labels': ['path/to/label.png', ...]
             }
    """
    if not os.path.isdir(voc_path):
        raise ValueError("Expected {} to be a directory".format(path))
        ##
    input_fname_pattern_real = '*.jpg'
    voc_imgs = os.path.join(voc_path, 'VOC2012/JPEGImages')
    dataset_voc = sorted(glob(os.path.join(voc_imgs, input_fname_pattern_real)))
    dataset_voc = [fn for fn in dataset_voc  if 'aug' not in fn]

    #images_dir = "{}/leftImg8bit".format(path)
    #if not os.path.isdir(images_dir):
    #    raise ValueError("Expected to find images directory at {}".format(images_dir))
    #    ##
    ##labels_dir = "{}/gtFine2".format(path)
    #labels_dir = "{}/gtFine".format(path)
    #if not os.path.isdir(labels_dir):
    #    raise ValueError("Expected to find labels directory at {}".format(labels_dir))

    def file_slug(path):
        """
        E.g '/path/to/file_leftImg8bit.png' -> 'file'
        """
        slug = path.strip('/').split('/')[-1].split('.')[0]
        #suffixes = ['_gtFine_leftImg8bit', '_leftImg8bit','_leftImg8bit_augx1','_gtFine_leftImg8bit_augx1', '_gtFine_labelIds','_gtFine_labelIds_augx1', '_labelIds','_labelIds_augx1']
        #suffixes = ['_gtFine_leftImg8bit', '_leftImg8bit', '_gtFine_labelIds', '_labelIds']
        suffixes = ['VOC2012/JPEGImages']
        for suffix in suffixes:
            if slug.endswith(suffix):
                slug = slug.split(suffix)[0]
        return slug
        #
    #def load_subset(subset, limit=None,):
    #    images_subset_dir = "{}/{}".format(images_dir, subset)
    #    labels_subset_dir = "{}/{}".format(labels_dir, subset)
    #    for expected_dir in [images_subset_dir, labels_subset_dir]:
    #        if not os.path.isdir(expected_dir):
    #            raise ValueError("Expected to directory at {}".format(expected_dir))
    #    images = sorted(glob.glob('{}/*.jpg'.format(images_subset_dir)))
    #    #labels = sorted(glob.glob('{}/*.jpg'.format(labels_subset_dir)))
    #    #
    #    # prune down to files that are in common (ok if there are missing images, or
    #    # we only have labels for a subset of the images)
    #    #
    #    image_slugs = set([file_slug(f) for f in images])
    #    #label_slugs = set([file_slug(f) for f in labels])
    #    common_slugs = image_slugs #& label_slugs
    #    #
    #    images = [f for f in images if file_slug(f) in common_slugs]
    #    #labels = [f for f in labels if file_slug(f) in common_slugs]
    #    #
    #    #pdb.set_trace()
    #    original_len = len(images)
    #    if limit:
    #        images = images[:limit]
    #        labels = labels[:limit]
    #    if len(images) != len(labels):
    #        raise ValueError("Found different number of images and labels at {} and {}\n({} vs {})".format(
    #            images_subset_dir, labels_subset_dir,
    #            len(images), len(labels)))
    #    if len(images) == 0:
    #        raise ValueError("Found zero common images between {} and {}".format(images_subset_dir, labels_subset_dir))
    #    mismatches = [(image, label) for image, label in zip(images, labels) if file_slug(image) != file_slug(label)]
    #    if mismatches:
    #        mismatches_summary = "\n\n".join(["{}: {}\nvs\n{}: {}".format(file_slug(image), image, file_slug(label), label) for image, label in mismatches])
    #        raise ValueError("Image label mismatches within {}:\n{}".format(path, mismatches_summary))
    #    result = {
    #        'images': images,
    #        'labels': labels
    #    }
    #    #if subset == 'train-all-data':
    #    if subset == 'train-all-data_lblIDs':
    #        result['limited'] = limit is not None and limit < original_len
    #    return result
    #    ##
    ####################################
    ## MAIN FUNCTION for load_dataset ##
    ####################################
    #subset_name = 'train-all-data_lblIDs'
    #train_subset = load_subset(subset_name, max_train_images)
    train_subset = dataset_voc
    ##
    if not slug:
        slug = path.strip('/').split('/')[-1]
        if train_subset['limited']:
            slug = "{}+{}".format(slug, max_train_images)
            #
    #del train_subset['limited']
    ##
    return {
        'slug': slug,
        'train': train_subset,
        'val': None
    }


def merge_datasets(datasets):
    def merge_subsets(subsets):
        return {'images': sum([s['images'] for s in subsets], []), 'labels': sum([s['labels'] for s in subsets], [])}
    pdb.set_trace()
    return {
        'slug': '_'.join([ds['slug'] for ds in datasets]),
        'train': merge_subsets([ds['train'] for ds in datasets]),
        'val': merge_subsets([ds['val'] for ds in datasets])
    }


def load_or_calculate_image_channel_mean(*, datasets, cache_dir, segment='train'):
    """
    Loads the the per image channel mean from the provided datasets, generating an caching mean images
    along the way if necessary. The mean images are calculated and cached per dataset so that different
    combinations can be calculated with minimal recomputation.

    :param datasets: [{'slug': 'dataset_slug', 'train': {'images': ['/path/to/image.png', ...]}}, ...]
    :param cache_dir: '/path/to/cache_dir'
    :return: rgb mean tuple e.g (95.3, 122.4, 100.4)
    """

    if segment not in ['train', 'val']:
        raise "{} must be one of 'train', 'val'".format(segment)

    segment_name_postfix = '' if segment is 'train' else '-val'

    def load_image_mean(dataset):
        #pdb.set_trace()
        cache_file_path_np = "{}/{}{}.npy".format(cache_dir, dataset['slug'], segment_name_postfix)
        cache_file_path_png = "{}/{}{}.png".format(cache_dir, dataset['slug'], segment_name_postfix)
        if os.path.exists(cache_file_path_np):
            print("loading cached image {}".format(cache_file_path_np))
            return np.load(cache_file_path_np)
        else:
            print("calculating and caching image mean to {}, {}".format(cache_file_path_np, cache_file_path_png))
            # mean_image = calculate_mean_image(dataset['train']['images'])
            #pdb.set_trace()
            #mean_image = calculate_mean_image_multi(dataset[segment]['images'])
            mean_image = calculate_mean_image_multi(dataset[segment])
            save_image(mean_image, np_path=cache_file_path_np, png_path=cache_file_path_png)
            return mean_image

    mean_images = [load_image_mean(dataset) for dataset in datasets]
    channel_means = [image_channel_mean(mean_image) for mean_image in mean_images]
    n = len(datasets)
    return (
        sum([r for r, g, b in channel_means]) / n,
        sum([g for r, g, b in channel_means]) / n,
        sum([b for r, g, b in channel_means]) / n,
    )


def calculate_mean_image_multi(image_paths, parallelism=None):
    image_shape = best_image_shape(image_paths)
    if not parallelism:
        parallelism = os.cpu_count()
    num_images = len(image_paths)
    chunksize = math.ceil(num_images / parallelism)
    with Pool(processes=parallelism) as pool:
        tasks = [pool.apply_async(calculate_image_sum, args=(image_paths[i: i+chunksize], image_shape))
                 for i in range(0, num_images, chunksize)]
        image_sums = [task.get() for task in tasks]
        result_mu = np.zeros(image_shape, dtype=np.float32)
        for image_sum in image_sums:
            result_mu += image_sum
        result_mu /= len(image_paths)
        return result_mu


def calculate_mean_image(image_paths):
    image_shape = best_image_shape(image_paths)

    return calculate_image_sum(image_paths, image_shape) / len(image_paths)


def calculate_image_sum(image_paths, image_shape):
    h, w = image_shape[:2]
    image_sum = np.zeros(image_shape)
    num_images = len(image_paths)
    for k, image_path in enumerate(image_paths):
        image = fit_image(np.array(Image.open(image_path)), w, h)
        image_sum += image
        print('[{:6d}/{:6d}] {}'.format(k + 1, num_images, image_path.split('/')[-1]))
    return image_sum


def save_image(image, *, np_path, png_path):
    int_image = np.uint8(np.around(image))
    np.save(np_path, int_image)
    Image.fromarray(int_image).save(png_path)


def image_channel_mean(image):
    return np.mean(np.mean(image, axis=0), axis=0)


def best_image_shape(image_paths):
    def mode_dimensions(image_paths):
        def dims(image_path):
            image = Image.open(image_path)
            return image.width, image.height

        return mode([dims(image_path) for image_path in image_paths])

    width, height = mode_dimensions(random.sample(image_paths, min(10, len(image_paths))))
    return (height, width, 3)
