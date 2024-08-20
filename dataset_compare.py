import os
import time
from mindspore import dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter

import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

# import torch
# import torchvision
# import torchvision.transforms as transforms

# import tensorflow as tf

data_dir = "/data1/datasets/ImageNet/train"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def mindspore_offload(batch_size, offload, dataset_num_workers=8):
    print("mindspore_offload.....")
    data_nums = 0
    start_time = time.time()
    dataset = ds.ImageFolderDataset(dataset_dir=data_dir, num_parallel_workers=dataset_num_workers, decode=False)
    dataset = dataset.map(operations=[C.Decode(), C.Resize((256, 256))], input_columns="image", num_parallel_workers=dataset_num_workers)

    if not offload:
        dataset = dataset.map(operations=[vision.RandomVerticalFlip(0.5),
                                          vision.RandomSharpness(),
                                          vision.Normalize(mean=mean, std=std),
                                          vision.HWC2CHW()], input_columns="image", num_parallel_workers=dataset_num_workers)
    dataset = dataset.batch(batch_size)

    for _ in dataset.create_tuple_iterator(output_numpy=True):
        data_nums = data_nums + 1
        if data_nums > 100:
            break
    end_time = time.time()
    cost_time = end_time - start_time

    print("mindspore cost time is: " + str(cost_time))
    return cost_time, dataset

# def pytorch_imagefolder(batch_size):
#     data_nums = 0
#     start_time = time.time()

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.RandomAdjustSharpness(1),
#         transforms.RandomVerticalFlip(),
#         transforms.Normalize(mean=mean, std=std)
#     ])
#     train_loader = torch.utils.data.DataLoader(
#         dataset=torchvision.datasets.ImageFolder(root=data_dir, transform=transform),
#         num_workers=32,
#         multiprocessing_context=None,
#         batch_size=batch_size
#     )
#     for _ in train_loader:
#         data_nums = data_nums + 1
#         if data_nums > 1000:
#             break
    
#     end_time = time.time()
#     cost_time = end_time - start_time
#     print("pytorch cost time is: " + str(cost_time))
#     return cost_time

# def tf_imagefolder(batch_size):
#     image_size = 224
#     data_nums = 0
#     tf.config.set_visible_devices([], 'GPU')
#     start_time = time.time()
#     train_filenames = [os.path.join(data_dir, classname, filename)
#                        for classname in os.listdir(data_dir)
#                        for filename in os.listdir(os.path.join(data_dir, classname))]
    
#     def random_sharpness(image, factor=0.5):
#         image = tf.cast(image, tf.float32)
#         blurred_image = tf.image.adjust_contrast(image, contrast_factor=0.5)
#         sharp_image = tf.add(tf.multiply(factor, image), tf.multiply(1.0-factor, blurred_image))

#         sharp_image = tf.cond(tf.random.uniform([], 0, 1) > 0.5,
#                               lambda: sharp_image,
#                               lambda: image)
#         return sharp_image
    
#     def parse_function(filename):
#         image_string = tf.io.read_file(filename)
#         image = tf.image.decode_jpeg(image_string, channels=3)
#         image = tf.image.resize(image, [256, 256])
#         image = tf.image.random_flip_up_down(image)
#         image = random_sharpness(image, factor=1.0)
#         image = (tf.cast(image, tf.float32) - mean) / std
#         image = tf.transpose(image, [2, 0, 1])
#         return image

#     train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
#     train_dataset = train_dataset.map(parse_function, num_parallel_calls=32)
#     train_dataset = train_dataset.batch(batch_size)

#     for _ in train_dataset:
#         data_nums = data_nums + 1
#         if data_nums > 1000:
#             break
#     end_time = time.time()
#     cost_time = end_time - start_time
#     print("tensorflow cost time is: " + str(cost_time))
#     return 
    
if __name__ == "__main__":
    from autotune_dataset import AutoTune
    _, dataset = mindspore_offload(64, False, 60)
    serialized_data = ds.serialize(dataset)
    autotune = AutoTune(serialized_data)
    params = [8, 64]
    res = autotune.optimize(params, init_points=2, n_iter=20)
    # print(res)
    # mindspore_offload(32, True, 32)
    # pytorch_imagefolder(32)
    # tf_imagefolder(32)

    # deserialized_dataset = ds.deserialize(json_filepath="/path/to/mnist_dataset_pipeline.json")