from os import path, walk, remove
from multiprocessing import Process, Pool, cpu_count
from numpy import array_split
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
from utils import DATA_PATH, try_makedirs


def apply_transformation(files):
    for head, tail, f_name in files:
        f = path.join(head, tail, f_name)
        try:
            print('Processing: ', f)
            img = imread(f)
        except:
            print('CANNOT OPEN: ', f)
            continue
        if img.shape[0] < 10 or img.shape[1] < 10:
            print('TOO SMALL: ', f)
            continue
        dst_dir = path.join(head + '_p', tail)
        if not path.exists(dst_dir):
            try_makedirs(dst_dir)
        img = resize(img, (256, 256, 3))
        img = equalize_adapthist(img, clip_limit=0.03)
        imsave(path.join(dst_dir, f_name), img)


def process(top_dir):
    stage_name = path.basename(top_dir)
    print('Processing {} data'.format(stage_name))
    img_paths = []
    for root, dirs, files in walk(top_dir):
        head, tail = path.split(root)
        img_paths += [(head, tail, f) for f in files]
    cpus = cpu_count()
    pool = Pool(processes=cpus)
    files_split = array_split(img_paths, cpus)
    pool.map(apply_transformation, files_split)


train_proc = Process(target=process, args=(path.join(DATA_PATH, 'train'), ))
train_proc.start()
val_proc = Process(target=process, args=(path.join(DATA_PATH, 'validation'), ))
val_proc.start()
test_proc = Process(target=process, args=(path.join(DATA_PATH, 'test'), ))
test_proc.start()

train_proc.join()
val_proc.join()
test_proc.join()
