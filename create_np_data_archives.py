import numpy as np
from imageio import imread

# Train data
with open('/mnt/grocery_data/Traderjoe/StPaul/strongly_labeled_train.txt') as f:
    lines = f.readlines()
    train_images, train_llabels, train_rlabels = [], [], []
    for line in lines:
        img, labels = line.strip().split(' ')
        train_images.append(imread(img))
        llabel, rlabel = labels.split(':')
        train_llabels.append(llabel)
        train_rlabels.append(rlabel)
    images = np.stack(train_images)
    llabels = np.stack(train_llabels)
    rlabels = np.stack(train_rlabels)
    import ipdb; ipdb.set_trace()
    np.savez('train.npz', images=images, llabels=llabels, rlabels=rlabels)
# Test data
with open('/mnt/grocery_data/Traderjoe/StPaul/strongly_labeled_test.txt') as f:
    lines = f.readlines()
    test_images, test_llabels, test_rlabels = [], [], []
    for line in lines:
        img, labels = line.strip().split(' ')
        test_images.append(imread(img))
        llabel, rlabel = labels.split(':')
        test_llabels.append(llabel)
        test_rlabels.append(rlabel)
    images = np.stack(test_images)
    llabels = np.stack(test_llabels)
    rlabels = np.stack(test_rlabels)
    np.savez('test.npz', images=images, llabels=llabels, rlabels=rlabels)
