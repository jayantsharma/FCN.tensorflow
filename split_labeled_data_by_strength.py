import os, sys, glob


store_dir = sys.argv[1]
label_file = '{}/labels.txt'.format(store_dir)
fps = 25 if store_dir.endswith('StPaul') else 30
train_data, test_data = [], []
with open(label_file) as f:
    lines = f.readlines()
    times = [ 60*m + s for line in lines for (m,s) in [tuple(map(int, (line.split(' ')[0].split(':'))))] ]

    for i in range(len(times)-1):
        st = times[i]
        et = times[i+1]
        _, labels = tuple(lines[i].strip().split(' '))
        sframe, eframe = (3*st + et)/4, (st + 3*et)/4
        sframe *= fps;  eframe *= fps
        frames = list(range(int(sframe)+1, int(eframe)+1))
        split = int(.9 * len(frames))   # 90:10 train:test split
        for frame in frames[:split]:
            train_data.append('{}/image/image{:07d}.jpg {}\n'.format(store_dir, frame, labels))
        for frame in frames[split:]:
            test_data.append('{}/image/image{:07d}.jpg {}\n'.format(store_dir, frame, labels))

with open('{}/strongly_labeled_train.txt'.format(store_dir), 'w') as f:
    for line in train_data:
        f.write(line)
with open('{}/strongly_labeled_test.txt'.format(store_dir), 'w') as f:
    for line in test_data:
        f.write(line)
