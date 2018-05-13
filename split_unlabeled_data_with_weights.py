import os, sys, glob
import random


store_dir = sys.argv[1]
label_file = '{}/labels.txt'.format(store_dir)
fps = 25 if store_dir.endswith('StPaul') else 30
train_data, test_data = [], []
with open(label_file) as f:
    lines = f.readlines()
    times = [ 60*m + s for line in lines for (m,s) in [tuple(map(int, (line.split(' ')[0].split(':'))))] ]

    for i in range(len(times)-2):
        st = times[i]
        mt = times[i+1]
        et = times[i+2] 
        _, pre_labels = tuple(lines[i].strip().split(' '))
        _, post_labels = tuple(lines[i+1].strip().split(' '))

        sframe, eframe = (st + 3*mt)/4, (3*mt + et)/4
        sframe *= fps;  eframe *= fps
        intermediate_frames = range(int(sframe)+1, int(eframe))

        pre_frame_range = list(range(int((3*st+mt)/4 * fps), int((st+3*mt)/4 * fps)))
        pre_frame_range = pre_frame_range[:int(.9 * len(pre_frame_range))]

        post_frame_range = list(range(int((3*mt+et)/4 * fps), int((mt+3*et)/4 * fps)))
        post_frame_range = post_frame_range[:int(.9 * len(post_frame_range))]

        for frame in intermediate_frames:
            pre_frame = random.choice(pre_frame_range)
            post_frame = random.choice(post_frame_range)

            train_data.append('{}/image/image{:07d}.jpg {} {}/image/image{:07d}.jpg {} {}/image/image{:07d}.jpg {}\n'
                    .format(store_dir, pre_frame, pre_labels.strip(), store_dir, post_frame, post_labels.strip(), store_dir, frame, (frame-pre_frame)/(post_frame-pre_frame)))

with open('{}/unlabeled_data.txt'.format(store_dir), 'w') as f:
    for line in train_data:
        f.write(line)
