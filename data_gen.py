from panda_visual_reach_env import PandaHover
import os
import numpy as np
import random
import cv2

def gen_rand_pos():
    if (random.randint(0, 1)):
        rand_x = random.uniform(0.3, 0.47)
    else:
        rand_x = random.uniform(0.53, 0.7)

    if (random.randint(0, 1)):
        rand_y = random.uniform(0.15, 0.22)
    else:
        rand_y = random.uniform(0.28, 0.7)
    return [rand_x, rand_y]

path = os.getcwd()
dir_pos_train = "/data/train/goal"
dir_neg_train = "/data/train/neg"
target_pos_train = path + dir_pos_train
target_neg_train = path + dir_neg_train
os.makedirs(target_pos_train)
os.makedirs(target_neg_train)

dir_pos_val = "/data/val/goal"
dir_neg_val = "/data/val/neg"
target_pos_val = path + dir_pos_val
target_neg_val = path + dir_neg_val
os.makedirs(target_pos_val)
os.makedirs(target_neg_val)

env = PandaHover()
train_num = 400
val_num = 100
for x in range(train_num):
    pos_number  = str(x) + "_p.png"
    pos_filename = target_pos_train + "/" + pos_number
    env.move(env.ball_pos)
    img = env.get_image()
    cv2.imwrite(pos_filename, img)
    neg_number  = str(x) + "_n.png"
    neg_filename = target_neg_train + "/" + neg_number

    rand_x = random.uniform(0.3, 0.7)
    rand_y = random.uniform(0.15, 0.7)
    rand_pos = gen_rand_pos()
    env.move(rand_pos)
    img = env.get_image()
    cv2.imwrite(neg_filename, img)

for x in range(val_num):
    pos_number  = str(x) + "_p.png"
    pos_filename = target_pos_val + "/" + pos_number
    env.move(env.ball_pos)
    img = env.get_image()
    cv2.imwrite(pos_filename, img)
    neg_number  = str(x) + "_n.png"
    neg_filename = target_neg_val + "/" + neg_number


    rand_pos = gen_rand_pos()
    env.move(rand_pos)
    img = env.get_image()
    cv2.imwrite(neg_filename, img)
