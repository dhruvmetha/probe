import random
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from math import cos, sin
from functools import partial
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import math
import json

def plot_rectangles(rectangles, A, B):
    """
    Plot rectangles within a 2D space of dimensions AxB.

    Parameters:
    - rectangles: An array of shape (n, 5) with each row representing a rectangle
      in the format [center_x, center_y, theta, width, height].
    - A, B: Dimensions of the 2D space.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, A)
    ax.set_ylim(-B/2, B/2)
    ax.set_aspect('equal')

    for i, rect in enumerate(rectangles):
        x, y, theta, width, length, movable = rect  # Adjust based on your rectangle representation
        polygon = create_polygon(x, y, theta, width, length)
        patch = patches.Polygon(list(polygon.exterior.coords), closed=True, edgecolor='r', fill=False)
        ax.add_patch(patch)
        rx, ry = polygon.centroid.x, polygon.centroid.y
        ax.text(rx, ry, str(movable), color='blue', fontsize=12)

    plt.show()

def create_polygon(x, y, theta, width, length):
    # Create rectangle centered at (0, 0) then rotate and translate
    rect = Polygon([(-width/2, -length/2), (-width/2, length/2), (width/2, length/2), (width/2, -length/2)])
    rect = rotate(rect, theta, use_radians=True)  # If theta is in radians; if in degrees, omit use_radians=True
    rect = translate(rect, x, y)
    return rect

def fits_within_space(rect, x_max, y_max):
    minx, miny, maxx, maxy = rect.bounds
    return minx >= 0 and miny >= 0 and maxx <= x_max and maxy <= y_max

def try_place_rectangle(rectangles, x, y, theta, width, length, x_max, y_max):
    new_rect = create_polygon(x, y, theta, width, length)
    if not fits_within_space(new_rect, x_max, y_max):
        return False
    for existing_rect in rectangles:
        x_e, y_e, theta_e, width_e, length_e = existing_rect
        existing_rect_poly = create_polygon(x_e,y_e,theta_e,width_e,length_e)
        if new_rect.intersects(existing_rect_poly) and new_rect.intersection(existing_rect_poly).area > 0:
            return False  # Overlaps, do not add
    return True

def generate_non_overlapping_rectangles(num_rectangles, length_range, width_range, space_length, space_width):
    rectangles = []

    for idx in range(num_rectangles):
        successful_placement = False
        while not successful_placement:
            length = round(random.uniform(*length_range), 2)
            width = round(random.uniform(*width_range), 2)
            x = round(random.uniform(width / 2, space_width - width / 2), 2)
            y = round(random.uniform(length / 2, space_length - length / 2), 2)
            theta = round(random.uniform(-np.pi/6, np.pi/6), 2)  # Orientation in degrees

            if try_place_rectangle(rectangles, x, y, theta, width, length, space_width, space_length):
                rectangles.append([
                    x, y, theta, width, length
                ])
                successful_placement = True
                # if successful_placement:
                #     print(idx, theta * 180 / np.pi)
    return rectangles

def write_rectangles_to_file(new_rectangles):
    with open('rectangles.txt', 'w') as file:
        for rect in new_rectangles:
            file.write(f'{rect[0]}, {rect[1]}, {rect[2]}, {rect[3]}, {rect[4]}\n')

if __name__=="__main__":
    from tqdm import tqdm
    space_length = 1.8  # meters
    space_width = 2.2   # meters
    length_range = (0.4, 1.7)  # meters
    width_range = (0.4, 0.4)  # meters
    
    world_list = []
    for i in tqdm(range(3000)):
    # while True:
        num_rectangles = np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
        rectangles_list = generate_non_overlapping_rectangles(num_rectangles,length_range,width_range,space_length,space_width)
        new_rectangles = []
        pos = []
        size = []
        name = []
        nature = []
        ctr = 0
        for rect in rectangles_list:
            x, y, theta, width, length = rect
            new_x = x + 0.5  # New x is the old y
            new_y = round(y - space_length/2, 2) # New y is the negative of the old x
            new_theta = theta # ((theta * 180/np.pi) + 90)  # Normalize the new orientation

            movable = random.choice([0, 1])
            if length > 1.4:
                movable = 1
            nature.append(movable)

            pos.append([new_x, new_y, new_theta])
            size.append([width, length, 0.4])
            name.append(f'box_{ctr}')
            ctr += 1
        data = {
            'pos': pos,
            'size': size,
            'name': name,
            'nature': nature
        }
        world_list.append(data)
    
    # print(world_list)
    
    with open('worlds.json', 'w') as file:
        json.dump(world_list, file)
            # The rectangle itself does not change shape; width and length remain the same
        #     new_rectangles.append([new_x, new_y, new_theta, width, length, movable])
        # write_rectangles_to_file(new_rectangles)
        # plot_rectangles(new_rectangles, space_width, space_length)