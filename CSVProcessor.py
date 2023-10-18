from math import ceil
from math import floor
from time import sleep
import numpy as np

from graphics import *
import csv


# Array of pixels
def print_picture(image):
    win = GraphWin("hi", 600, 600)
    for pixel in range(784):
        pixel_y = floor(pixel / 28)
        pixel_x = pixel - pixel_y * 28
        box_size = 16
        rect = Rectangle(Point(pixel_x * box_size, pixel_y * box_size),
                         Point(pixel_x * box_size + box_size, pixel_y * box_size + box_size))
        grayscale = floor(image[pixel] * 255)
        rect.setFill(color_rgb(grayscale, grayscale, grayscale))
        rect.draw(win)


def find_coordinates(num):
    y = floor(num / 28)
    x = num - 28 * y
    return x, y


def convert_training(file, amount='all'):
    csv_file = open(file, newline='')
    csv_reader = csv.reader(csv_file, delimiter=',')

    images_training_array = []
    row_count = 0
    image_number = 0

    for row in csv_reader:
        if row_count == 0:
            pass
        elif amount == 'all' or row_count <= amount:
            images_training_array.append(([], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            image_number = image_number + 1
            column_count = 0
            for column in range(785):
                if column_count == 0:
                    images_training_array[image_number - 1][1][int(row[0])] = 1
                    column_count = 1
                else:
                    images_training_array[image_number - 1][0].append(int(row[column]) / 255)
        row_count = row_count + 1
    return images_training_array


def convert_testing(file, amount='all'):
    csv_file = open(file, newline='')
    csv_reader = csv.reader(csv_file, delimiter=',')

    images_testing_array = []
    row_count = 0
    image_number = 0
    for row in csv_reader:
        if row_count == 0:
            pass
        elif amount == 'all' or row_count <= amount:
            images_testing_array.append([])
            for column in range(784):
                images_testing_array[image_number].append(int(row[column]) / 255)
            image_number = image_number + 1
        row_count = row_count + 1
    return images_testing_array
