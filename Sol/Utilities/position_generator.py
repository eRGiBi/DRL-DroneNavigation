import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class PositionGenerator():

    def __init__(self, bounds, max_distance):
        self.bounds = bounds
        self.max_distance = max_distance
        
    def nearest_point_on_line(x1, y1, z1, x2, y2, z2, point):
        # Calculate direction vector of the line
        line_vector = np.array([x2 - x1, y2 - y1, z2 - z1])

        # Calculate vector from line start point to the given point
        point_vector = np.array([point[0] - x1, point[1] - y1, point[2] - z1])

        # Calculate the parameter t for the nearest point on the line
        t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)

        # Calculate coordinates of the nearest point on the line
        nearest_point = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)])

        return nearest_point


    def coordinate_differences_to_nearest_point(self, x1, y1, z1, x2, y2, z2, point):
        nearest_point = self.nearest_point_on_line(self, x1, y1, z1, x2, y2, z2, point)

        diff_x = point[0] - nearest_point[0]
        diff_y = point[1] - nearest_point[1]
        diff_z = point[2] - nearest_point[2]

        return diff_x, diff_y, diff_z

    def points_around_line(self, x1, y1, z1, x2, y2, z2):
        points = []

        # Calculate the length of the line
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Normalize the direction vector
        ux = dx / length
        uy = dy / length
        uz = dz / length

        # Generate random points around the line
        while length > 0:
            # Calculate a random distance within the specified range
            distance = random.uniform(0, self.max_distance)

            # Calculate random offsets perpendicular to the line
            offset_x = random.uniform(-1, 1)
            offset_y = random.uniform(-1, 1)
            offset_z = random.uniform(-1, 1)

            # Scale the offsets to ensure they don't exceed the maximum distance
            offset_length = math.sqrt(offset_x ** 2 + offset_y ** 2 + offset_z ** 2)
            scale_factor = min(distance / offset_length, 1.0)  # Ensure the offset is within the range
            offset_x *= scale_factor
            offset_y *= scale_factor
            offset_z *= scale_factor

            # Calculate the coordinates of the point
            x = x1 + ux * length + offset_x
            y = y1 + uy * length + offset_y
            z = z1 + uz * length + offset_z

            points.append((x, y, z))

            # Move to the next position along the line
            length -= self.max_distance

        return points


    def generate_random_point_around_line(self, x1, y1, z1, x2, y2, z2):
        # Generate a random parameter t between 0 and 1
        t = random.uniform(0, 1)

        # Interpolate coordinates using t
        x_point = x1 + t * (x2 - x1)
        y_point = y1 + t * (y2 - y1)
        z_point = z1 + t * (z2 - z1)

        # Generate random offsets for each coordinate
        offset_x = random.uniform(-self.max_distance, self.max_distance)
        offset_y = random.uniform(-self.max_distance, self.max_distance)
        offset_z = random.uniform(-self.max_distance, self.max_distance)

        # Ensure the point is within max_distance from the line
        dx = x_point - x1
        dy = y_point - y1
        dz = z_point - z1
        length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        print(length)
        scale_factor = self.max_distance / length if length > self.max_distance else 1

        # Apply the offsets to the coordinates
        x_point += offset_x * scale_factor
        y_point += offset_y * scale_factor
        z_point += offset_z * scale_factor

        return x_point, y_point, z_point


    def test(self):
        global result, diff
        x1, y1, z1 = 0, 0, 0
        x2, y2, z2 = 3, 3, 3
        max_distance = 1
        PositionGenerator = PositionGenerator([x1, y1, z1, x2, y2, z2], max_distance)
        # Generate a single random point along the line
        result = [PositionGenerator.generate_random_point_around_line(x1, y1, z1, x2, y2, z2) for i in range(10)]
        diff = [PositionGenerator.coordinate_differences_to_nearest_point(x1, y1, z1, x2, y2, z2, point) for point in
                result]
        print(result)
        print(diff)

        # # Example usage
        # x1, y1, z1 = 0, 0, 0
        # x2, y2, z2 = 3, 3, 3
        # max_distance = 1
        #
        # result = points_around_line(x1, y1, z1, x2, y2, z2, max_distance)
        #
        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotting points
        x_points = [point[0] for point in result]
        y_points = [point[1] for point in result]
        z_points = [point[2] for point in result]
        ax.scatter(x_points, y_points, z_points, color='b')
        # Plotting line
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


