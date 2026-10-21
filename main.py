from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass
import os

@dataclass
class TriangleData:
    triangle_corner_points: np.ndarray
    picked_colors: np.ndarray

def importimage(filepath: str) -> Image.Image:
    """
    Takes a file path, opens the image with Pillow, applies any rotation stored in
    its EXIF data, and returns the corrected Pillow image.
    """
    opened_image = Image.open(filepath) 
    transposed_image = ImageOps.exif_transpose(opened_image)
    return transposed_image

def to_greyscale_array(input_image: Image.Image) -> np.ndarray:
    """
    Takes a Pillow image, converts it to greyscale, and returns it as a 2D NumPy
    array with one brightness value from 0 to 255 for each pixel.
    """
    greyscale_input_image = input_image.convert("L")
    greyscale_input_image_array = np.array(greyscale_input_image)
    return greyscale_input_image_array

def sobel_edgestrength(greyscalearray: np.ndarray) -> np.ndarray:
    """
    Takes a 2D greyscale array, applies the Sobel filter in the vertical and
    horizontal directions, and combines the two results with Pythagorean Theorem. 
    It returns a float array of the same shape that holds 
    the edge strength of each pixel.
    """
    float_greyscalearray = greyscalearray.astype(float)
    sobel_vert = ndimage.sobel(float_greyscalearray, axis = 0)
    sobel_hor = ndimage.sobel(float_greyscalearray, axis = 1)
    edgestrength = np.hypot(sobel_vert, sobel_hor)
    return edgestrength

def edge_probability(edgestrength_array: np.ndarray, 
                     weighting_power:float = 2.0, evenness:float = 0.2) -> np.ndarray:
    """
    Takes a 2D edge-strength array, a weighting power, and an evenness from 0 to 1,
    raising a ValueError if the evenness is outside that range. It raises every
    edge-strength value to the power, scales the results to add up to 1, blends in an
    equal probability for every pixel by the evenness share, and returns a 1D float
    array with one probability per pixel, read row by row, leaving the input unchanged.
    """
    if evenness > 1 or evenness < 0:
        raise ValueError("evenness can only be a float between 0 and 1")
    edgestrength_ravel_array = edgestrength_array.ravel()
    powered_array = edgestrength_ravel_array ** weighting_power
    probability_array = powered_array / powered_array.sum()
    blended_probability_array = probability_array*(1 - evenness) + evenness*(1 / len(probability_array))
    return blended_probability_array

def pick_from_probability(probability_array: np.ndarray, rng: np.random.Generator,
                          starting_count: int = 120) -> np.ndarray:
    """
    Takes a 1D probability array with one value per pixel, a NumPy random generator,
    and a starting count that defaults to 60. It uses the generator to pick that many
    different pixel positions, with pixels that have higher probabilities more likely
    to be chosen, and returns the picks as a 1D integer array of positions in the flat list.
    """
    pixel_count = len(probability_array)  
    # Picks positions in the flattened pixel list, not (x, y) points.
    # rng moves on after each call, so a second call gives different picks.
    picked_pixels = rng.choice(pixel_count, size = starting_count, 
                               replace = False, p = probability_array)
    return picked_pixels

def picked_to_xy_points(picked_pixels: np.ndarray, 
                        edgestrength_shape: tuple[int, int]) -> np.ndarray:
    """
    Takes a 1D array of picked pixel positions from the flattened image and the shape
    of the edge-strength array as rows and columns. It converts each position back to
    its row and column and returns a 2D array of points in pixel units, with one row
    per point and the x value (column) first and the y value (row) second.
    """
    rows, columns = np.unravel_index(picked_pixels, edgestrength_shape)
    # Points are (x, y), so columns (x) go first and rows (y) second.
    picked_2d = np.column_stack([columns,rows])
    return picked_2d

def points_with_corners(picked_2d: np.ndarray, edgemap_shape:tuple[int, int]) -> np.ndarray: 
    """
    Takes a 2D array of x and y points and the shape of the edge map as rows and
    columns. It adds the four image corners to the end of the points and returns the
    combined array, with N + 4 rows and two columns, in pixel units.
    """
    # The shape is (rows, columns), so index 1 is the width and index 0 is the height.
    # The last pixel is one less than the count, because counting starts at 0.
    last_column = edgemap_shape[1] - 1
    last_row = edgemap_shape[0] - 1
    corners = np.array ([[0, 0],
                         [last_column, 0],
                         [0, last_row],
                         [last_column, last_row]])
    combined_picked_array = np.concatenate((picked_2d, corners), axis = 0)
    return combined_picked_array

def triangulate_points(combined_picked_array: np.ndarray) -> np.ndarray:
    """
    Takes a 2D array of x and y points and runs a Delaunay triangulation on them with
    SciPy. It returns a 3D array with one entry per triangle, holding that triangle's
    three corners as x and y rows, in pixel units.
    """
    tri = Delaunay(combined_picked_array)
    simplices = tri.simplices
    triangle_corner_points = combined_picked_array[simplices]
    return triangle_corner_points

def triangle_draw(triangle_corner_points: np.ndarray, 
                  background_image: np.ndarray | None = None, edge_colour:str = "black",
                 line_width:float = 0.5, 
                 triangle_colours: np.ndarray | None = None) -> tuple[Figure, Axes]:
    """
    Takes an array of triangle corner points, with an optional background image, edge
    colour, line width, and an optional array of triangle colours on the 0 to 255 scale.
    It draws the triangles on a new Matplotlib figure and axes, over the background image
    if one is given. If colours are given, the triangles are filled and edged in those
    colours, and otherwise they are unfilled with edges in the edge colour.
    It returns the figure and axes without calling plt.show, and when there is no
    background it also fits the view to the triangles, flips the y axis, and sets equal
    scaling.
    """
    fig, ax = plt.subplots()
    if background_image is not None:
        ax.imshow(background_image)
    if triangle_colours is not None:
        triangle_colours_matplot = triangle_colours / 255
        collection = PolyCollection(triangle_corner_points, # type: ignore
                                     facecolors = triangle_colours_matplot,
                                       linewidths = line_width,
                                     edgecolors = triangle_colours_matplot)
    else:
        collection = PolyCollection(triangle_corner_points, # type: ignore
                                 facecolors = "none", linewidths = line_width,
                                 edgecolors = edge_colour)
    ax.add_collection(collection)
    if background_image is None:
        ax.autoscale()
        # Only needed without a background. imshow already puts row 0 at the top,
        # so flipping again with a background would turn the picture upside down.
        ax.invert_yaxis()
        ax.set_aspect("equal")
    return fig, ax

def triangle_label_array(triangle_corner_points: np.ndarray,
                         edgestrength_array_shape: tuple[int, int]) -> np.ndarray:
    """
    Takes an array of triangle corner points and the shape of the edge-strength array
    as rows and columns. For each triangle it tests every pixel inside the triangle's
    bounding box, keeps the pixels whose centre lies inside the triangle or on its edge,
    and writes the triangle's position plus 1 as its ID. It returns a 2D integer array of
    that shape, where each pixel holds the ID of the triangle covering it (0 if none).
    """
    rows, columns = edgestrength_array_shape
    labels = np.zeros((rows, columns), dtype=np.int32)
    tolerance = 1e-9
    for position, triangle in enumerate(triangle_corner_points):
        # Bounding box of the triangle, clipped to the image. Points are (x, y) = (column, row).
        left = max(int(np.floor(triangle[:, 0].min())), 0)
        right = min(int(np.ceil(triangle[:, 0].max())), columns - 1)
        top = max(int(np.floor(triangle[:, 1].min())), 0)
        bottom = min(int(np.ceil(triangle[:, 1].max())), rows - 1)
        grid_x, grid_y = np.meshgrid(np.arange(left, right + 1), np.arange(top, bottom + 1))
        a, b, c = triangle
        # Each value says which side of one edge the pixel centre is on.
        side_ab = (b[0] - a[0]) * (grid_y - a[1]) - (b[1] - a[1]) * (grid_x - a[0])
        side_bc = (c[0] - b[0]) * (grid_y - b[1]) - (c[1] - b[1]) * (grid_x - b[0])
        side_ca = (a[0] - c[0]) * (grid_y - c[1]) - (a[1] - c[1]) * (grid_x - c[0])
        # Outside means some edge says negative and another says positive.
        has_negative = (side_ab < -tolerance) | (side_bc < -tolerance) | (side_ca < -tolerance)
        has_positive = (side_ab > tolerance) | (side_bc > tolerance) | (side_ca > tolerance)
        inside = ~(has_negative & has_positive)
        labels[grid_y[inside], grid_x[inside]] = position + 1
    return labels

def triangle_rgb_by_brightness(labelled_triangle_array:np.ndarray,
                            image_colour_array: np.ndarray, 
                            grey_array: np.ndarray, 
                            triangles_corner_points: np.ndarray) -> np.ndarray:
    """
    Takes a 2D array of triangle IDs, an image colour array, a greyscale array of the same
    size, and the triangle corner points. It sorts the pixels by triangle and then by
    brightness, picks the middle pixel of each triangle, and returns a (triangles, 3) array
    with that pixel's red, green and blue values. A triangle with no pixels inside takes the
    image colour at the centre of its corners instead.
    """
    flat_brightness = grey_array.ravel()
    flat_label = labelled_triangle_array.ravel()
    sorted_pixels = np.lexsort((flat_brightness, flat_label))
    triangle_pixel_count = np.bincount(flat_label, 
                                       minlength = len(triangles_corner_points) + 1)
    empty_triangles = triangle_pixel_count[1:] == 0
    block_position = np.cumsum(triangle_pixel_count)
    block_start = block_position - triangle_pixel_count
    block_middle = block_start + (triangle_pixel_count // 2)
    block_middle[triangle_pixel_count == 0] = 0
    mid_pixel = sorted_pixels[block_middle][1:]
    flat_colour = image_colour_array.reshape(-1, 3)
    picked_colours = flat_colour[mid_pixel]
    if empty_triangles.any():
        centres = np.round(triangles_corner_points[empty_triangles].mean(axis=1)).astype(int)
        columns = np.clip(centres[:, 0], 0, image_colour_array.shape[1] - 1)
        rows = np.clip(centres[:, 1], 0, image_colour_array.shape[0] - 1)
        picked_colours[empty_triangles] = image_colour_array[rows, columns]
    return picked_colours

def triangles_to_image(triangle_corner_points: np.ndarray,
                       triangle_colours: np.ndarray,
                       image_shape: tuple[int, int],
                       output_path: str | None = None) -> Image.Image:
    """
    Takes an array of triangle corner points, an array of triangle colours on the 0 to 255
    scale, the image shape as rows and columns, and an optional file path. It builds the
    triangle label array, looks up each pixel's triangle colour (black where no triangle
    covers a pixel), saves the image to the path if one is given, and returns the Pillow image.
    """
    labels = triangle_label_array(triangle_corner_points, image_shape)
    lookup = np.vstack([[0, 0, 0], triangle_colours]).astype(np.uint8)
    canvas_image = Image.fromarray(lookup[labels], "RGB")
    if output_path is not None:
        canvas_image.save(output_path)
    return canvas_image

def triangle_split(triangle_corner_points: np.ndarray,
                        rng: np.random.Generator,
                        spread: float) -> np.ndarray:
    """
    Takes an array of triangle corner points, a random generator and a spread for the
    Gaussian. It cuts the longest side of every triangle at a point found by drawing a
    fraction from a Gaussian centred on 0.5, redrawing any fraction outside 0.1 to 0.9,
    joins that point to the opposite corner, and returns the new triangles' corners as a
    (2 x triangles, 3, 2) array, with every first half listed before every second half.
    """
    rolled_corners = np.roll(triangle_corner_points, shift=-1,axis=1)
    third_corner = np.roll(triangle_corner_points, shift=-2,axis=1)
    diff_array = triangle_corner_points - rolled_corners
    squared_diff_array = diff_array**2
    squared_diff_array = squared_diff_array.sum(axis = 2)
    longest_side_position = np.argmax(squared_diff_array, axis=1)
    triangle_numbers = np.arange(len(triangle_corner_points))
    side_start = triangle_corner_points[triangle_numbers, longest_side_position]
    side_end = rolled_corners[triangle_numbers, longest_side_position]
    side_opposite = third_corner[triangle_numbers, longest_side_position]
    vector = side_end - side_start
    fraction_array = rng.normal(0.5, spread, size = len(triangle_corner_points))
    mask = (fraction_array > 0.9) | (fraction_array < 0.1)
    while mask.any():
        badcount = mask.sum()
        replacement_array = rng.normal(0.5, spread, size = badcount)
        fraction_array[mask] = replacement_array
        mask = (fraction_array > 0.9) | (fraction_array < 0.1)
    fraction_array = fraction_array.reshape(-1, 1)
    cut_point_array = side_start + (fraction_array * vector)
    triangle1 = np.stack([side_start, cut_point_array, side_opposite], axis=1)
    triangle2 = np.stack([side_end, cut_point_array, side_opposite], axis=1)
    split_triangles = np.concatenate([triangle1, triangle2], axis=0)
    return split_triangles

def coin_split(triangle_array: np.ndarray,
                split_chance: float,
                rng: np.random.Generator) -> np.ndarray:
    """Takes an array of triangles, a split chance between 0 and 1 and a
    random generator, and draws one whole number from 1 to 100 for each
    triangle. Returns a boolean array with one value per triangle, which is
    True where the draw is at or below the chance, and raises a ValueError
    if the chance is outside 0 to 1.
    """
    if split_chance < 0 or split_chance > 1:
        raise ValueError("split_chance must be a float between 0 and 1")
    pass_array = rng.integers(1, 101, size = len(triangle_array))
    mask = (pass_array <= split_chance*100)
    return mask

def recursion_driver(triangle_corner_points: np.ndarray,
                        rng: np.random.Generator,
                        spread: float = 0.2,
                        split_chance: float = 0.7,) -> np.ndarray:
    """Takes an array of triangle corner points, a random generator, a spread and a
    starting split chance, and runs up to 13 rounds in which each triangle still in
    play flips a coin to be split by triangle_split, with the split chance halved
    after every round. Returns a (triangles, 3, 2) array holding the triangles that
    stopped splitting together with those still active when the rounds end.
    """
    i = 0
    active_pile = triangle_corner_points
    inactive_pile = np.empty_like(triangle_corner_points)[:0]
    while i <= 20 and len(active_pile) > 0:
        triangle_check_array = coin_split(active_pile, split_chance, rng)
        inactive_pile = np.concatenate([inactive_pile, 
                                               active_pile[~triangle_check_array]],axis=0)
        active_pile = triangle_split(active_pile[triangle_check_array], rng, spread)
        i = i + 1
        split_chance = split_chance * (1/2) 
    recursed_array = np.concatenate([inactive_pile, 
                                       active_pile],axis=0)
    return recursed_array

def borrow_colour(triangle_colours, original_image_array, rng, borrow_chance=0.3):
    """
    Takes a (triangles, 3) array of triangle colours, the original photo as an array, a
    random generator and a borrow chance from 0 to 1 (default 0.3), and raises a ValueError
    outside that range. Using its own random stream spawned from the generator, each
    triangle that wins a flip takes the colour of one pixel picked uniformly from the whole
    photo, and the function returns a new (triangles, 3) array without changing the input.
    """
    if borrow_chance < 0 or borrow_chance > 1:
        raise ValueError("borrow_chance must be a float between 0 and 1")
    borrow_rng = rng.spawn(1)[0]
    triangle_count = len(triangle_colours)
    borrows = borrow_rng.random(triangle_count) < borrow_chance
    rows = borrow_rng.integers(0, original_image_array.shape[0], size = triangle_count)
    columns = borrow_rng.integers(0, original_image_array.shape[1], size = triangle_count)
    new_colours = triangle_colours.copy()
    new_colours[borrows] = original_image_array[rows[borrows], columns[borrows]]
    return new_colours

def bend_offsets(image_shape: tuple[int, int], curve_strength: float, curve_smoothness: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Takes the image shape as rows and columns, a bend strength and a smoothness (both as shares
    of the image's longer side) and a random generator. It makes two smooth noise fields with a
    Gaussian blur, scales them to the strength, fades them to 0 near the picture's edges, and
    shrinks them if needed so the warp never folds (the smallest Jacobian determinant stays at
    0.05 or more). Returns the column offsets, the row offsets and the scale that was used (1.0
    when nothing had to shrink).
    """
    rows, columns = image_shape
    longer = max(rows, columns)
    blur = max(curve_smoothness * longer, 1.0)
    fade_width = max(min(2 * blur, 0.10 * min(rows, columns)), 1.0)

    def ramp(length):
        # 0 at both ends, easing up to 1 over fade_width pixels (smoothstep, so no sharp corner)
        position = np.minimum(np.arange(length), np.arange(length)[::-1]) / fade_width
        position = np.clip(position, 0, 1)
        return position * position * (3 - 2 * position)

    fade = (ramp(rows)[:, None] * ramp(columns)[None, :]).astype(np.float32)
    # The noise is blurred at a smaller size and then stretched back up. The bends are smooth, so
    # this looks the same but is much faster than blurring the full-size noise.
    shrink = max(1, int(blur // 8))
    small_shape = (-(-rows // shrink), -(-columns // shrink))
    at_rows = (np.arange(rows, dtype=np.float32) + 0.5) / shrink - 0.5
    at_columns = (np.arange(columns, dtype=np.float32) + 0.5) / shrink - 0.5
    grid_r, grid_c = np.meshgrid(at_rows, at_columns, indexing="ij")
    offsets = []
    for _ in range(2):
        small = ndimage.gaussian_filter(rng.standard_normal(small_shape, dtype=np.float32), blur / shrink)
        noise = ndimage.map_coordinates(small, [grid_r, grid_c], order=3, mode="nearest")   # cubic, so the bends have no kinks
        spread = float(noise.std())
        noise = noise / spread if spread > 0 else noise
        offsets.append(noise * (curve_strength * longer) * fade)
    column_offsets, row_offsets = offsets
    # The warp folds where the Jacobian determinant of (x + dx, y + dy) drops to 0 or below.
    dx_drow, dx_dcol = np.gradient(column_offsets)
    dy_drow, dy_dcol = np.gradient(row_offsets)

    def smallest_determinant(scale):
        return float(((1 + scale * dx_dcol) * (1 + scale * dy_drow)
                      - (scale * dx_drow) * (scale * dy_dcol)).min())

    scale = 1.0
    if smallest_determinant(1.0) < 0.05:
        low, high = 0.0, 1.0
        for _ in range(30):
            middle = (low + high) / 2
            if smallest_determinant(middle) >= 0.05:
                low = middle
            else:
                high = middle
        scale = low
    return column_offsets * scale, row_offsets * scale, scale

def bent_triangles_to_image(triangle_corner_points: np.ndarray,
                            triangle_colours: np.ndarray,
                            image_shape: tuple[int, int],
                            column_offsets: np.ndarray,
                            row_offsets: np.ndarray,
                            output_path: str | None = None) -> Image.Image:
    """
    Takes the triangle corner points, their colours, the image shape as rows and columns and the
    two offset fields from bend_offsets. It builds the triangle label array at twice the size,
    looks up each sub-pixel's triangle at its bent position, colours it, and shrinks every 2 x 2
    block back to one pixel so the curved edges come out smooth. Saves the picture if a path is
    given and returns it as a Pillow image the same size as the photo.
    """
    rows, columns = image_shape
    # Pixel j of the 2x grid sits at 1x position (j - 0.5) / 2, so the triangles scale by 2, plus 0.5.
    fine_labels = triangle_label_array(triangle_corner_points * 2 + 0.5, (rows * 2, columns * 2))
    lookup = np.vstack([[0, 0, 0], triangle_colours]).astype(np.uint8)
    picture = np.empty((rows, columns, 3), dtype=np.uint8)
    fine_columns = np.arange(columns * 2, dtype=np.float32)
    strip = 256   # rows of the final picture handled at a time, to keep memory low
    for top in range(0, rows, strip):
        bottom = min(top + strip, rows)
        fine_rows = np.arange(top * 2, bottom * 2, dtype=np.float32)
        grid_r, grid_c = np.meshgrid(fine_rows, fine_columns, indexing="ij")
        at_r, at_c = (grid_r - 0.5) / 2, (grid_c - 0.5) / 2
        shift_c = ndimage.map_coordinates(column_offsets, [at_r, at_c], order=1, mode="nearest")
        shift_r = ndimage.map_coordinates(row_offsets, [at_r, at_c], order=1, mode="nearest")
        # The outer ring of 2x sub-pixels lies half a pixel outside the triangles, so lookups stay
        # one sub-pixel inside the edge.
        look_r = np.clip(grid_r + 2 * shift_r, 1, rows * 2 - 2)
        look_c = np.clip(grid_c + 2 * shift_c, 1, columns * 2 - 2)
        ids = np.asarray(ndimage.map_coordinates(fine_labels, [look_r, look_c], order=0,
                                                 mode="nearest")).astype(np.int64)
        fine_strip = Image.fromarray(lookup[ids], "RGB")
        picture[top:bottom] = np.asarray(fine_strip.resize((columns, bottom - top), Image.Resampling.BOX))
    bent_image = Image.fromarray(picture, "RGB")
    if output_path is not None:
        bent_image.save(output_path)
    return bent_image

def make_triangle_art(image_path: str,
                      seed: int,
                      starting_count: int = 120,
                      weighting_power: float = 2.0,
                      evenness: float = 0.2,
                      spread: float = 0.2,
                      split_chance: float = 0.7,
                      borrow_chance: float = 0.3,
                      curve_strength: float = 0.0,
                      curve_smoothness: float = 0.04,
                      output_path: str | None = None) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    Takes an image path, a seed and the settings for the pipeline, and checks every setting
    (and the image size, read from the file header only) before any step runs, raising a
    ValueError for a bad value. It then runs importimage through borrow_colour and
    triangles_to_image, giving the point picking, the recursion and the borrowing each their
    own random stream spawned from the seed. It saves the final picture if an output path is
    given, and returns the triangle corner points, the final triangle colours and the
    final Pillow image. With curve_strength above 0 (a share of the longer side, default 0 for
    straight edges), every edge is bent by a smooth warp from its own random stream, with
    curve_smoothness setting how wide the bends are; bends are reduced automatically if they
    would fold a triangle over.
    """
    whole_types = (int, np.integer)
    real_types = (int, float, np.integer, np.floating)
 
    # Checks. Each comparison is written so that NaN fails it.
    if isinstance(seed, bool) or not isinstance(seed, whole_types) or seed < 0:
        raise ValueError("seed must be a whole number of 0 or more")
    if isinstance(starting_count, bool) or not isinstance(starting_count, whole_types) or starting_count < 1:
        raise ValueError("starting_count must be a whole number of 1 or more")
    if (isinstance(weighting_power, bool) or not isinstance(weighting_power, real_types)
            or not (np.isfinite(weighting_power) and weighting_power >= 0)):
        raise ValueError("weighting_power must be a finite number of 0 or more")
    if (isinstance(spread, bool) or not isinstance(spread, real_types)
            or not (np.isfinite(spread) and spread >= 0)):
        raise ValueError("spread must be a finite number of 0 or more")
    if (isinstance(curve_strength, bool) or not isinstance(curve_strength, real_types)
            or not (np.isfinite(curve_strength) and curve_strength >= 0)):
        raise ValueError("curve_strength must be a finite number of 0 or more")
    if (isinstance(curve_smoothness, bool) or not isinstance(curve_smoothness, real_types)
            or not (np.isfinite(curve_smoothness) and 0 < curve_smoothness <= 1)):
        raise ValueError("curve_smoothness must be a number above 0 and at most 1")
    for setting_name, setting_value in (("evenness", evenness),
                                        ("split_chance", split_chance),
                                        ("borrow_chance", borrow_chance)):
        if (isinstance(setting_value, bool) or not isinstance(setting_value, real_types)
                or not (0 <= setting_value <= 1)):
            raise ValueError(f"{setting_name} must be a number between 0 and 1")
    if output_path is not None:
        output_path = os.fspath(output_path)
        if os.path.splitext(output_path)[1].lower() not in Image.registered_extensions():
            raise ValueError("output_path must end in an image file extension Pillow can save")
        if not os.path.isdir(os.path.dirname(output_path) or "."):
            raise ValueError("the folder of output_path does not exist")
    # Only the file header is read here, so no pixels are loaded yet.
    with Image.open(image_path) as header:
        width, height = header.size
    if width < 2 or height < 2:
        raise ValueError("the image must be at least 2 x 2 pixels")
    if starting_count > width * height:
        raise ValueError("starting_count cannot be larger than the number of pixels")
 
    # One child stream per step, so changing a setting in one step cannot move the draws of another.
    # spawn(4) gives the same first three streams as spawn(3), so straight pictures do not change.
    pick_seed, recursion_seed, borrow_seed, bend_seed = np.random.SeedSequence(seed).spawn(4)
    pick_rng = np.random.default_rng(pick_seed)
    recursion_rng = np.random.default_rng(recursion_seed)
    borrow_rng = np.random.default_rng(borrow_seed)
 
    image = importimage(image_path).convert("RGB")
    image_array = np.array(image)
    grey_array = to_greyscale_array(image)
    edgestrength = sobel_edgestrength(grey_array)
    probability = edge_probability(edgestrength, weighting_power, evenness)
    picked = pick_from_probability(probability, pick_rng, starting_count)
    xy_points = picked_to_xy_points(picked, edgestrength.shape)
    with_corners = points_with_corners(xy_points, edgestrength.shape)
    triangle_points = triangulate_points(with_corners)
    recursed = recursion_driver(triangle_points, recursion_rng, spread, split_chance)
    labelled = triangle_label_array(recursed, edgestrength.shape)
    median_colours = triangle_rgb_by_brightness(labelled, image_array, grey_array, recursed)
    final_colours = borrow_colour(median_colours, image_array, borrow_rng, borrow_chance)
    if curve_strength == 0:
        # Straight edges: no bending work at all, so this costs no extra time.
        final_image = triangles_to_image(recursed, final_colours, edgestrength.shape, output_path)
    else:
        column_offsets, row_offsets, _ = bend_offsets(edgestrength.shape, curve_strength,
                                                      curve_smoothness, np.random.default_rng(bend_seed))
        final_image = bent_triangles_to_image(recursed, final_colours, edgestrength.shape,
                                              column_offsets, row_offsets, output_path)
    return recursed, final_colours, final_image


# not relevant anymore code

# def get_triangle_rgb(labelled_triangle_array: np.ndarray, 
#                      image_colour_array :np.ndarray,
#                      triangle_corner_points: np.ndarray) -> np.ndarray:
#     """
#     Takes a 2D array of triangle IDs, an image colour array with red, green and blue values
#     for each pixel, and the triangle corner points, which are only used to count the
#     triangles. It calls ndimage.median on each colour channel to find the median value of the
#     pixels inside every triangle, and returns a float array of shape (triangles, 3) with the
#     red, green and blue medians in triangle order.
#     """
#     # The stop value is excluded, so + 1 makes the IDs run from 1 up to the triangle count.
#     ID_list = np.arange(1, len(triangle_corner_points) + 1)
#     red_split = image_colour_array[:, :, 0]
#     green_split = image_colour_array[:, :, 1]
#     blue_split = image_colour_array[:, :, 2]
#     # A triangle with no pixels gets a meaningless value here, not an error.
#     red_median = ndimage.median(red_split, labels = labelled_triangle_array,
#                                  index = ID_list)
#     green_median = ndimage.median(green_split, labels = labelled_triangle_array,
#                                      index = ID_list)
#     blue_median = ndimage.median(blue_split, labels = labelled_triangle_array,
#                                      index = ID_list)
#     stacked_medians = np.column_stack([red_median, green_median, blue_median])
#     return stacked_medians


#parameters and seeds
seed = 42
rng = np.random.default_rng(seed)

if __name__ == "__main__":
    # bottlebrush, with the matplotlib views
    bottlebrush_triangles, bottlebrush_colours, bottlebrush_final_image = make_triangle_art(
        "bottlebrush.jpg", seed, output_path = "bottlebrush_triangles.png")
    bottlebrush_img_array = np.array(importimage("bottlebrush.jpg"))

    bottlebrush_triangle_draw_noBG, _ = triangle_draw(bottlebrush_triangles)
    bottlebrush_triangle_draw_BG, _ = triangle_draw(bottlebrush_triangles, bottlebrush_img_array)
    bottlebrush_triangle_colour_draw_noBG, _ = triangle_draw(bottlebrush_triangles,
                                                            triangle_colours = bottlebrush_colours)
    print("bottlebrush:", len(bottlebrush_triangles), "triangles")
    bottlebrush_final_image.show()

    # every other photo, one picture each
    other_photos = ["ajumma", "bird", "chairs", "goat", "nets",
                    "poles", "retrospace1", "rocks", "sekiro"]
    show_images = False   # True opens every final picture as well

    for name in other_photos:
        photo_triangles, photo_colours, photo_final_image = make_triangle_art(
            name + ".jpg", seed, output_path = name + "_triangles.png")
        print(f"{name}: {photo_final_image.size[0]}x{photo_final_image.size[1]}, "
            f"{len(photo_triangles)} triangles")
        if show_images:
            photo_final_image.show()

    plt.show()






