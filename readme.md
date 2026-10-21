# Triangle art

## What the project is

This project turns a photo into a picture made only of triangles. The idea came from simple rules that give complex results, like the Game of Life and the Mandelbrot set, and the splitting rule follows Tyler Hobbs's "Aesthetically Pleasing Triangle Subdivision". Big triangles are placed where the photo has the most detail, then they are split again and again with a random rule, so the result looks like the photo but also a bit foreign. The code is in Python with NumPy, SciPy, Pillow and matplotlib, and a Streamlit page lets other people use it in a browser.

## How the pipeline works

Everything lives in `main.py`. The steps run in this order:

1. `importimage` opens the photo with Pillow and applies the EXIF rotation with `ImageOps.exif_transpose`, because phone photos are often stored sideways.
2. `to_greyscale_array` turns the photo into a 2D array of brightness values from 0 to 255.
3. `sobel_edgestrength` runs the Sobel filter down and across the greyscale array and combines the two results with `np.hypot`, which gives the edge strength of every pixel.
4. `edge_probability` raises each edge strength to `weighting_power`, scales the values so they add up to 1, and blends in an equal share for every pixel by the `evenness` amount. It raises a ValueError if evenness is outside 0 to 1.
5. `pick_from_probability` uses `rng.choice` without replacement to pick `starting_count` different pixels, so pixels on strong edges are more likely to be picked.
6. `picked_to_xy_points` turns the flat positions back into (x, y) points with `np.unravel_index`, with the column first and the row second.
7. `points_with_corners` adds the four corners of the image, so the triangles cover the whole picture.
8. `triangulate_points` joins the points with a Delaunay triangulation from SciPy and returns a (triangles, 3, 2) array of corner coordinates.
9. `recursion_driver` runs the splitting rounds with two piles. The active pile holds triangles that may still split, and the inactive pile holds triangles that lost a coin flip. Each round, `coin_split` draws a whole number from 1 to 100 for every active triangle, and a triangle splits when its number is at or below `split_chance` times 100. The winners go to `triangle_split`, the losers move to the inactive pile, and the split chance halves for the next round. The loop stops when no triangles are active or after the round limit.
10. `triangle_split` handles every triangle at once. It uses `np.roll` and `np.argmax` to find each triangle's longest side, draws a cut fraction from a Gaussian centred on 0.5 with `spread` as its standard deviation, redraws any fraction outside 0.1 to 0.9, and cuts the side there. Each triangle becomes two triangles that share the cut point and the opposite corner.
11. `triangle_label_array` works out which triangle covers each pixel. For every triangle, it tests whether the centre of each pixel in the triangle's bounding box is inside the triangle or on its edge, and writes the triangle's number into a label array.
12. `triangle_rgb_by_brightness` sorts all pixels by triangle and then by brightness with one `np.lexsort`, and gives each triangle the colour of its middle pixel. A triangle with no pixels takes the colour at the centre of its three corners.
13. `borrow_colour` gives each triangle a `borrow_chance` of taking the colour of one random pixel from anywhere in the photo.
14. `triangles_to_image` looks up each pixel's colour from the label array and returns a Pillow image at the exact size of the photo, saving it if a path is given.

`triangle_draw` is a separate matplotlib view. It can show the triangle outlines, the outlines over the photo, or the filled triangles, and it returns the figure without calling `plt.show`.

## The master function

`make_triangle_art` runs the whole pipeline from one call. It takes an image path, a seed, the six settings and an optional output path, and returns the triangles, the final colours and the finished picture. Before any work starts, it checks every setting and the image size, which it reads from the file header only, and raises a ValueError with a plain message for any bad value. The checks are written so that NaN fails them too.

The settings and their defaults are:

- `starting_count` = 120, the number of points picked before the 4 corners.
- `weighting_power` = 2.0, how strongly the points follow the edges.
- `evenness` = 0.2, the share of probability spread evenly over all pixels.
- `spread` = 0.2, how far from the middle a side can be cut.
- `split_chance` = 0.7, the starting chance that a triangle splits.
- `borrow_chance` = 0.3, the chance that a triangle borrows a random photo colour.

The seed is turned into three separate random streams with `np.random.SeedSequence(seed).spawn(3)`: one for picking points, one for the recursion and one for borrowing. Because each step has its own stream, changing one setting cannot change the random draws of another step. The same photo, seed and settings always give the same picture.

The testing area at the bottom of `main.py` sits behind `if __name__ == "__main__":`. It makes a picture of bottlebrush with the matplotlib views, then one picture for each of my other photos (ajumma, bird, chairs, goat, nets, poles, retrospace1, rocks and sekiro), saved as `<name>_triangles.png`.

## The web app

The web app has two files. `picture_service.py` does the work and has no Streamlit code in it, so it can be tested on its own. It reads the seed box (blank means a random seed), writes the upload to a temporary folder, shrinks photos to 3000 px on the longest side after the EXIF rotation, and calls `make_triangle_art`. At most two pictures are made at the same time, and anyone else waits for a free slot. Every problem becomes a plain message, and the temporary folder is deleted when the run ends, so no files are kept.

`streamlit_app.py` is the page itself. It has a photo upload, a seed box and a slider for each setting inside a form, so moving a slider does nothing until Make picture is pressed. The right side shows the picture, the seed used and a Save picture button that downloads `triangles_seed<seed>.png`. Each visitor's session keeps only their latest picture.

The code is on GitHub as RaphaelJakob/Triangle-Visualisation, on the `main` branch, with `requirements.txt` listing streamlit, pillow-heif, numpy (1.25 or newer), scipy, pillow and matplotlib. My photos and their output pictures are kept out of the repo with `.gitignore`, since the repo is public. A separate Gradio version also exists for running on my own computer only, with no public link and no shrinking.

## Decisions I made

- I chose approach A: triangulate the photo first, then apply the splitting rule to those triangles.
- The splitting rule is Hobbs-style only, cutting the longest side at a Gaussian-placed point. The Sierpinski version with holes was dropped.
- Image detail only places the big starting triangles, and the recursion depth comes from the weighted coin flip.
- Points stay in pixel units, with x as the column and y as the row.
- Colours come from the middle pixel by brightness, because the per-channel median gave muddy colours that no pixel in the triangle actually had.
- Colour borrowing uses one flat chance for every triangle. Borrowing more for small triangles looked worse.
- Empty triangles take the colour at their centre instead of raising an error.
- The output picture is drawn with Pillow at the photo's exact size, and matplotlib is kept for viewing.
- For the web app, I picked Streamlit Community Cloud because friends can use it while my computer is off, and a public repo is fine with me. Hugging Face Spaces now needs a paid plan for a Gradio app, and a Gradio share link only works while my computer is on and expires after 72 hours.

## Problems I solved

- Black hairlines between triangles were pixels that no triangle painted. Pillow drops the decimal part of corner points, and the recursion cut points are almost never whole numbers, so a corner sitting in the middle of a neighbour's side got pulled off that side and left a thin unpainted wedge. The exact pixel-centre test in `triangle_label_array` fixed this, and the gaps went from hundreds of pixels to 0 on the test photos.
- Phone photos came out sideways until `ImageOps.exif_transpose` was added, because Pillow and `np.array` ignore the EXIF rotation.
- The muddy colours were fixed by switching from per-channel medians to the brightness median.
- Random draws in one step used to depend on the settings of other steps. Giving each step its own stream fixed this.

## What I learned

- NumPy arrays index as [row, column], while points and Pillow sizes use (x, y), which is why the code swaps `shape[1]` and `shape[0]` in places.
- `.shape` is an attribute, so it has no brackets, and an image array's three shape numbers are rows, columns and colour values.
- In a slice, a colon keeps an axis and a whole number drops it, so `[:, :, 0]` gives a 2D array and `[:0]` keeps zero triangles but keeps the inner shape.
- `np.concatenate`, `np.stack`, `np.column_stack` and `np.lexsort` all take their arrays as one list or tuple. `np.stack` adds a new axis, while `np.concatenate` joins along an existing one.
- `np.roll` along axis 1 pairs each corner with the next one and wraps the last back to the first, and `np.argmax` then finds the longest side of every triangle at once.
- A comparison on an array gives a True/False mask. Masks are combined with `|` and `&`, flipped with `~`, checked with `.any()` or `.all()`, and used to overwrite only the True entries.
- `rng.integers(low, high)` leaves out the high end, so 1 to 100 needs 101, and the random generator moves on after every call, so the same seed only repeats from a fresh generator.
- Delaunay's `.simplices` holds the positions of the input points, and `points[simplices]` swaps them for their coordinates.
- Thinking in whole arrays, instead of looping over one triangle at a time, is fast and turned out to be one of my favourite parts of the project.

## What I will do next

1. Check the app runs on Streamlit Community Cloud, then ask two or three friends to try it on their phones.
2. Run the tests on my own photos, including a rotated phone photo and a HEIC photo, since the checks so far only used stand-in photos.
3. Fix the docstrings that no longer match the code. `recursion_driver` says 13 rounds but runs up to 21, and `pick_from_probability` still says the default is 60 instead of 120.
4. Add curved edges with Gaussian noise (default strength 0), then the animation and the credits in the README.
5. Look around for more inspiration, and later add black lines between triangles as an optional setting.