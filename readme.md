# Triangle art

## What the project is

This project turns a photo into a picture made only of triangles. The idea came from simple rules that give complex results, like the Game of Life and the Mandelbrot set, and the splitting rule follows Tyler Hobbs's "Aesthetically Pleasing Triangle Subdivision". Big triangles are placed where the photo has the most detail, then they are split again and again with a random rule, so the result looks like the photo but also a bit foreign. The code is in Python with NumPy, SciPy, Pillow and matplotlib, and a Streamlit page lets other people use it in a browser.

You can try it online at https://triangle-visualisation.streamlit.app/.

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
14. With straight edges, `triangles_to_image` looks up each pixel's colour from the label array and returns a Pillow image at the exact size of the photo, saving it if a path is given.
15. With `curve_strength` above 0, `bend_offsets` and `bent_triangles_to_image` draw the picture with bent edges instead, as described in the bent edges section.

`triangle_draw` is a separate matplotlib view. It can show the triangle outlines, the outlines over the photo, or the filled triangles, and it returns the figure without calling `plt.show`.

## The master function

`make_triangle_art` runs the whole pipeline from one call. It takes an image path, a seed, the eight settings and an optional output path, and returns the triangles, the final colours and the finished picture. Before any work starts, it checks every setting and the image size, which it reads from the file header only, and raises a ValueError with a plain message for any bad value. The checks are written so that NaN fails them too.

The settings and their defaults are:

- `starting_count` = 960, the number of points picked before the 4 corners.
- `weighting_power` = 2.0, how strongly the points follow the edges.
- `evenness` = 0.2, the share of probability spread evenly over all pixels.
- `spread` = 0.2, how far from the middle a side can be cut.
- `split_chance` = 0.7, the starting chance that a triangle splits.
- `borrow_chance` = 0.3, the chance that a triangle borrows a random photo colour.
- `curve_strength` = 0.0, how far the edges bend, as a share of the image's longer side. At 0 the edges stay straight.
- `curve_smoothness` = 0.04, how wide each bend is, also as a share of the longer side.

The seed is turned into four separate random streams with `np.random.SeedSequence(seed).spawn(4)`: one for picking points, one for the recursion, one for borrowing and one for bending. Because each step has its own stream, changing one setting cannot change the random draws of another step. The same photo, seed and settings always give the same picture. Adding the bending stream did not change the first three streams, so straight pictures are the same as before bending existed.

The testing area at the bottom of `main.py` sits behind `if __name__ == "__main__":`. It makes a picture of bottlebrush with the matplotlib views, then one picture for each of my other photos (ajumma, bird, chairs, goat, nets, poles, retrospace1, rocks and sekiro), saved as `<name>_triangles.png`.

## Bent edges

Bending is a whole-image warp. `bend_offsets` makes two smooth noise fields, one for sideways movement and one for up and down, and each pixel looks up its triangle at a slightly moved position. Because every pixel still belongs to exactly one triangle, two neighbouring triangles always bend their shared edge the same way and no gaps can open. Curving each edge on its own would have needed matching curves on every shared edge, which is hard because the recursion leaves many corners in the middle of a neighbour's side.

The noise is blurred at a smaller size and stretched back up with cubic interpolation, which is much faster than blurring it at full size. The bends fade to 0 near the picture's edges, so the outer border stays straight and does not smear. To keep my rule that no triangle folds over, the function checks the Jacobian determinant of the warp with `np.gradient` and shrinks the bends until the smallest value is at least 0.05. If a strength would fold, it is reduced automatically.

`bent_triangles_to_image` builds the label array at twice the size, bends it, colours it and shrinks every 2 x 2 block back to one pixel with `Image.Resampling.BOX`, so the curved edges come out smooth instead of stair-stepped. Bending is off by default. With `curve_strength` at 0, none of this code runs, so straight pictures take no extra time. In tests at the web app's photo size, a straight picture took about 2 seconds and a bent one about 7 to 9 seconds.

## The web app

The web app has two files. `picture_service.py` does the work and has no Streamlit code in it, so it can be tested on its own. It reads the seed box (blank means a random seed), writes the upload to a temporary folder, shrinks photos to 3000 px on the longest side after the EXIF rotation, and calls `make_triangle_art`. At most two pictures are made at the same time, and anyone else waits for a free slot. Every problem becomes a plain message, and the temporary folder is deleted when the run ends, so no files are kept.

`streamlit_app.py` is the page itself. It has a photo upload, a seed box and a slider for each setting, including the two bending sliders, inside a form, so moving a slider does nothing until Make picture is pressed. The right side shows the picture, the seed used and a Save picture button that downloads `triangles_seed<seed>.png`. Each visitor's session keeps only their latest picture.

The code is on GitHub as RaphaelJakob/Triangle-Visualisation, on the `main` branch, with `requirements.txt` listing streamlit, pillow-heif, numpy (1.25 or newer), scipy, pillow and matplotlib. My photos and their output pictures are kept out of the repo with `.gitignore`, since the repo is public. A separate Gradio version also exists for running on my own computer only, with no public link and no shrinking.

## Decisions I made

- I chose approach A: triangulate the photo first, then apply the splitting rule to those triangles.
- The splitting rule is Hobbs-style only, cutting the longest side at a Gaussian-placed point. The Sierpinski version with holes was dropped.
- Image detail only places the big starting triangles, and the recursion depth comes from the weighted coin flip.
- Points stay in pixel units, with x as the column and y as the row.
- Colours come from the middle pixel by brightness, because the per-channel median gave muddy colours that no pixel in the triangle actually had.
- Colour borrowing uses one flat chance for every triangle. Borrowing more for small triangles looked worse.
- Empty triangles take the colour at their centre instead of raising an error.
- Bending lives in `main.py` as two settings of the master function, so the web app stays a thin wrapper and bent pictures can also be made without the app.
- Bent pictures use 2x smoothing for clean curved edges, even though it is slower.
- Bending is off by default, so it does not add time to every picture.
- The output picture is drawn with Pillow at the photo's exact size, and matplotlib is kept for viewing.
- For the web app, I picked Streamlit Community Cloud because friends can use it while my computer is off, and a public repo is fine with me. Hugging Face Spaces now needs a paid plan for a Gradio app, and a Gradio share link only works while my computer is on and expires after 72 hours.

## Problems I solved

- Black hairlines between triangles were pixels that no triangle painted. Pillow drops the decimal part of corner points, and the recursion cut points are almost never whole numbers, so a corner sitting in the middle of a neighbour's side got pulled off that side and left a thin unpainted wedge. The exact pixel-centre test in `triangle_label_array` fixed this, and the gaps went from hundreds of pixels to 0 on the test photos.
- Phone photos came out sideways until `ImageOps.exif_transpose` was added, because Pillow and `np.array` ignore the EXIF rotation.
- The muddy colours were fixed by switching from per-channel medians to the brightness median.
- Random draws in one step used to depend on the settings of other steps. Giving each step its own stream fixed this.
- The first bending previews had stair-step edges and colour smears along the border. The 2x smoothing and the fade to 0 near the edges fixed both.
- Strong bends could fold triangles over. The Jacobian check now shrinks any bend that would fold.
- The first version of bending added 16 to 26 seconds per picture. Blurring the noise at a smaller size brought the extra time down to about 6 seconds.

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

## How I built it

I wrote the core of the project myself while learning NumPy along the way. This includes loading the photo and fixing its rotation, the greyscale and Sobel edge steps, the edge-weighted point picking, adding the corners, the Delaunay triangulation, the brightness-median colours, the matplotlib view, and the whole recursion (`triangle_split`, `coin_split` and `recursion_driver`) and many other small points. The design decisions, ideas, direction and creative side listed above are also entirely mine.

Once the core worked, I used Claude, an AI assistant made by Anthropic, to help finish the project. Claude updated `triangles_to_image`, made the fallback colour for empty triangles, `borrow_colour`, update the exact version of `triangle_label_array` to fix the hairlines, assisted with the master function `make_triangle_art`,created the bending functions, the Streamlit web app and the tests, and helped me decide how to host the app. I reviewed and ran this code, and I am working through each part so that I can explain every line myself.

## Credits

The splitting rule in this project is based on Tyler Hobbs's article "Aesthetically Pleasing Triangle Subdivision" (https://www.tylerxhobbs.com/words/aesthetically-pleasing-triangle-subdivision). His article describes cutting a triangle's longest side at a point near the middle and repeating the split, which is the idea behind `triangle_split` and `recursion_driver`. His work is purely generative, while this project applies the same kind of split to triangles built from a photo.

