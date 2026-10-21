# Generative triangle art

## What the project is

This is a Python program that turns a photo into an artwork made only of triangles. Points are placed on the image first, with more of them where the picture changes sharply, and a Delaunay triangulation joins them into triangles. Each triangle can then be split again and again by a recursive rule based on Tyler Hobbs's article "Aesthetically Pleasing Triangle Subdivision", so the result looks like the original photo but abstracted and a little foreign. The final picture is saved at exactly the same size as the photo.

## What I have finished

### Loading the photo and finding edges

- `importimage` opens a photo with Pillow and applies the rotation stored in its EXIF data, because a plain open ignores it.
- `to_greyscale_array` and `sobel_edgestrength` turn the photo into a grey array and then into an edge-strength map, using the Sobel filter in both directions and combining the two with `np.hypot`.

### Picking points and triangulating

- `edge_probability` raises the edge strengths to a weighting power (2.0 by default), scales them to add up to 1, and blends in an equal chance for every pixel through an evenness setting (0.2 by default). A ValueError is raised if the evenness is outside 0 to 1.
- `pick_from_probability` uses the random generator to choose that many different pixels, and `picked_to_xy_points` turns the flat positions back into x and y points in pixel units.
- `points_with_corners` adds the four image corners, and `triangulate_points` runs the Delaunay triangulation and returns a (triangles, 3, 2) array of corner coordinates.

### Recursion

- `triangle_split` works on every triangle at once. It finds each longest side with `np.roll` and `np.argmax`, draws the cut position as a fraction from a Gaussian centred on 0.5, redraws any fraction outside 0.1 to 0.9, and joins the cut to the opposite corner, so each old triangle becomes two new ones.
- `coin_split` returns one True or False per triangle by drawing a whole number from 1 to 100 and comparing it with the split chance, and it raises a ValueError if the chance is outside 0 to 1.
- `recursion_driver` keeps two piles. The active pile holds triangles that may still split, and the inactive pile holds the ones that failed a flip. Every round flips a coin for the active pile only, stores the losers, splits the winners and halves the split chance. The loop ends when the active pile is empty or after 21 rounds, and the function returns both piles joined together.

### Colouring the result

- `triangle_label_array` draws every triangle onto a blank Pillow image, filled with its position plus 1, so each pixel knows which triangle covers it and 0 means none.
- `triangle_rgb_by_brightness` sorts the pixels by triangle and then by brightness, and the middle pixel of each triangle gives its colour. A triangle with no pixels takes the image colour at the centre of its corners instead.
- `triangle_draw` shows the triangles with Matplotlib, with an optional background photo or fill colours, and `triangles_to_image` draws the filled triangles onto a Pillow image of the original size and can save it.
- A `TriangleData` dataclass holds the corners and the colours, but the pipeline does not use it yet.

## Decisions I made

- Splitting follows the Hobbs-style rule only: the longest side is cut at a Gaussian-placed point. How often a triangle splits comes from a weighted coin flip, and image detail only decides where the starting points go.
- The recursion runs on the Delaunay triangles first. Colours are worked out once afterwards, on the final triangles, so that every triangle gets one.
- The per-channel median looked muddy, so the colour is now the brightness median, which always picks a real pixel from the photo.
- Small triangles from deep splits can end up with no pixels, so an empty one gets the image colour at its centre and no error is raised.
- One random generator, seeded with 42, is passed into every random step, so a picture can be recreated.
- Points stay in pixel units, with x as the column and y as the row.
- Filled triangles are edged in their own colours, because I did not want black borders by default.

## What I will do next

- Write `borrow_colour`, where each triangle has a flat chance (30% by default) of taking its colour from a random spot elsewhere in the image.
- Add curved edges drawn with Gaussian noise, with a strength setting that defaults to 0 so the edges stay straight.
- Build a master function that takes the settings and the seed, and then an animation saved as a file that varies recursion depth, triangle count and curve strength and morphs from the original photo.
- Find out why thin dark sliver triangles still show up in some test pictures.
- Turn it into a Gradio web app so friends can change the settings, with about 30 seconds per run being acceptable.
- Write a README with credits, including the Hobbs article.