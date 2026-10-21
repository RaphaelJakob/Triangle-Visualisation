"""The part of the web app that makes pictures. It has no Streamlit code, so it can be tested on
its own. It is a thin wrapper around make_triangle_art from main.py: it only reads the seed,
shrinks big photos, limits how many runs happen at once and cleans up its temporary files.
"""
import os
import secrets
import tempfile
import threading
import traceback

from PIL import Image, UnidentifiedImageError

from main import importimage, make_triangle_art

# iPhone HEIC photos need the pillow-heif plugin before Pillow can open them.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

MAX_SIDE = 3000          # longest side a photo is shrunk to before a run; None keeps full size
MAX_UPLOAD_MB = 25       # largest photo accepted (also set in .streamlit/config.toml)
RUNS_AT_ONCE = 2         # runs that can happen at the same time; others wait
DEFAULTS = dict(starting_count=120, weighting_power=2.0, evenness=0.2,
                spread=0.2, split_chance=0.7, borrow_chance=0.3)

# One shared set of run slots for everyone using the app. Python imports this module only once
# per server, so every visitor's session uses the same two slots.
RUN_SLOTS = threading.Semaphore(RUNS_AT_ONCE)


class PictureError(Exception):
    """A problem to show to the person using the app, as a plain message."""


def parse_seed(seed_text: str | None) -> int | None:
    """
    Takes the text from the seed box. Returns None when it is blank (meaning pick a random
    seed) and the whole number otherwise, and raises a PictureError if it is not a whole number.
    """
    if seed_text is None or seed_text.strip() == "":
        return None
    try:
        return int(seed_text.strip())
    except ValueError:
        raise PictureError("The seed must be a whole number of 0 or more, or left blank.") from None


def as_whole_number(value):
    """Turns a float like 120.0 into the int 120, and leaves anything else unchanged."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def prepare_photo(photo_path: str, folder: str) -> str:
    """
    Takes a photo path and a temporary folder. If the photo's longest side is above MAX_SIDE,
    it opens the photo with importimage (so the EXIF rotation is applied first), shrinks it to
    fit MAX_SIDE, saves it as a PNG in the folder and returns that path. Otherwise it returns
    the original path unchanged.
    """
    with Image.open(photo_path) as header:
        width, height = header.size
    if MAX_SIDE is None or max(width, height) <= MAX_SIDE:
        return photo_path
    image = importimage(photo_path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    small_path = os.path.join(folder, "photo_small.png")
    image.save(small_path, compress_level=1)   # lossless; low compression is faster to write
    return small_path


def make_picture(photo_bytes: bytes | None, seed: int | None = None,
                 starting_count=DEFAULTS["starting_count"], weighting_power=DEFAULTS["weighting_power"],
                 evenness=DEFAULTS["evenness"], spread=DEFAULTS["spread"],
                 split_chance=DEFAULTS["split_chance"],
                 borrow_chance=DEFAULTS["borrow_chance"]) -> tuple[bytes, int]:
    """
    Takes the uploaded photo's bytes, a seed (None means pick a random one) and the six
    settings. It writes the photo to a temporary folder, shrinks it if it is big, waits for a
    free run slot, runs make_triangle_art and reads the picture back as PNG bytes. The folder
    and everything in it is deleted before returning. Returns the PNG bytes and the seed used,
    and raises a PictureError with a plain message for any problem.
    """
    if not photo_bytes:
        raise PictureError("Please upload a photo first.")
    if len(photo_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise PictureError(f"The photo is larger than {MAX_UPLOAD_MB} MB.")
    if seed is None:
        seed = secrets.randbelow(2**31)
    seed = as_whole_number(seed)
    starting_count = as_whole_number(starting_count)

    with RUN_SLOTS:
        with tempfile.TemporaryDirectory() as folder:
            photo_path = os.path.join(folder, "photo")   # Pillow reads the type from the content
            with open(photo_path, "wb") as photo_file:
                photo_file.write(photo_bytes)
            output_path = os.path.join(folder, "picture.png")
            try:
                used_path = prepare_photo(photo_path, folder)
                make_triangle_art(used_path, seed, starting_count, weighting_power, evenness,
                                  spread, split_chance, borrow_chance, output_path=output_path)
            except ValueError as error:
                raise PictureError(f"Setting problem: {error}") from None
            except (UnidentifiedImageError, OSError):
                hint = "" if HEIC_SUPPORTED else " HEIC photos need the pillow-heif package on the server."
                raise PictureError("This file could not be opened as a photo." + hint) from None
            except Exception:
                traceback.print_exc()   # full details go to the server log, not to the person
                raise PictureError("Something went wrong while making the picture. "
                                   "Try other settings or another photo.") from None
            with open(output_path, "rb") as picture_file:
                png_bytes = picture_file.read()
    return png_bytes, seed
