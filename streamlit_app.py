"""Streamlit page for the triangle artwork. Run locally with:  streamlit run streamlit_app.py"""
import streamlit as st

from picture_service import DEFAULTS, PictureError, make_picture, parse_seed

st.set_page_config(page_title="Triangle art", layout="wide")
st.title("Triangle art")
st.write("Upload a photo, pick the settings and press **Make picture**. Leave the seed blank for a "
         "random one. The same photo, seed and settings always give the same picture.")

left, right = st.columns(2)

with left:
    # A form, so moving a slider does not rerun anything until the button is pressed.
    with st.form("settings"):
        photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp", "heic", "heif"])
        seed_text = st.text_input("Seed (leave blank for a random one)")
        starting_count = st.slider("starting_count: points picked before the 4 corners",
                                   1, 2000, int(DEFAULTS["starting_count"]), 1)   # all whole numbers, so the slider gives an int
        weighting_power = st.slider("weighting_power: how strongly points follow edges",
                                    0.0, 6.0, DEFAULTS["weighting_power"], 0.1)
        evenness = st.slider("evenness: share of points spread evenly",
                             0.0, 1.0, DEFAULTS["evenness"], 0.01)
        spread = st.slider("spread: how far from the middle sides are cut",
                           0.0, 0.5, DEFAULTS["spread"], 0.01)
        split_chance = st.slider("split_chance: starting chance a triangle splits",
                                 0.0, 1.0, DEFAULTS["split_chance"], 0.01)
        borrow_chance = st.slider("borrow_chance: chance a triangle takes a random photo colour",
                                  0.0, 1.0, DEFAULTS["borrow_chance"], 0.01)
        curve_strength = st.slider("curve_strength: how much the edges bend (0 = straight, faster)",
                                   0.0, 0.03, DEFAULTS["curve_strength"], 0.001, format="%.3f")
        curve_smoothness = st.slider("curve_smoothness: how wide each bend is",
                                     0.02, 0.12, DEFAULTS["curve_smoothness"], 0.01)
        submitted = st.form_submit_button("Make picture", type="primary")

    if submitted:
        try:
            seed = parse_seed(seed_text)
            with st.spinner("Making your picture (it waits here if two others are running)..."):
                png_bytes, seed_used = make_picture(
                    photo.getvalue() if photo is not None else None, seed,
                    starting_count, weighting_power, evenness, spread, split_chance, borrow_chance,
                    curve_strength, curve_smoothness)
            # Kept in this visitor's session only, so the picture stays when the page reruns.
            st.session_state["result"] = dict(png=png_bytes, seed=seed_used)
        except PictureError as error:
            st.session_state.pop("result", None)
            st.error(str(error))

with right:
    result = st.session_state.get("result")
    if result is not None:
        st.image(result["png"], output_format="PNG")
        st.write("Seed used:")
        st.code(str(result["seed"]), language=None)
        st.download_button("Save picture (PNG)", data=result["png"],
                           file_name=f"triangles_seed{result['seed']}.png", mime="image/png")
