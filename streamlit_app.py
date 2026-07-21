import colorsys
import io
import zipfile

import albumentations as A
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.postproc.functional.cellpose.cellpose import (
    post_proc_cellpose,
)
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization
from patch_inference import predict_large_image
from skimage.measure import regionprops_table

CKPT = "models/best.pt"
DEVICE = "cpu"
TRANSFORM = A.Compose([MinMaxNormalization(always_apply=True)])
OUTLINE_COLORS = {"Live": (0, 255, 0), "Dead": (255, 0, 0)}

@st.cache_resource
def load_model():
    model = cellpose_nuclei(
        n_nuc_classes=2, enc_name="efficientnet_b3", enc_pretrain=False
    )
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def predict(model, img_rgb):
    img_norm = TRANSFORM(image=img_rgb)["image"]
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    out = model(tensor)
    type_prob = torch.softmax(out["nuc"].type_map, dim=1)
    fg_prob = type_prob[0, 1].cpu().numpy()
    flow = out["nuc"].aux_map[0].cpu().numpy()
    instances = post_proc_cellpose(fg_prob > 0.5, flow, min_size=30)
    return instances

def classify_organoids(instances, gray_img, threshold=50):
    live, dead = [], []
    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        mask = instances == inst_id
        if gray_img[mask].mean() >= threshold:
            live.append(inst_id)
        else:
            dead.append(inst_id)
    return live, dead

def random_label_cmap(n=2**16, h=(0, 1), lightness=(0.4, 1), s=(0.2, 0.8)):
    h_vals = np.random.uniform(*h, n)
    l_vals = np.random.uniform(*lightness, n)
    s_vals = np.random.uniform(*s, n)
    cols = np.stack(
        [colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h_vals, l_vals, s_vals)],
        axis=0,
    )
    cols[0] = 0
    return mpl.colors.ListedColormap(cols)

_LABEL_CMAP = random_label_cmap()

def render_instance_mask(inst):
    return (_LABEL_CMAP(inst)[:, :, :3] * 255).astype(np.uint8)

def _draw_instance_boxes(overlay, instances, instance_ids, color, thickness=3):
    for inst_id in instance_ids:
        ys, xs = np.where(instances == inst_id)
        if len(ys) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        for t in range(thickness):
            overlay[y1:y2+1, x1+t] = color
            overlay[y1:y2+1, x2-t] = color
            overlay[y1+t, x1:x2+1] = color
            overlay[y2-t, x1:x2+1] = color

def draw_classified_outlines(img, instances, live_ids, dead_ids):
    overlay = img.copy()
    if len(np.unique(instances)) <= 1:
        return overlay
    _draw_instance_boxes(overlay, instances, live_ids, OUTLINE_COLORS["Live"])
    _draw_instance_boxes(overlay, instances, dead_ids, OUTLINE_COLORS["Dead"])
    return overlay

def _build_summary_row(name, df):
    t = len(df)
    if t == 0:
        return {"Image": name, "Organoids": 0, "Live": 0, "Dead": 0}
    lv = int((df["Status"] == "Live").sum())
    dd = int((df["Status"] == "Dead").sum())
    total_area = df["area"].sum()
    live_df = df[df["Status"] == "Live"]
    dead_df = df[df["Status"] == "Dead"]
    return {
        "Image": name,
        "Organoids": t,
        "Total area (px²)": round(total_area),
        "Live": lv,
        "Dead": dd,
        "Mean live area (px²)": round(live_df["area"].mean()) if len(live_df) > 0 else None,
        "Mean dead area (px²)": round(dead_df["area"].mean()) if len(dead_df) > 0 else None,
        "Mean area (px²)": round(df["area"].mean()),
    }

def compute_stats(instances, img, threshold=50):
    gray = np.mean(img, axis=2)
    props = regionprops_table(
        instances,
        intensity_image=gray,
        properties=("label", "area", "mean_intensity"),
    )
    df = pd.DataFrame(props)
    if len(df) == 0:
        return df
    df["Status"] = np.where(df["mean_intensity"] >= threshold, "Live", "Dead")
    return df

st.set_page_config(page_title="OrganoIDNet", layout="wide")
st.title("OrganoIDNet")
st.caption("If you use this application, please cite the OrganoIDNet paper: https://doi.org/10.1007/s13402-024-00958-2")

cellseg_model = load_model()

if "results" not in st.session_state:
    st.session_state.results = None

def load_image(f):
    return np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))

def _mask_to_bytes(mask, stem="mask"):
    buf = io.BytesIO()
    Image.fromarray(mask, mode="I").save(buf, format="TIFF")
    return buf.getvalue(), f"{stem}.tif"

with st.form(key="analyze_form"):
    uploaded_files = st.file_uploader(
        "Upload organoid images (256\u00d7256 patches or larger, up to 2000px per side; up to 20)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    intensity_threshold = st.number_input(
        "Intensity threshold for live/dead classification (0 = all live, 255 = all dead)",
        min_value=0, max_value=255, value=50, step=1,
    )

    submitted = st.form_submit_button("Analyze", use_container_width=True)

def predict_fn(p):
    return predict(cellseg_model, p)

if submitted:
    st.session_state.results = None
    n_files = len(uploaded_files)
    if n_files > 20:
        st.error("Maximum 20 images allowed.")
        st.stop()

    # ── Single image ────────────────────────────────────────────────────────
    if n_files == 1:
        f = uploaded_files[0]
        img = load_image(f)
        prog = st.progress(0, text="Segmenting organoids...")

        def on_single_tile(d, t):
            prog.progress(d / t, text=f"Tile {d}/{t}")

        try:
            instances = predict_large_image(
                predict_fn,
                img,
                progress_callback=on_single_tile,
            )
        except Exception as e:
            prog.empty()
            st.error(f"Inference failed: {e}")
            st.stop()
        prog.empty()
        gray = np.mean(img, axis=2)
        live_ids, dead_ids = classify_organoids(instances, gray, intensity_threshold)
        stats_df = compute_stats(instances, img, intensity_threshold)
        stem = f.name.rsplit(".", 1)[0]
        mask_bytes, _ = _mask_to_bytes(instances, stem)
        st.session_state.results = {
            "mode": "single",
            "name": f.name,
            "img": img,
            "instances": instances,
            "stats_df": stats_df,
            "live_ids": live_ids,
            "dead_ids": dead_ids,
            "mask_bytes": mask_bytes,
            "mask_stem": stem,
        }
        st.rerun()

    # ── Multiple images ─────────────────────────────────────────────────────
    else:
        results = []
        prog = st.progress(0, text="Segmenting organoids...")
        for i, f in enumerate(uploaded_files):
            img = load_image(f)
            name = f.name

            def on_batch_tile(d, t, name=name, i=i):
                frac = (i + d / t) / n_files
                prog.progress(min(frac, 1.0), text=f"{name}  |  tile {d}/{t}")

            try:
                instances = predict_large_image(
                    predict_fn,
                    img,
                    progress_callback=on_batch_tile,
                )
            except Exception as e:
                prog.empty()
                st.error(f"Inference failed for {name}: {e}")
                st.stop()
            stats_df = compute_stats(instances, img, intensity_threshold)
            results.append((f.name, instances, stats_df))
        prog.empty()

        all_dfs = []
        for name, _, df in results:
            if len(df) > 0:
                df = df.copy()
                df.insert(0, "Image", name)
                all_dfs.append(df)

        if not all_dfs:
            st.warning("No organoids detected in any of the uploaded images.")
            st.session_state.results = None
        else:
            combined = pd.concat(all_dfs, ignore_index=True)
            csv = combined.to_csv(index=False).encode()
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for name, instances, _ in results:
                    stem = name.rsplit(".", 1)[0]
                    mask_bytes, _ = _mask_to_bytes(instances, stem)
                    zf.writestr(f"{stem}_instance_mask.tif", mask_bytes)
            st.session_state.results = {
                "mode": "multi",
                "results": results,
                "combined": combined,
                "csv": csv,
                "zip": zip_buf.getvalue(),
            }
            st.rerun()

# ── Display results from session state ───────────────────────────────────
if st.session_state.results is not None:
    r = st.session_state.results

    if r["mode"] == "single":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(r["img"], caption="Input", width="stretch")
        with col2:
            st.image(
                render_instance_mask(r["instances"]),
                caption="Instance mask",
                width="stretch",
            )
        with col3:
            st.image(
                draw_classified_outlines(r["img"], r["instances"], r["live_ids"], r["dead_ids"]),
                caption="Overlay",
                width="stretch",
            )

        total = len(r["stats_df"])
        if total > 0:
            st.subheader("Summary")
            _cfg = {
                "Image": st.column_config.TextColumn("Image", alignment="center"),
                "Organoids": st.column_config.NumberColumn("Organoids", alignment="center"),
                "Total area (px²)": st.column_config.NumberColumn("Total area (px²)", alignment="center"),
                "Live": st.column_config.NumberColumn("Live", alignment="center"),
                "Dead": st.column_config.NumberColumn("Dead", alignment="center"),
                "Mean live area (px²)": st.column_config.NumberColumn("Mean live area (px²)", alignment="center"),
                "Mean dead area (px²)": st.column_config.NumberColumn("Mean dead area (px²)", alignment="center"),
                "Mean area (px²)": st.column_config.NumberColumn("Mean area (px²)", alignment="center"),
            }
            st.dataframe(
                pd.DataFrame([_build_summary_row(r["name"], r["stats_df"])]),
                width="stretch",
                hide_index=True,
                column_config=_cfg,
            )

            st.subheader("Per-organoid details")
            display = r["stats_df"][["label", "area", "mean_intensity", "Status"]]
            display.columns = ["ID", "Area (px²)", "Mean intensity", "Status"]
            st.dataframe(display, width="stretch", hide_index=True)

            st.download_button(
                "Download instance mask as TIFF",
                r["mask_bytes"],
                f"{r['mask_stem']}.tif",
                "image/tiff",
            )

    else:
        results = r["results"]

        st.subheader("Per-image summary")
        _cfg = {
            "Image": st.column_config.TextColumn("Image", alignment="center"),
            "Organoids": st.column_config.NumberColumn("Organoids", alignment="center"),
            "Total area (px²)": st.column_config.NumberColumn("Total area (px²)", alignment="center"),
            "Live": st.column_config.NumberColumn("Live", alignment="center"),
            "Dead": st.column_config.NumberColumn("Dead", alignment="center"),
            "Mean live area (px²)": st.column_config.NumberColumn("Mean live area (px²)", alignment="center"),
            "Mean dead area (px²)": st.column_config.NumberColumn("Mean dead area (px²)", alignment="center"),
            "Mean area (px²)": st.column_config.NumberColumn("Mean area (px²)", alignment="center"),
        }
        summary_rows = [_build_summary_row(name, df) for name, _, df in results]
        st.dataframe(
            pd.DataFrame(summary_rows),
            width="stretch",
            hide_index=True,
            column_config=_cfg,
        )

        st.download_button(
            "Download all data as CSV",
            r["csv"],
            "organoid_data.csv",
            "text/csv",
        )

        st.download_button(
            "Download all instance masks as ZIP",
            r["zip"],
            "instance_masks.zip",
            "application/zip",
        )
