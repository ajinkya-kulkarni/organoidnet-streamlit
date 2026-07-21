import colorsys
import io
import zipfile

import albumentations as A
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
    }
)
sns.set_theme(
    style="ticks",
    rc={
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
    },
)
COLORS = {"Live": "#27ae60", "Dead": "#e74c3c"}


def plot_area_distribution(df):
    figs = {}
    for status in ("Live", "Dead"):
        sub = df[df["Status"] == status]
        n = len(sub)
        if n == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        bins = min(30, max(8, n // 5))
        sns.histplot(
            sub["area"],
            bins=bins,
            stat="density",
            alpha=0.3,
            color=COLORS[status],
            edgecolor=COLORS[status],
            linewidth=0.4,
            ax=ax,
        )
        if n >= 2:
            sns.kdeplot(
                sub["area"], color=COLORS[status], linewidth=1.8, bw_adjust=0.5, ax=ax
            )
            mean_val = sub["area"].mean()
            ax.axvline(
                mean_val,
                color=COLORS[status],
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=f"{status} mean",
            )
            leg = ax.legend(fontsize=8, framealpha=0.9, edgecolor="#b0b0b0")
            leg.get_frame().set_linewidth(0.5)
        ax.set_title(f"{status} organoids — area distribution", fontsize=12, pad=6)
        ax.set_xlabel("Area (px²)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax, top=True, right=True)
        fig.tight_layout()
        figs[status] = fig
    return figs


def show_summary_metrics(total_organoids, total_area, n_live, n_dead, mean_live_area, mean_dead_area, mean_area):
    columns = st.columns(7)
    columns[0].metric("Total organoids", total_organoids)
    columns[1].metric("Total area", f"{total_area:.0f} px²")
    columns[2].metric("Live", n_live)
    columns[3].metric("Dead", n_dead)
    columns[4].metric("Mean live area", f"{mean_live_area:.0f} px²" if mean_live_area else "—")
    columns[5].metric("Mean dead area", f"{mean_dead_area:.0f} px²" if mean_dead_area else "—")
    columns[6].metric("Mean area", f"{mean_area:.0f} px²")


def show_morphology(df, title):
    st.subheader(title)
    for fig in plot_area_distribution(df).values():
        st.pyplot(fig)


st.set_page_config(page_title="OrganoIDNet", layout="wide")
st.title("OrganoIDNet")
st.caption("If you use this application, please cite the OrganoIDNet paper: https://doi.org/10.1007/s13402-024-00958-2")

cellseg_model = load_model()


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

    intensity_threshold = st.slider(
        "Intensity threshold for live/dead classification (0 = all live, 255 = all dead)",
        min_value=0, max_value=255, value=50,
    )

    submitted = st.form_submit_button("Analyze", use_container_width=True)


def predict_fn(p):
    return predict(cellseg_model, p)


if not uploaded_files:
    st.stop()

if submitted:
    n_files = len(uploaded_files)
    if n_files > 20:
        st.error("Maximum 20 images allowed.")
        st.stop()

    # ── Single image: full detail view ──────────────────────────────────────
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
        total = len(stats_df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Input", width="stretch")
        with col2:
            st.image(
                render_instance_mask(instances),
                caption="Instance mask",
                width="stretch",
            )
        with col3:
            st.image(
                draw_classified_outlines(img, instances, live_ids, dead_ids),
                caption="Overlay",
                width="stretch",
            )

        n_live = len(live_ids)
        n_dead = len(dead_ids)
        live_df = stats_df[stats_df["Status"] == "Live"] if total > 0 else pd.DataFrame()
        dead_df = stats_df[stats_df["Status"] == "Dead"] if total > 0 else pd.DataFrame()
        total_area = stats_df["area"].sum() if total > 0 else 0
        mean_live_area = live_df["area"].mean() if len(live_df) > 0 else 0
        mean_dead_area = dead_df["area"].mean() if len(dead_df) > 0 else 0
        mean_area = stats_df["area"].mean() if total > 0 else 0

        show_summary_metrics(
            total, total_area, n_live, n_dead, mean_live_area, mean_dead_area, mean_area
        )

        if total > 0:
            show_morphology(stats_df, "Area distributions")

            st.subheader("Per-organoid details")
            display = stats_df[
                [
                    "label",
                    "area",
                    "mean_intensity",
                    "Status",
                ]
            ]
            display.columns = [
                "ID",
                "Area (px²)",
                "Mean intensity",
                "Status",
            ]
            st.dataframe(display, width="stretch", hide_index=True)

            stem = f.name.rsplit(".", 1)[0]
            mask_bytes, mask_name = _mask_to_bytes(instances, stem)
            st.download_button(
                "Download instance mask as TIFF",
                mask_bytes,
                mask_name,
                "image/tiff",
            )

    # ── Multiple images: aggregate view ─────────────────────────────────────
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

        all_dfs = [df for _, _, df in results if len(df) > 0]

        if not all_dfs:
            st.warning("No organoids detected in any of the uploaded images.")
        else:
            combined = pd.concat(all_dfs, ignore_index=True)

            total_organoids = len(combined)
            n_live = int((combined["Status"] == "Live").sum())
            n_dead = int((combined["Status"] == "Dead").sum())
            live_df = combined[combined["Status"] == "Live"] if n_live > 0 else pd.DataFrame()
            dead_df = combined[combined["Status"] == "Dead"] if n_dead > 0 else pd.DataFrame()
            total_area = combined["area"].sum()
            mean_live_area = live_df["area"].mean() if len(live_df) > 0 else 0
            mean_dead_area = dead_df["area"].mean() if len(dead_df) > 0 else 0
            mean_area = combined["area"].mean()

            show_summary_metrics(
                total_organoids,
                total_area,
                n_live,
                n_dead,
                mean_live_area,
                mean_dead_area,
                mean_area,
            )

            st.subheader("Per-image summary")
            summary_rows = []
            for name, _, df in results:
                t = len(df)
                lv = int((df["Status"] == "Live").sum()) if t > 0 else 0
                dd = int((df["Status"] == "Dead").sum()) if t > 0 else 0
                summary_rows.append(
                    {
                        "Image": name,
                        "Organoids": t,
                        "Live": lv,
                        "Dead": dd,
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

            show_morphology(combined, "Area distributions (aggregate)")

            csv = combined.to_csv(index=False).encode()
            st.download_button(
                "Download all data as CSV",
                csv,
                "organoid_data.csv",
                "text/csv",
            )

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for name, instances, _ in results:
                    stem = name.rsplit(".", 1)[0]
                    mask_bytes, _ = _mask_to_bytes(instances, stem)
                    zf.writestr(f"{stem}_instance_mask.tif", mask_bytes)
            st.download_button(
                "Download all instance masks as ZIP",
                zip_buf.getvalue(),
                "instance_masks.zip",
                "application/zip",
            )
