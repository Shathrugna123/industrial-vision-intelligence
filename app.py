
from pathlib import Path
import sys
import json
import io

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

# Make project modules importable when Streamlit runs app.py.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from mvtec_dataset import MVTecDataset, IMAGE_SIZE
from feature_extractor import FeatureExtractor, DEVICE
from patchcore_baseline import build_memory_bank


CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MAX_MEMORY_FEATURES = 2000
BATCH_SIZE = 4

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_model():
    model = FeatureExtractor(pretrained=True).to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_memory_bank(category):
    dataset = MVTecDataset(category, split="train")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_model()
    features = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            features.append(model(images).cpu())

    feature_maps = torch.cat(features, dim=0)

    return build_memory_bank(
        feature_maps,
        max_features=MAX_MEMORY_FEATURES,
    )


def generate_anomaly_map(image_tensor, memory_bank):
    model = load_model()

    with torch.no_grad():
        feature_map = model(
            image_tensor.unsqueeze(0).to(DEVICE)
        )[0].cpu()

    # [C, H, W] -> [H*W, C]
    vectors = feature_map.permute(1, 2, 0).reshape(
        -1, feature_map.shape[0]
    )

    # Process query pixels in chunks to limit memory usage.
    distances = []
    chunk_size = 128

    with torch.no_grad():
        for start in range(0, len(vectors), chunk_size):
            chunk = vectors[start:start + chunk_size]
            nearest = torch.cdist(
                chunk, memory_bank
            ).min(dim=1).values
            distances.append(nearest)

    score_map = torch.cat(distances).reshape(
        feature_map.shape[1],
        feature_map.shape[2],
    )

    score_map = F.interpolate(
        score_map[None, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return score_map.numpy()


def create_overlay(image, anomaly_map):
    heatmap = np.clip(anomaly_map, 0, None)
    low, high = np.percentile(heatmap, [1, 99])

    if high > low:
        heatmap = (heatmap - low) / (high - low)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap = np.clip(heatmap, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Vision Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.title("Industrial Vision Intelligence & Quality Reliability System")

st.write(
    "Prototype for unsupervised visual anomaly detection and "
    "defect localization using a ResNet18 feature memory bank."
)

st.info(
    "This is a research prototype. Anomaly scores are not probabilities, "
    "and the inspection decision is a configurable screening result, "
    "not a certified quality verdict."
)


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

with st.sidebar:
    st.header("Inspection settings")

    category = st.selectbox(
        "Product category",
        CATEGORIES,
    )

    threshold = st.slider(
        "Anomaly threshold (P95 score)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.25,
        help=(
            "Illustrative starting threshold only. Calibrate it using "
            "held-out normal validation images before operational use."
        ),
    )

    st.caption(f"Inference device: {DEVICE}")
    st.caption("Model: frozen ResNet18 + nearest-neighbor memory bank")


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["png", "jpg", "jpeg", "bmp"],
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        display_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_tensor = IMAGE_TRANSFORM(image)

        with st.spinner(
            f"Loading {category} reference features..."
        ):
            memory_bank = load_memory_bank(category)

        with st.spinner("Analyzing image..."):
            anomaly_map = generate_anomaly_map(
                image_tensor,
                memory_bank,
            )

        # P95 is the experimental image scoring method.
        score = float(np.percentile(anomaly_map, 95))

        decision = (
            "REVIEW — POTENTIAL ANOMALY"
            if score >= threshold
            else "NO ANOMALY FLAGGED"
        )

        # --------------------------------------------------
        # Inspection result
        # --------------------------------------------------

        st.subheader("Inspection result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "P95 anomaly score",
            f"{score:.4f}",
        )

        col2.metric(
            "Configured threshold",
            f"{threshold:.2f}",
        )

        col3.metric(
            "Screening result",
            decision,
        )

        st.caption(
            "The threshold is user-configurable and not yet calibrated "
            "for production. A low score does not prove the product "
            "is defect-free."
        )

        # --------------------------------------------------
        # Original image, heatmap and overlay
        # --------------------------------------------------

        original_col, heatmap_col, overlay_col = st.columns(3)

        with original_col:
            st.markdown("**Original image**")
            st.image(
                display_image,
                use_container_width=True,
            )

        with heatmap_col:
            st.markdown("**Anomaly heatmap**")

            fig_heat, ax_heat = plt.subplots(figsize=(5, 5))

            im = ax_heat.imshow(
                anomaly_map,
                cmap="jet",
            )

            ax_heat.axis("off")

            fig_heat.colorbar(
                im,
                ax=ax_heat,
                fraction=0.046,
                pad=0.04,
            )

            fig_heat.tight_layout()
            st.pyplot(fig_heat)
            plt.close(fig_heat)

        with overlay_col:
            st.markdown("**Localization overlay**")

            overlay_fig = create_overlay(
                display_image,
                anomaly_map,
            )

            st.pyplot(overlay_fig)
            plt.close(overlay_fig)

        # --------------------------------------------------
        # Inspection report data
        # --------------------------------------------------

        report = {
            "filename": uploaded_file.name,
            "category": category,
            "model": (
                "Simplified PatchCore-style ResNet18 "
                "feature memory bank"
            ),
            "device": str(DEVICE),
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "image_score_method": (
                "95th percentile of anomaly map"
            ),
            "anomaly_score": round(score, 6),
            "configured_threshold": threshold,
            "screening_result": decision,
            "warning": (
                "Research prototype. Scores are not probabilities. "
                "Threshold requires validation on held-out normal images."
            ),
        }

        # --------------------------------------------------
        # Download inspection reports
        # --------------------------------------------------

        st.subheader("Download inspection report")

        # JSON report
        report_json = json.dumps(
            report,
            indent=2,
        )

        # CSV report, suitable for Excel and spreadsheets
        csv_report = pd.DataFrame(
            [report]
        ).to_csv(index=False)

        json_col, csv_col = st.columns(2)

        with json_col:
            st.download_button(
                "Download JSON report",
                data=report_json,
                file_name="inspection_report.json",
                mime="application/json",
                use_container_width=True,
            )

        with csv_col:
            st.download_button(
                "Download CSV report",
                data=csv_report,
                file_name="inspection_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # --------------------------------------------------
        # Download anomaly heatmap
        # --------------------------------------------------

        heatmap_normalized = (
            (anomaly_map - anomaly_map.min())
            / (np.ptp(anomaly_map) + 1e-8)
            * 255
        )

        heatmap_bytes = np.clip(
            heatmap_normalized,
            0,
            255,
        ).astype(np.uint8)

        buffer = io.BytesIO()

        Image.fromarray(
            heatmap_bytes
        ).save(
            buffer,
            format="PNG",
        )

        st.download_button(
            "Download anomaly heatmap",
            data=buffer.getvalue(),
            file_name="anomaly_heatmap.png",
            mime="image/png",
        )

    except Exception as exc:
        st.error(f"Inspection failed: {exc}")
        st.exception(exc)


else:
    st.markdown(
        """
        ### How to use the prototype

        1. Select a product category in the sidebar.
        2. Upload an image of that product.
        3. Review the anomaly score and heatmap.
        4. Download the JSON or CSV inspection report.
        5. Download the anomaly heatmap if required.

        **Dataset limitation:** The reference bank contains normal MVTec
        training images. For meaningful results, upload images similar
        to the selected category and the dataset's image conditions.
        """
    )
