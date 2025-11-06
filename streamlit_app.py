"""Streamlit interface for MedExplain AI – Multimodal Clinical Insight Assistant."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image

from vision_pipeline import (
    DEVICE as VISION_DEVICE,
    FINETUNED_MODEL_PATH as VISION_MODEL_PATH,
    generate_gradcam,
    get_data_transforms,
    load_model,
    visualize_heatmap_on_image,
)
from nlp_pipeline import FINETUNED_MODEL_PATH as NLP_MODEL_PATH, generate_summary

st.set_page_config(page_title="MedExplain AI", layout="wide")


@st.cache_resource(show_spinner=False)
def load_vision_model():
    return load_model(VISION_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_vision_transforms():
    transforms = get_data_transforms()
    return transforms["val"]


def prepare_image(image: Image.Image, transform) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)


def display_gradcam(image: Image.Image, model) -> Tuple[Image.Image, int, float]:
    transform = load_vision_transforms()
    input_tensor = prepare_image(image, transform)
    baseline_tensor = input_tensor.clone()
    input_tensor = input_tensor.to(VISION_DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()

    heatmap = generate_gradcam(model, input_tensor, target_category=predicted_class)
    overlay_tensor = visualize_heatmap_on_image(heatmap, baseline_tensor.squeeze(0))
    overlay_np = overlay_tensor.detach().permute(1, 2, 0).numpy()
    overlay_img = Image.fromarray((overlay_np * 255).astype(np.uint8))
    return overlay_img, predicted_class, confidence


def render_xray_tab():
    st.header("X-Ray Diagnosis")
    st.write(
        "Upload a chest X-ray to classify it as Normal or Pneumonia. "
        "A Grad-CAM heatmap highlights regions that influenced the prediction."
    )
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"]) 

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        model = load_vision_model()
        overlay_img, predicted_class, confidence = display_gradcam(image, model)

        label_map = {0: "Normal", 1: "Pneumonia"}
        st.subheader("Prediction")
        st.write(f"**Class:** {label_map.get(predicted_class, 'Unknown')} | **Confidence:** {confidence:.2%}")

        st.subheader("Grad-CAM Heatmap")
        st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)
    else:
        st.info("Please upload a chest X-ray image to generate predictions.")

    st.caption(
        f"Vision model checkpoint: {VISION_MODEL_PATH if VISION_MODEL_PATH.exists() else 'Not found. Train and update path.'}"
    )


def render_text_tab():
    st.header("Text Summarization")
    st.write("Paste a clinical or scientific passage to receive a concise summary.")

    user_input = st.text_area("Enter text to summarize", height=300)
    if st.button("Generate Summary"):
        if user_input.strip():
            with st.spinner("Generating summary..."):
                summary = generate_summary(user_input)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("Please provide text to summarize.")

    st.caption(
        f"NLP model checkpoint: {NLP_MODEL_PATH if NLP_MODEL_PATH.exists() else 'Not found. Run fine_tune() to create it.'}"
    )


def main():
    st.title("MedExplain AI – Multimodal Clinical Insight Assistant")
    st.write(
        "This demo combines a vision pipeline for chest X-ray analysis with an NLP pipeline for scientific text summarization."
    )

    tab1, tab2 = st.tabs(["X-Ray Diagnosis", "Text Summarization"])
    with tab1:
        render_xray_tab()
    with tab2:
        render_text_tab()


if __name__ == "__main__":
    main()
