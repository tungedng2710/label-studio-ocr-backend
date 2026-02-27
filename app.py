"""Streamlit demo UI for the Surya OCR backend.

Layout:
  - Left column  : image upload + Run OCR button
  - Right column : fixed-height preview of the annotated image and extracted text

Advanced settings are shown in a popup (st.popover) so they don't clutter the main
layout.  The right-hand output section has a fixed CSS height and scrolls internally
so the page never expands vertically just because a large image was submitted.
"""

from __future__ import annotations

import base64
import json
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw
import io

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Surya OCR Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS – fix the output preview height so it never grows with the image
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Keep the output preview panel at a fixed viewport height */
    .ocr-output-panel {
        height: 70vh;
        overflow-y: auto;
        border: 1px solid #d3d3d3;
        border-radius: 6px;
        padding: 0.75rem;
        background: #fafafa;
    }
    /* Remove extra top-padding Streamlit adds to columns */
    div[data-testid="column"] > div:first-child {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("🔍 Surya OCR Demo")
st.caption("Upload an image and run the Surya OCR backend to detect and transcribe text.")

# ---------------------------------------------------------------------------
# Layout: two equal columns
# ---------------------------------------------------------------------------
left_col, right_col = st.columns(2, gap="large")

# ===========================================================================
# LEFT COLUMN – input controls
# ===========================================================================
with left_col:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    # -----------------------------------------------------------------------
    # Advanced Settings – shown as a popup so they don't take up layout space
    # -----------------------------------------------------------------------
    with st.popover("⚙️ Advanced Settings"):
        st.markdown("### Advanced Settings")
        backend_url = st.text_input(
            "OCR Backend URL",
            value=os.getenv("OCR_BACKEND_URL", "http://localhost:9090"),
            help="URL of the running Surya OCR ML backend.",
        )
        disable_math = st.checkbox(
            "Disable math mode",
            value=False,
            help="Turn off formula detection (faster on non-scientific documents).",
        )
        show_polygons = st.checkbox(
            "Draw detection polygons on preview",
            value=True,
            help="Overlay the detected text-line polygons on the preview image.",
        )
        text_color = st.color_picker("Polygon color", value="#FF0000")

    # -----------------------------------------------------------------------
    # Run button
    # -----------------------------------------------------------------------
    run_ocr = st.button("▶ Run OCR", type="primary", use_container_width=True)

    # Show a thumbnail of the uploaded image beneath the controls
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

# ===========================================================================
# RIGHT COLUMN – output preview (fixed height via CSS wrapper)
# ===========================================================================
with right_col:
    st.subheader("Output")

    # We wrap all output inside a fixed-height div so the page does *not*
    # expand vertically when a large result image is rendered.
    output_placeholder = st.empty()

    if not run_ocr or uploaded_file is None:
        with output_placeholder.container():
            st.markdown(
                '<div class="ocr-output-panel">'
                '<p style="color:#aaa;margin-top:1rem;text-align:center;">'
                "Upload an image and click <strong>▶ Run OCR</strong> to see results here."
                "</p></div>",
                unsafe_allow_html=True,
            )
    else:
        # -------------------------------------------------------------------
        # Call the backend
        # -------------------------------------------------------------------
        image_bytes = uploaded_file.getvalue()
        b64_image = base64.b64encode(image_bytes).decode()
        data_url = f"data:{uploaded_file.type};base64,{b64_image}"

        payload = {
            "tasks": [
                {
                    "id": 1,
                    "data": {"image": data_url},
                }
            ],
        }
        if disable_math:
            payload["params"] = {"disable_math": True}

        try:
            with st.spinner("Running OCR…"):
                response = requests.post(
                    f"{backend_url.rstrip('/')}/predict",
                    json=payload,
                    timeout=120,
                )
            response.raise_for_status()
            result_json = response.json()
        except requests.exceptions.ConnectionError:
            with output_placeholder.container():
                st.markdown('<div class="ocr-output-panel">', unsafe_allow_html=True)
                st.error(f"Could not connect to the backend at **{backend_url}**. "
                         "Make sure the OCR service is running.")
                st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        except requests.exceptions.HTTPError as exc:
            with output_placeholder.container():
                st.markdown('<div class="ocr-output-panel">', unsafe_allow_html=True)
                st.error(f"Backend returned an error: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        except Exception as exc:
            with output_placeholder.container():
                st.markdown('<div class="ocr-output-panel">', unsafe_allow_html=True)
                st.error(f"Unexpected error: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # -------------------------------------------------------------------
        # Parse results
        # -------------------------------------------------------------------
        predictions = result_json.get("results", result_json.get("predictions", []))
        if not predictions:
            with output_placeholder.container():
                st.markdown(
                    '<div class="ocr-output-panel">'
                    '<p style="color:#aaa;">No predictions returned.</p>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.stop()

        regions = predictions[0].get("result", [])

        # -------------------------------------------------------------------
        # Optionally draw polygons on the image
        # -------------------------------------------------------------------
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_image.size

        if show_polygons:
            draw = ImageDraw.Draw(pil_image, "RGBA")
            hex_color = text_color.lstrip("#")
            r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            fill_color = (r, g, b, 40)
            outline_color = (r, g, b, 220)

            for region in regions:
                if region.get("type") != "polygon":
                    continue
                pts_pct = region.get("value", {}).get("points", [])
                if not pts_pct:
                    continue
                pts_px = [
                    (int(x / 100 * img_w), int(y / 100 * img_h))
                    for x, y in pts_pct
                ]
                draw.polygon(pts_px, fill=fill_color, outline=outline_color)

        # Convert annotated image to bytes for display
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        annotated_bytes = buf.getvalue()

        # -------------------------------------------------------------------
        # Collect transcription lines
        # -------------------------------------------------------------------
        lines: list[str] = []
        for region in regions:
            if region.get("type") == "textarea":
                texts = region.get("value", {}).get("text", [])
                lines.extend(texts)

        # -------------------------------------------------------------------
        # Render inside fixed-height wrapper
        # -------------------------------------------------------------------
        annotated_b64 = base64.b64encode(annotated_bytes).decode()
        text_html = (
            "<br>".join(
                f'<span style="font-size:0.85rem;">{line}</span>'
                for line in lines
            )
            or '<span style="color:#aaa;">No text detected.</span>'
        )

        with output_placeholder.container():
            st.markdown(
                f"""
                <div class="ocr-output-panel">
                  <img src="data:image/png;base64,{annotated_b64}"
                       style="max-width:100%;border-radius:4px;" />
                  <hr style="margin:0.5rem 0;" />
                  <h4 style="margin:0 0 0.4rem 0;">Extracted text</h4>
                  {text_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Summary metrics below the fixed panel
        textarea_count = sum(1 for r in regions if r.get("type") == "textarea")
        st.caption(f"✅ {textarea_count} text line(s) detected.")
