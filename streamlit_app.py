
import os
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.patches import Circle, Ellipse

# ==============================
# Palette CSV helpers
# ==============================

PALETTE_FILE = "palette.csv"

# Colab 버전처럼 palette.csv 없으면 기본값 생성
if not os.path.exists(PALETTE_FILE):
    df_init = pd.DataFrame(
        [
            {"name": "sky", "r": 0.4, "g": 0.7, "b": 1.0},
            {"name": "lemon", "r": 1.0, "g": 1.0, "b": 0.843},
        ]
    )
    df_init.to_csv(PALETTE_FILE, index=False)


def read_palette() -> pd.DataFrame:
    try:
        return pd.read_csv(PALETTE_FILE)
    except Exception:
        return pd.DataFrame(columns=["name", "r", "g", "b"])


def load_csv_palette() -> list:
    """palette.csv → [(r,g,b), ...]  (0~1 범위)"""
    df = read_palette()
    colors = []
    for _, row in df.iterrows():
        colors.append((row["r"], row["g"], row["b"]))
    if not colors:
        colors = [(0.4, 0.7, 1.0), (1.0, 1.0, 0.843)]
    return colors


# ==============================
# Color schemes
# ==============================

def hex_to_rgb_frac(hexcolor: str):
    hexcolor = hexcolor.lstrip("#")
    r = int(hexcolor[0:2], 16) / 255.0
    g = int(hexcolor[2:4], 16) / 255.0
    b = int(hexcolor[4:6], 16) / 255.0
    return (r, g, b)


# pastel 계열 (원래 consistent_color_scheme)
consistent_color_scheme = [
    "#fbb4ae",
    "#b3cde3",
    "#ccebc5",
    "#decbe4",
    "#fed9a6",
    "#ffffcc",
    "#e5d8bd",
    "#fddaec",
    "#f2f2f2",
    "#b3e2cd",
]

# vivid / neon / monochrome, 원래 코드 기반
vivid_color_scheme = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#9781bf",
]

neonPop_color_scheme = [
    "#FEFF00",
    "#39FF14",
    "#00FFFF",
    "#FF00FF",
    "#FF1493",
    "#FF4500",
    "#FFFFF0",
    "#7CFC00",
    "#ADFF2F",
]

monochrome_color_scheme = [
    "#000000",
    "#333333",
    "#666666",
    "#999999",
    "#CCCCCC",
    "#EEEEEE",
    "#F5F5F5",
]


def adjust_lightness(color_rgb, factor: float):
    r, g, b = color_rgb
    return tuple(min(1.0, max(0.0, c * factor)) for c in (r, g, b))


def get_color_scheme(selection: str):
    """팔레트 이름 → RGB 리스트(0~1)"""
    if selection == "vivid":
        return [hex_to_rgb_frac(c) for c in vivid_color_scheme]
    if selection == "pastel":
        return [hex_to_rgb_frac(c) for c in consistent_color_scheme]
    if selection == "neonPop":
        return [hex_to_rgb_frac(c) for c in neonPop_color_scheme]
    if selection == "monochrome":
        return [hex_to_rgb_frac(c) for c in monochrome_color_scheme]
    if selection == "csv":
        return load_csv_palette()
    # default
    return [hex_to_rgb_frac(c) for c in consistent_color_scheme]


# ==============================
# Core blob drawing (Streamlit용)
# ==============================

def draw_poster(
    seed: int | None = None,
    visualization_mode: str = "filled_gradient",  # "filled_gradient" or "contour_line"
    palette_name: str = "pastel",
    bgcolor: str = "#fdf7f2",
    margin: float = 0.1,
    num_blobs: int = 6,
    wobble_factor: float = 0.05,
    min_radius: float = 0.06,
    max_radius: float = 0.09,
    num_gradient_steps: int = 30,
    horizon_position: float = 0.6,
):
    # seed 고정/랜덤
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    color_scheme = get_color_scheme(palette_name)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # 배경색
    bg_rgb = hex_to_rgb_frac(bgcolor)
    fig.patch.set_facecolor(bg_rgb)
    ax.set_facecolor(bg_rgb)

    # 위쪽/아래쪽 살짝 다른 톤으로 나눠서 "벽+바닥" 느낌
    ax.axhspan(0, horizon_position * 4 / 3, color=bg_rgb, zorder=0)
    ax.axhspan(
        horizon_position * 4 / 3,
        4 / 3,
        color=(bg_rgb[0] * 0.95, bg_rgb[1] * 0.95, bg_rgb[2] * 0.95),
        zorder=0,
    )

    blobs = []

    for _ in range(num_blobs):
        # 화면 안 랜덤 위치
        x = random.uniform(margin, 1.0 - margin)
        y = random.uniform(margin, 4 / 3 * horizon_position * 1.1)

        # y 기반으로 depth(먼/가까운) 계산
        depth_ratio = y / (4 / 3)

        # 멀리 있을수록 작아지게
        max_possible_radius = min_radius + (max_radius - min_radius) * (1 - depth_ratio)
        radius = random.uniform(min_radius, max_possible_radius)

        center_color = random.choice(color_scheme)
        edge_color = (
            random.choice([c for c in color_scheme if c != center_color])
            if len(color_scheme) > 1
            else center_color
        )

        blobs.append(
            {
                "center": (x, y),
                "radius": radius,
                "center_color": center_color,
                "edge_color": edge_color,
                "depth_ratio": depth_ratio,
            }
        )

    # 멀리 있는 것부터 먼저 그리기
    blobs.sort(key=lambda b: b["center"][1], reverse=True)

    for blob in blobs:
        x, y = blob["center"]
        radius = blob["radius"]
        depth_ratio = blob["depth_ratio"]

        center_rgb = blob["center_color"]
        edge_rgb = blob["edge_color"]

        # ---- 단순 그림자(타원) ----
        shadow_scale = 1.4
        shadow_y_offset = -0.15 * radius
        shadow_alpha = 0.25 * (1 - depth_ratio)
        shadow = Ellipse(
            (x, y + shadow_y_offset),
            width=radius * shadow_scale,
            height=radius * 0.45,
            angle=0,
            linewidth=0,
            facecolor=(0, 0, 0, shadow_alpha),
            zorder=1,
        )
        ax.add_patch(shadow)

        # ---- 컨투어 모드 ----
        if visualization_mode == "contour_line":
            steps = max(3, int(num_gradient_steps / 4))
            for i in range(steps):
                t = i / (steps - 1)
                r = radius * (1.0 - 0.4 * t)
                wobble = wobble_factor * radius
                theta = np.linspace(0, 2 * np.pi, 400)
                rr = r + wobble * np.sin(theta * 3 + random.random() * 2 * np.pi)
                xx = x + rr * np.cos(theta)
                yy = y + rr * np.sin(theta)
                col = tuple(
                    (1 - t) * center_rgb[j] + t * edge_rgb[j] for j in range(3)
                )
                alpha = 0.8 * (1 - t) ** 1.5
                ax.plot(xx, yy, color=(*col, alpha), linewidth=1.2, zorder=3)
        else:
            # ---- 3D-like 그라데이션 모드 ----
            for i in range(num_gradient_steps):
                t = i / (num_gradient_steps - 1)
                r = radius * (1.0 - 0.7 * t)
                wobble = wobble_factor * radius
                theta = np.linspace(0, 2 * np.pi, 400)
                rr = r + wobble * np.sin(theta * 3 + random.random() * 2 * np.pi)
                xx = x + rr * np.cos(theta)
                yy = y + rr * np.sin(theta)

                col = tuple(
                    (1 - t) * center_rgb[j] + t * edge_rgb[j] for j in range(3)
                )
                # 위쪽/가까운 쪽에서 조금 더 밝게
                light = 1.0 + 0.3 * (1 - depth_ratio) * (1 - t)
                col = adjust_lightness(col, light)
                alpha = max(0.0, 1.0 - t ** 1.8)

                ax.fill(xx, yy, color=(*col, alpha), linewidth=0, zorder=2 + i * 0.001)

        # ---- 하이라이트 ----
        highlight_r = radius * 0.35
        highlight = Circle(
            (x - radius * 0.15, y + radius * 0.2),
            highlight_r,
            facecolor=(1, 1, 1, 0.25 * (1 - depth_ratio)),
            edgecolor="none",
            zorder=4,
        )
        ax.add_patch(highlight)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4 / 3)
    fig.tight_layout()
    return fig


# ==============================
# Streamlit UI
# ==============================

def main():
    st.set_page_config(
        page_title="Generative Poster · Interactive · 3D · CSV",
        layout="centered",
    )

    st.title("Generative Poster · Interactive · 3D · CSV")
    st.write("Colab ipywidgets 버전을 Streamlit 웹 앱으로 옮긴 버전입니다.")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Controls")

        use_random_seed = st.checkbox("Use random seed", value=True)
        seed = None
        if not use_random_seed:
            seed = st.number_input(
                "Seed", min_value=0, max_value=1_000_000, value=0, step=1
            )

        visualization_mode = st.radio(
            "Visualization Mode",
            options=["filled_gradient", "contour_line"],
            format_func=lambda x: "Filled Gradient"
            if x == "filled_gradient"
            else "Contour Line",
        )

        palette_name = st.selectbox(
            "Palette",
            options=["vivid", "pastel", "neonPop", "monochrome", "csv"],
            format_func=lambda x: x.upper() if x != "csv" else "CSV (palette.csv)",
            index=1,
        )

        bgcolor = st.color_picker("Background color", value="#fdf7f2")

        margin = st.slider("Margin", 0.0, 0.3, 0.1, 0.01)
        num_blobs = st.slider("Number of blobs", 3, 12, 6, 1)
        wobble_factor = st.slider("Wobble amount", 0.0, 0.2, 0.05, 0.01)
        horizon_position = st.slider("Horizon position", 0.3, 0.9, 0.6, 0.01)

        st.markdown("CSV 팔레트는 같은 폴더의 `palette.csv`를 사용합니다.")

    with col_left:
        fig = draw_poster(
            seed=seed,
            visualization_mode=visualization_mode,
            palette_name=palette_name,
            bgcolor=bgcolor,
            margin=margin,
            num_blobs=num_blobs,
            wobble_factor=wobble_factor,
            horizon_position=horizon_position,
        )
        st.pyplot(fig)

        if palette_name == "csv":
            st.subheader("Current CSV Palette")
            df = read_palette()
            st.dataframe(df)


if __name__ == "__main__":
    main()
