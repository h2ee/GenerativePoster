
# streamlit_app.py
# Generative Poster: Interactive · 3D · CSV
# (Dong-Hee Kim 버전 Streamlit 포팅)

import os
import random
import math
import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.patches import Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap


# =========================
# 헬퍼 함수들
# =========================

def hex_to_rgb_frac(hexcolor):
    """#RRGGBB → (r,g,b) 0~1"""
    if hexcolor.startswith("#"):
        hexcolor = hexcolor[1:]
    r = int(hexcolor[0:2], 16) / 255.0
    g = int(hexcolor[2:4], 16) / 255.0
    b = int(hexcolor[4:6], 16) / 255.0
    return (r, g, b)


def adjust_lightness(color_rgb, factor):
    """
    RGB 색상의 명도(Lightness)를 조절 (원래 노트북 코드 그대로)
    factor: 1.0 원본, 0.5 더 어둡게, 1.2 더 밝게
    """
    h, l, s = colorsys.rgb_to_hls(*color_rgb)
    new_l = max(0.0, min(1.0, l * factor))
    return colorsys.hls_to_rgb(h, new_l, s)


# =========================
# CSV 팔레트 초기화 & 로딩
# =========================

PALETTE_FILE = "palette.csv"

# 원래 노트북에 있던 파란 팔레트 값들
BLUE_PALETTE = [
    ("royal",       0.067, 0.118, 0.424),
    ("space",       0.114, 0.161, 0.318),
    ("prussian",    0.000, 0.192, 0.322),
    ("navy",        0.000, 0.000, 0.502),
    ("yale",        0.055, 0.302, 0.573),
    ("egyptian",    0.063, 0.204, 0.651),
    ("azure",       0.000, 0.502, 1.000),
    ("sapphire",    0.059, 0.322, 0.729),
    ("olympic",     0.000, 0.557, 0.800),
    ("cornflower",  0.396, 0.576, 0.961),
    ("independence",0.298, 0.318, 0.427),
    ("teal",        0.000, 0.502, 0.506),
    ("maya",        0.451, 0.761, 0.984),
    ("pigeon",      0.447, 0.522, 0.647),
    ("turkishBlue", 0.310, 0.592, 0.639),
    ("carolina",    0.341, 0.627, 0.827),
    ("steel",       0.275, 0.510, 0.706),
    ("tiffany",     0.506, 0.847, 0.808),
    ("babyBlue",    0.537, 0.812, 0.941),
    ("airForce",    0.345, 0.545, 0.682),
    ("electric",    0.494, 0.976, 1.000),
    ("powder",      0.690, 0.875, 0.898),
    ("turquoise",   0.247, 0.878, 0.816),
]

# palette.csv 없으면 한 번만 생성
if not os.path.exists(PALETTE_FILE):
    df_init = pd.DataFrame(
        [{"name": n, "r": r, "g": g, "b": b} for (n, r, g, b) in BLUE_PALETTE]
    )
    df_init.to_csv(PALETTE_FILE, index=False)


def read_palette():
    return pd.read_csv(PALETTE_FILE)


def load_csv_palette():
    df = read_palette()
    return [(row.r, row.g, row.b) for row in df.itertuples()]


# =========================
# 컬러 스킴 정의 (원본과 동일)
# =========================

consistent_color_scheme_hex = [
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
vivid_color_scheme_hex = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#9781bf",
    "#999999",
]
neonPop_color_scheme_hex = [
    "#FEFF00",
    "#39FF14",
    "#00FFFF",
    "#FF00FF",
    "#FF1493",
    "#FF4500",
    "#FFFFF0",
    "#7CFC00",
    "#ADFF2F",
    "#00BFFF",
]
monochrome_color_scheme_hex = [
    "#000000",
    "#333333",
    "#666666",
    "#999999",
    "#CCCCCC",
    "#EEEEEE",
    "#FFFFFF",
]


def get_color_scheme(selection_index: int):
    """
    0: vivid, 1: pastel, 2: neonPop, 3: monochrome, 4: CSV
    → 각 블롭에서 쓰는 색 리스트 반환
    """
    if selection_index == 0:
        return [hex_to_rgb_frac(c) for c in vivid_color_scheme_hex]
    elif selection_index == 1:
        return [hex_to_rgb_frac(c) for c in consistent_color_scheme_hex]
    elif selection_index == 2:
        return [hex_to_rgb_frac(c) for c in neonPop_color_scheme_hex]
    elif selection_index == 3:
        return [hex_to_rgb_frac(c) for c in monochrome_color_scheme_hex]
    elif selection_index == 4:
        return load_csv_palette()
    else:
        return [hex_to_rgb_frac(c) for c in consistent_color_scheme_hex]


# =========================
# 메인 드로잉 함수
# (원래 update_poster()를 파라미터화)
# =========================

def draw_poster(
    color_scheme_selection: int = 4,
    visualization_mode: str = "filled_gradient",
    manual_seed: int | None = None,
    bgcolor: str = "white",
    horizon_position: float = 0.66,
    margin: float = 0.0,
    num_blobs_min: int = 2,
    num_blobs_max: int = 30,
    wobble_factor: float = 0.0,
    min_radius: float = 0.06,
    max_radius: float = 0.09,
    num_gradient_steps: int = 30,
    wobble_density: int = 30,
    contour_outline_thickness: float = 0.25,
    contour_random_steps_min: int = 2,
    contour_random_steps_max: int = 5,
):
    """
    원래 노트북의 update_poster() 내용을 거의 그대로 옮긴 함수.
    Streamlit에서는 모든 UI 값을 인자로 넘겨서 사용.
    """
    # ----------------
    # 0. 랜덤 시드 세팅
    # ----------------
    if manual_seed is not None:
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        seed_msg = f"Using manual seed: {manual_seed}"
    else:
        seed = random.randint(0, 1_000_000)
        random.seed(seed)
        np.random.seed(seed)
        seed_msg = f"Using random seed: {seed}"

    # 빛 방향 / 강도 (원래 코드)
    light_direction = np.array([-0.5, 1.0])
    light_intensity = 0.4
    SHADOW_GRADIENT_STEPS = 15

    # 포스터 비율 3:4
    fig, ax = plt.subplots(figsize=(6, 8))

    # ----------------
    # 1. 배경 그라데이션
    # ----------------
    bgcol = bgcolor
    c1_rgb = np.array(plt.cm.colors.to_rgb(bgcol))
    c1_rgb = [c1_rgb[0] * 0.95, c1_rgb[1] * 0.95, c1_rgb[2] * 0.95]
    c2_rgb = np.array(plt.cm.colors.to_rgb(bgcol))
    mid_rgb = (np.array(c1_rgb) + np.array(c2_rgb)) / 2

    stops_bg = [
        (0.0, c2_rgb),
        (horizon_position - 0.02, c2_rgb),
        (horizon_position, mid_rgb),
        (horizon_position + 0.02, c1_rgb),
        (1.0, c1_rgb),
    ]

    gradient_cmap = LinearSegmentedColormap.from_list(
        "background_gradient", stops_bg
    )

    gradient_image = gradient_cmap(np.linspace(0, 1, 256))[:, np.newaxis, :3]
    ax.imshow(
        gradient_image,
        aspect="auto",
        extent=(0, 1, 0, 4 / 3),
        origin="lower",
        zorder=-1,
    )

    # 축/테두리 제거
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ----------------
    # 2. 팔레트 선택
    # ----------------
    color_scheme = get_color_scheme(color_scheme_selection)

    # ----------------
    # 3. 블롭들 생성
    # ----------------
    if num_blobs_min > num_blobs_max:
        num_blobs_min, num_blobs_max = num_blobs_max, num_blobs_min

    num_blobs = random.randint(num_blobs_min, num_blobs_max)
    blobs_to_draw = []

    for i in range(num_blobs):
        # 중심좌표 (margin 안에서 랜덤)
        x = random.uniform(margin, 1.0 - margin)
        y = random.uniform(margin, 4 / 3 * horizon_position * 1.1)

        depth_ratio = y / (4 / 3)
        max_possible_radius = min_radius + (max_radius - min_radius) * (1 - depth_ratio)
        radius = random.uniform(min_radius, max_possible_radius)

        center_color = random.choice(color_scheme)
        # color_scheme이 CSV면 이미 (r,g,b) 튜플 / hex면 위에서 변환
        center_rgb = center_color

        # edge 색: center와 다른 색
        if len(color_scheme) > 1:
            edge_rgb = random.choice([c for c in color_scheme if c != center_rgb])
        else:
            edge_rgb = center_rgb

        brighter_center_rgb = adjust_lightness(center_rgb, 1.05)
        darker_edge_rgb = adjust_lightness(edge_rgb, 0.7)

        # 공용 그라데이션 정의 (원래 gradient_definition)
        gradient_definition = [
            (0.0, brighter_center_rgb, 1.0),
            (0.05, center_rgb, 1.0),
            (0.4, edge_rgb, 1.0),
            (0.75, darker_edge_rgb, 1.0),
            (0.9, edge_rgb, 1.0),
            (1.0, edge_rgb, 0.8),
        ]

        colors_and_alpha = []
        for pos, color_tuple, alpha in gradient_definition:
            rgba = color_tuple + (alpha,)
            colors_and_alpha.append((pos, rgba))

        # 그림자용 stop (원래 shadow_stops)
        shadow_stops = [
            {"pos": 0, "alpha": 2},
            {"pos": 80, "alpha": 1.2},
            {"pos": 100, "alpha": 0},
        ]
        shadow_rgba = []
        for s in shadow_stops:
            p = s["pos"] / 100.0
            a = s["alpha"] / 100.0
            shadow_rgba.append((p, (0, 0, 0, a)))

        shadow_cmap = LinearSegmentedColormap.from_list(
            f"custom_radial_gradient_shadow_{i}", shadow_rgba, N=256
        )

        # 뾰족한 윤곽 (wobble)
        theta = np.linspace(0, 2 * np.pi, wobble_density, endpoint=False)
        r_base = radius
        r_wobbled = r_base + np.random.uniform(
            -wobble_factor * r_base,
            wobble_factor * r_base,
            size=theta.shape,
        )

        x_contour = x + r_wobbled * np.cos(theta)
        y_contour = y + r_wobbled * np.sin(theta)
        centroid_x = np.mean(x_contour)
        centroid_y = np.mean(y_contour)

        sd_y_contour = y + 0.3 * r_wobbled * np.sin(theta)

        blobs_to_draw.append(
            {
                "center": (x, y),
                "centroid": (centroid_x, centroid_y),
                "radius": radius,
                "colors_and_alpha": colors_and_alpha,
                "shadow_cmap": shadow_cmap,
                "x_contour": x_contour,
                "y_contour": y_contour,
                "sd_y_contour": sd_y_contour,
            }
        )

    # y 큰(먼) 순으로 sort → 멀리 있는 것 먼저 그림
    blobs_to_draw.sort(key=lambda b: b["center"][1], reverse=True)
    bg_rgb = plt.cm.colors.colorConverter.to_rgb(bgcolor)

    # 각 블롭에 대해 깊이 반영해서 cmap 만들기
    for blob_data in blobs_to_draw:
        y = blob_data["center"][1]
        depth_ratio = y / (4 / 3)
        effect_strength = depth_ratio**2
        original = blob_data["colors_and_alpha"]

        adjusted = []
        for pos, (r, g, b, a) in original:
            new_r = r * (1 - effect_strength) + bg_rgb[0] * effect_strength
            new_g = g * (1 - effect_strength) + bg_rgb[1] * effect_strength
            new_b = b * (1 - effect_strength) + bg_rgb[2] * effect_strength
            new_a = a * (1 - effect_strength)
            adjusted.append((pos, (new_r, new_g, new_b, new_a)))

        cmap = LinearSegmentedColormap.from_list(
            f"custom_cmap_{id(blob_data)}", adjusted
        )
        blob_data["cmap"] = cmap

    # ----------------
    # 4. 그림자 먼저
    # ----------------
    for blob_data in blobs_to_draw:
        y = blob_data["center"][1]
        depth_ratio = y / (4 / 3)
        effect_strength = depth_ratio**2
        shadow_alpha_multiplier = 1 - effect_strength
        shadow_offset_multiplier = 0.5 + effect_strength

        radius = blob_data["radius"]
        centroid = blob_data["centroid"]
        x_contour = blob_data["x_contour"]
        sd_y_contour = blob_data["sd_y_contour"]
        shadow_cmap = blob_data["shadow_cmap"]

        offset_x = 0.0
        offset_y = -2.0 * radius * shadow_offset_multiplier
        sd_centroid = (centroid[0] + offset_x, centroid[1] + offset_y)
        sd_x_contour = x_contour + offset_x
        sd_y_contour = sd_y_contour + offset_y

        base_zorder = (4 / 3) - y

        if visualization_mode == "filled_gradient":
            max_shadow_dist = np.max(
                np.sqrt(
                    (sd_x_contour - sd_centroid[0]) ** 2
                    + (sd_y_contour - sd_centroid[1]) ** 2
                )
            )

            for j in range(SHADOW_GRADIENT_STEPS, 0, -1):
                scale_factor = j / SHADOW_GRADIENT_STEPS
                x_scaled = sd_centroid[0] + (sd_x_contour - sd_centroid[0]) * scale_factor * 1.1
                y_scaled = sd_centroid[1] + (sd_y_contour - sd_centroid[1]) * scale_factor * 1.1

                distances = np.sqrt(
                    (x_scaled - sd_centroid[0]) ** 2
                    + (y_scaled - sd_centroid[1]) ** 2
                )
                if max_shadow_dist > 0:
                    normalized = distances / max_shadow_dist
                else:
                    normalized = np.zeros_like(distances)

                color = shadow_cmap(np.mean(normalized))
                color = (color[0], color[1], color[2], color[3] * shadow_alpha_multiplier)

                shadow_patch = Polygon(
                    list(zip(x_scaled, y_scaled)),
                    closed=True,
                    facecolor=color,
                    linewidth=0,
                    zorder=base_zorder,
                )
                ax.add_patch(shadow_patch)

    # ----------------
    # 5. 블롭 본체
    # ----------------
    for blob_data in blobs_to_draw:
        centroid = blob_data["centroid"]
        cmap = blob_data["cmap"]
        x_contour = blob_data["x_contour"]
        y_contour = blob_data["y_contour"]
        radius = blob_data["radius"]
        y = blob_data["center"][1]

        depth_ratio = y / (4 / 3)
        base_zorder = (4 / 3) - y

        # 하이라이트 방향
        offset_x = light_direction[0] * radius * light_intensity
        offset_y = light_direction[1] * radius * light_intensity
        gradient_center = (centroid[0] + offset_x, centroid[1] + offset_y)

        if visualization_mode == "filled_gradient":
            grid_resolution = 100
            x_min, x_max = np.min(x_contour), np.max(x_contour)
            y_min, y_max = np.min(y_contour), np.max(y_contour)

            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= x_range * 0.05
            x_max += x_range * 0.05
            y_min -= y_range * 0.05
            y_max += y_range * 0.05

            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, grid_resolution),
                np.linspace(y_min, y_max, grid_resolution),
            )

            distances = np.sqrt(
                (grid_x - gradient_center[0]) ** 2
                + (grid_y - gradient_center[1]) ** 2
            )
            max_dist = np.max(
                np.sqrt(
                    (x_contour - gradient_center[0]) ** 2
                    + (y_contour - gradient_center[1]) ** 2
                )
            )
            if max_dist > 0:
                normalized = distances / max_dist
            else:
                normalized = np.zeros_like(distances)

            mesh = ax.pcolormesh(
                grid_x,
                grid_y,
                normalized,
                cmap=cmap,
                shading="gouraud",
                zorder=base_zorder + 0.01,
            )

            blob_path_polygon = Polygon(
                list(zip(x_contour, y_contour)),
                closed=True,
                fc="none",
                ec="none",
                lw=0,
            )
            ax.add_patch(blob_path_polygon)
            mesh.set_clip_path(blob_path_polygon)

        elif visualization_mode == "contour_line":
            random_steps = random.randint(
                contour_random_steps_min, contour_random_steps_max
            )
            line_color = (0, 0, 0, 1)

            for j in range(random_steps, 0, -1):
                scale_factor = j / random_steps
                x_scaled = centroid[0] + (x_contour - centroid[0]) * scale_factor
                y_scaled = centroid[1] + (y_contour - centroid[1]) * scale_factor

                blob_outline = Polygon(
                    list(zip(x_scaled, y_scaled)),
                    closed=True,
                    facecolor="none",
                    edgecolor=line_color,
                    linewidth=contour_outline_thickness * scale_factor,
                    zorder=np.mean(y_contour),
                )
                ax.add_patch(blob_outline)

    # 포스터 텍스트
    ax.text(
        0.05,
        4 / 3 - 0.05,
        "Generative Poster",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        fontfamily="sans-serif",
    )
    ax.text(
        0.05,
        4 / 3 - 0.1,
        "Interactive · 3D · CSV",
        fontsize=8,
        fontweight="light",
        ha="left",
        va="top",
        fontfamily="sans-serif",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4 / 3)
    ax.set_box_aspect(4 / 3)

    fig.tight_layout()

    # 저장도 그대로 유지 (원래 코드)
    plt.savefig("poster.png", dpi=300, bbox_inches="tight", pad_inches=0)

    return fig, seed_msg


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="Generative Poster · Interactive · 3D · CSV",
        layout="wide",
    )

    st.title("Generative Poster · Interactive · 3D · CSV")
    st.write(
        "그라데이션 사각형 + 원형 마스크로 만든 3D-like 블롭 포스터를 "
        "Streamlit 앱으로 옮긴 버전입니다."
    )

    col_canvas, col_controls = st.columns([2, 1])

    with col_controls:
        st.subheader("Controls")

        # Palette / Mode
        palette_label_to_value = {
            "vivid": 0,
            "pastel": 1,
            "neonPop": 2,
            "monochrome": 3,
            "CSV (palette.csv)": 4,
        }
        palette_label = st.selectbox(
            "Palette",
            list(palette_label_to_value.keys()),
            index=4,
        )
        color_scheme_selection = palette_label_to_value[palette_label]

        vis_mode_label = st.radio(
            "Visualization Mode",
            ["Filled Gradient", "Contour Line"],
            index=0,
        )
        visualization_mode = (
            "filled_gradient" if vis_mode_label == "Filled Gradient" else "contour_line"
        )

        # Seed
        use_random = st.checkbox("Use random seed", value=True)
        manual_seed = None
        if not use_random:
            manual_seed = st.number_input(
                "Manual Seed", min_value=0, max_value=1_000_000, value=0, step=1
            )

        # Background & horizon
        bgcolor = st.color_picker("Background Color", value="#ffffff")
        horizon_position = st.slider(
            "Horizon Position (0 = bottom, 1 = top)", 0.2, 0.8, 0.66, 0.01
        )

        margin = st.slider("Margin", 0.0, 0.5, 0.0, 0.01)

        # Blob count
        num_blobs_min = st.slider("Min Blobs", 1, 100, 2, 1)
        num_blobs_max = st.slider("Max Blobs", 1, 100, 30, 1)

        st.markdown("---")
        st.markdown("**Shape / Gradient controls**")

        wobble_factor = st.slider("Wobble Factor", 0.0, 5.0, 0.0, 0.01)
        min_radius = st.slider("Min Radius", 0.01, 0.5, 0.06, 0.01)
        max_radius = st.slider("Max Radius", 0.01, 0.5, 0.09, 0.01)
        num_gradient_steps = st.slider("Gradient Steps", 1, 100, 30, 1)
        wobble_density = st.slider("Wobble Density", 3, 800, 30, 1)

        st.markdown("---")
        st.markdown("**Contour mode controls**")
        contour_outline_thickness = st.slider(
            "Stroke Weight", 0.01, 1.0, 0.25, 0.01
        )
        contour_random_steps_min = st.slider(
            "Min Contour Steps", 1, 10, 2, 1
        )
        contour_random_steps_max = st.slider(
            "Max Contour Steps", 1, 10, 5, 1
        )

    # 그림 그리기
    with col_canvas:
        fig, seed_msg = draw_poster(
            color_scheme_selection=color_scheme_selection,
            visualization_mode=visualization_mode,
            manual_seed=manual_seed if not use_random else None,
            bgcolor=bgcolor,
            horizon_position=horizon_position,
            margin=margin,
            num_blobs_min=num_blobs_min,
            num_blobs_max=num_blobs_max,
            wobble_factor=wobble_factor,
            min_radius=min_radius,
            max_radius=max_radius,
            num_gradient_steps=num_gradient_steps,
            wobble_density=wobble_density,
            contour_outline_thickness=contour_outline_thickness,
            contour_random_steps_min=contour_random_steps_min,
            contour_random_steps_max=contour_random_steps_max,
        )
        st.pyplot(fig)
        st.caption(seed_msg)

    # CSV 팔레트 확인용
    if color_scheme_selection == 4:
        st.subheader("Current CSV Palette (palette.csv)")
        df = read_palette()
        st.dataframe(df)


if __name__ == "__main__":
    main()
