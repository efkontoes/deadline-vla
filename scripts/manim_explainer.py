"""Method explainer for the README (Manim Community Edition).

    pip install -e .[viz]
    manim -ql --format gif -r 640,360 --fps 12 scripts/manim_explainer.py DeadlineExplainer

Every allocation, latency and speed shown is computed live by the project code: the real allocator
(``budget/allocator.py::allocate_for_speed``, fixed floor, speed objective), the latency table of the light
run, the governor (``budget/governor.py::safe_speed``) and the closing-speed weights for a threat
(``budget/relative.py``). Nothing in the bars or numbers is drawn by hand.

What it shows, faithfully to v4: the allocation does not grow or shrink with speed. While moving it is the
cheapest allocation that still sees enough in the direction of travel (sharp ahead, low elsewhere); the
deadline acts through the governor, which turns that allocation's latency into the speed the drone may fly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Circle,
    FadeIn,
    FadeOut,
    MathTex,
    Polygon,
    Rectangle,
    Scene,
    Sector,
    Text,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    config,
)

from deadlinevla.budget.allocator import allocate_for_speed, camera_axes, quality_table, view_weights
from deadlinevla.budget.governor import safe_speed
from deadlinevla.budget.latency_model import LatencyModel, LatencyTable, synthetic_prior
from deadlinevla.budget.relative import Contacts, camera_closing, closing_weights

config.background_color = "#fcfcfb"
INK, INK2, BLUE, RED, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#e34948", "#eb6834"
TIER = {0: "#e4e3df", 1: "#c6dbef", 2: "#6baed6", 3: "#2171b5"}
TOKENS = (0, 16, 64, 256)
YAWS = (0, 60, 120, 180, 240, 300)
NAMES = ("F", "FL", "BL", "B", "BR", "FR")
A, M = 8.4, 0.5           # measured braking deceleration and margin
EXTRA = 0.19              # control period + measured tracking lag added to the governor's blind time (s)

ROOT = Path(__file__).resolve().parents[1]
_table = ROOT / "runs" / "light_surrogate" / "latency_table.npz"
LAT = (LatencyModel(LatencyTable.load(_table), "p95", 0.005, TOKENS) if _table.exists()
       else synthetic_prior("synthetic_orin_nx", TOKENS))
Q = quality_table("log1p_tokens", TOKENS)
AXES = camera_axes(YAWS)
ALL_HIGH_LAT = LAT.predict([3] * 6)


def allocation(speed: float, heading_deg: float, threat_dist: float | None) -> tuple[np.ndarray, float]:
    v = speed * np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg)), 0.0])
    if threat_dist is not None:
        threat = Contacts(p=np.array([[-threat_dist, 0.0]]), R=np.array([0.75]), w=np.array([[6.0, 0.0]]))
        w = closing_weights(camera_closing(threat, v, AXES), 2.0, 0.05, 0.3)
    else:
        w = view_weights(v, AXES, 2.0, 0.05, 0.3)
    return allocate_for_speed(w, LAT, Q, 0.25)


class DeadlineExplainer(Scene):
    def construct(self):
        title = Text("DeadlineVLA", font_size=36, color=INK, weight="BOLD").to_edge(UP, buff=0.25)
        sub = Text("six cameras, a token budget, and a speed governor", font_size=20, color=INK2)
        sub.next_to(title, DOWN, buff=0.1)
        self.play(Write(title), FadeIn(sub), run_time=1)

        speed = ValueTracker(0.0)
        heading = ValueTracker(0.0)
        dist = ValueTracker(12.0)
        threat_on = ValueTracker(0.0)
        threat_d = ValueTracker(10.0)
        drone_pos = LEFT * 3.4 + DOWN * 0.35
        scale = 0.3

        def current():
            td = threat_d.get_value() if threat_on.get_value() > 0.5 else None
            return allocation(speed.get_value(), heading.get_value(), td)

        def wedges():
            alloc, _ = current()
            g = VGroup()
            for yaw, tier in zip(YAWS, alloc):
                g.add(Sector(radius=0.5 + 0.42 * int(tier), angle=math.radians(60),
                             start_angle=math.radians(yaw - 30), arc_center=drone_pos,
                             fill_color=TIER[int(tier)], fill_opacity=0.9, stroke_width=1, stroke_color="#fcfcfb"))
            return g

        def velocity_arrow():
            s = speed.get_value()
            if s < 0.3:
                return VGroup()
            h = math.radians(heading.get_value())
            tip = drone_pos + 0.32 * s * np.array([math.cos(h), math.sin(h), 0.0])
            return Arrow(drone_pos, tip, buff=0, color=ORANGE, stroke_width=5, max_tip_length_to_length_ratio=0.25)

        drone = Polygon(RIGHT * 0.3, LEFT * 0.18 + UP * 0.18, LEFT * 0.18 + DOWN * 0.18, color=INK,
                        fill_opacity=1).move_to(drone_pos)
        cams, arrow = always_redraw(wedges), always_redraw(velocity_arrow)

        def bars():
            alloc, lat = current()
            g = VGroup()
            for i, (tier, nm) in enumerate(zip(alloc, NAMES)):
                h = 0.1 + 0.42 * int(tier)          # height by tier; the token total is printed above
                b = Rectangle(width=0.36, height=h, fill_color=TIER[int(tier)], fill_opacity=1, stroke_width=0)
                b.move_to(RIGHT * (1.45 + i * 0.52) + UP * 0.05, aligned_edge=DOWN)
                g.add(b, Text(nm, font_size=13, color=INK2).next_to(b, DOWN, buff=0.05))
            total = int(sum(TOKENS[int(t)] for t in alloc))
            g.add(Text(f"{total} visual tokens per tick  ·  {lat * 1000:.0f} ms", font_size=17, color=INK)
                  .move_to(RIGHT * 2.75 + UP * 1.55))
            return g

        token_bars = always_redraw(bars)
        caption = Text("hovering: every camera at mid resolution", font_size=18, color=INK2).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(cams), FadeIn(drone), FadeIn(token_bars), FadeIn(caption), run_time=1)
        self.wait(1)

        # 1. Moving: the pattern follows the direction of travel; the total stays low.
        new_cap = Text("moving: the cheapest allocation that still sees enough — sharp toward travel, low elsewhere",
                       font_size=16, color=INK2).to_edge(DOWN, buff=0.2)
        self.add(arrow)
        self.play(speed.animate.set_value(4.0), FadeOut(caption), FadeIn(new_cap), run_time=1.2)
        self.play(heading.animate.set_value(180.0), run_time=4)
        self.play(heading.animate.set_value(0.0), run_time=1.5)
        self.wait(0.5)

        # 2. The governor: latency becomes safe speed.
        obstacle = always_redraw(lambda: Rectangle(width=0.2, height=1.5, color=INK2, fill_opacity=1).move_to(
            drone_pos + RIGHT * min(dist.get_value(), 13) * scale))
        eq = MathTex(r"v^* = -aL + \sqrt{a^2L^2 + 2a(d-m)}", font_size=28, color=INK)
        eq.move_to(RIGHT * 2.75 + DOWN * 1.15)

        def gov_bars():
            d = dist.get_value()
            _, lat = current()
            ours = safe_speed(lat + EXTRA, d, A, M)
            hi = safe_speed(ALL_HIGH_LAT + EXTRA, d, A, M)
            g = VGroup()
            for row, (label, val, col) in enumerate(((f"ours ({lat * 1000:.0f} ms)", ours, BLUE),
                                                    (f"all-high ({ALL_HIGH_LAT * 1000:.0f} ms)", hi, ORANGE))):
                y = DOWN * (1.95 + 0.42 * row)
                bar = Rectangle(width=max(0.02, 0.22 * val), height=0.26, fill_color=col, fill_opacity=1,
                                stroke_width=0).move_to(RIGHT * 1.6 + y, aligned_edge=LEFT)
                g.add(Text(label, font_size=14, color=INK).move_to(RIGHT * 0.55 + y),
                      bar, Text(f"{val:.1f} m/s", font_size=14, color=INK).next_to(bar, RIGHT, buff=0.08))
            g.add(Text(f"obstacle ahead: d = {d:.1f} m", font_size=15, color=INK2).move_to(RIGHT * 2.75 + DOWN * 1.6))
            return g

        govs = always_redraw(gov_bars)
        cap2 = Text("fewer tokens → lower latency → higher safe speed (the governor's v*)", font_size=16,
                    color=INK2).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(obstacle), Write(eq), FadeIn(govs), FadeOut(new_cap), FadeIn(cap2), run_time=1.2)
        self.play(dist.animate.set_value(2.0), run_time=4)
        self.wait(0.8)

        # 3. A threat from behind: closing speed, not only own motion.
        threat = always_redraw(lambda: Circle(radius=0.17, color=RED, fill_opacity=1, stroke_width=0).move_to(
            drone_pos + LEFT * threat_d.get_value() * scale))
        cap3 = Text("something closing from behind raises the rear camera's tier", font_size=16,
                    color=RED).to_edge(DOWN, buff=0.2)
        self.play(FadeOut(obstacle), FadeOut(eq), FadeOut(govs), FadeOut(cap2), FadeIn(cap3),
                  speed.animate.set_value(3.0), run_time=1)
        threat_on.set_value(1.0)
        self.play(FadeIn(threat), run_time=0.5)
        self.play(threat_d.animate.set_value(3.0), run_time=3)
        self.wait(1)

        end = VGroup(
            Text("In simulation: 36% fewer visual tokens than uniform,", font_size=22, color=INK),
            Text("with the same success and safety.", font_size=22, color=INK),
            Text("synthetic Jetson latency · surrogate perception · not yet a real VLA", font_size=15, color=INK2),
        ).arrange(DOWN, buff=0.14).move_to(ORIGIN + DOWN * 0.2)
        self.play(*[FadeOut(m) for m in list(self.mobjects) if m is not title], run_time=0.8)
        self.play(FadeIn(end), run_time=0.8)
        self.wait(2.5)
