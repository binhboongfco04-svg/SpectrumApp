# ui_components.py
import os
import math
import numpy as np

from kivy.metrics import dp, sp
from kivy.uix.modalview import ModalView
from kivy.uix.filechooser import FileChooserIconView
from kivy.clock import Clock
from kivy.core.window import Window
from kivy_garden.graph import Graph, LinePlot

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel, MDIcon
from kivymd.uix.button import MDRaisedButton, MDFlatButton, MDRectangleFlatIconButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.widget import MDWidget

from kivy.uix.image import Image
from kivy.uix.anchorlayout import AnchorLayout

import sys
import threading
import processing

ANDROID = False
Chooser = None
SharedStorage = None

if 'ANDROID_ARGUMENT' in os.environ:
    print("DEBUG: Running on Android (detected via ANDROID_ARGUMENT)")
    try:
        from androidstorage4kivy import SharedStorage, Chooser
        ANDROID = True
        print("DEBUG: androidstorage4kivy OK")
    except Exception as e:
        print(f"DEBUG: androidstorage4kivy FAILED: {e}")
        ANDROID = False
else:
    print("DEBUG: Running on Desktop")

PRIMARY_COLOR = (0.098, 0.608, 0.596, 1)

HEX_COLORS_RGBA = [
    (0.122, 0.467, 0.706, 1),
    (1.0,   0.498, 0.055, 1),
    (0.173, 0.627, 0.173, 1),
    (0.839, 0.153, 0.157, 1),
    (0.580, 0.404, 0.741, 1),
    (0.549, 0.337, 0.294, 1),
]


def rh(pct):
    return Window.height * pct / 100

def rw(pct):
    return Window.width * pct / 100


# ── File chooser ──
def open_file_chooser(is_multi, callback):
    print(f"DEBUG: platform={sys.platform}, ANDROID={ANDROID}, Chooser={Chooser}")
    if ANDROID:
        _open_android_gallery(is_multi, callback)
    else:
        _open_desktop_chooser(is_multi, callback)

def _open_android_gallery(is_multi, callback):
    chooser = Chooser(_android_chooser_callback)
    _open_android_gallery._pending_callback = callback
    _open_android_gallery._is_multi = is_multi
    chooser.choose_content("image/*", multiple=is_multi)

def _android_chooser_callback(uri_list):
    from androidstorage4kivy import SharedStorage
    callback = getattr(_open_android_gallery, '_pending_callback', None)
    if not callback or not uri_list:
        return
    ss = SharedStorage()
    local_paths = [p for uri in uri_list if (p := ss.copy_from_shared(uri))]
    if local_paths:
        Clock.schedule_once(lambda dt: callback(local_paths), 0)

def _open_desktop_chooser(is_multi, callback):
    content = MDBoxLayout(orientation="vertical", padding=dp(10), spacing=dp(10))
    chooser = FileChooserIconView(
        filters=["*.jpg", "*.png", "*.bmp"],
        multiselect=is_multi,
        path=os.getcwd()
    )
    btn_box = MDBoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
    btn_cancel = MDFlatButton(text="HỦY")
    btn_ok = MDRaisedButton(text="CHỌN")
    btn_box.add_widget(btn_cancel)
    btn_box.add_widget(btn_ok)
    content.add_widget(chooser)
    content.add_widget(btn_box)
    popup = ModalView(size_hint=(0.9, 0.9), auto_dismiss=False)
    popup.add_widget(content)

    def on_ok(instance):
        if chooser.selection:
            callback(chooser.selection)
        popup.dismiss()

    btn_ok.bind(on_release=on_ok)
    btn_cancel.bind(on_release=lambda x: popup.dismiss())
    popup.open()


# ── Nút reset ──
def _make_reset_btn(callback):
    return MDRectangleFlatIconButton(
        text="Xóa dữ liệu",
        icon="delete-outline",
        theme_text_color="Custom",
        text_color=(0, 0, 0, 1),
        icon_color=(0, 0, 0, 1),
        line_color=(0, 0, 0, 0),
        on_release=callback,
    )

# ── Empty state card content ──
def _make_empty_card_content(reset_callback, empty_text):
    card_content = MDBoxLayout(orientation="vertical", spacing=dp(6))
    top_bar = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(32))
    top_bar.add_widget(MDWidget())
    top_bar.add_widget(_make_reset_btn(reset_callback))
    card_content.add_widget(top_bar)
    empty_box = MDBoxLayout(orientation="vertical", spacing=dp(8), size_hint_y=1)
    empty_box.add_widget(Image(
        source="chart.png", size_hint=(1, 1),
        allow_stretch=True, keep_ratio=True,
    ))
    empty_box.add_widget(MDLabel(
        text=empty_text, halign="center", font_style="Body1",
        size_hint_y=None, height=dp(48),
    ))
    card_content.add_widget(empty_box)
    return card_content


# ── Thẻ chọn ảnh ──
def _make_image_card(icon, title_label_ref, status_label_ref, on_release):
    anchor = AnchorLayout(anchor_x="left", anchor_y="center", size_hint=(1, 1))
    inner = MDBoxLayout(
        orientation="vertical", spacing=dp(4),
        size_hint=(1, None), adaptive_height=True,
    )
    icon_row = MDBoxLayout(
        orientation="horizontal", spacing=dp(8),
        size_hint=(1, None), height=dp(28),
    )
    icon_row.add_widget(MDIcon(
        icon=icon, theme_text_color="Custom", text_color=PRIMARY_COLOR,
        font_size="22sp", size_hint=(None, None), size=(dp(26), dp(26)),
    ))
    title_label_ref.valign = "center"
    title_label_ref.size_hint_y = None
    title_label_ref.height = dp(28)
    icon_row.add_widget(title_label_ref)
    status_label_ref.size_hint_y = None
    status_label_ref.height = dp(16)
    inner.add_widget(icon_row)
    inner.add_widget(status_label_ref)
    anchor.add_widget(inner)
    card = MDCard(
        radius=[dp(12)] * 4, line_color=PRIMARY_COLOR, line_width=1.5,
        md_bg_color=(1, 1, 1, 1), padding=[dp(12), dp(6), dp(12), dp(6)],
        ripple_behavior=True, on_release=on_release, elevation=0, size_hint=(1, 1),
    )
    card.add_widget(anchor)
    return card


# ── Nút hành động chính ──
def _make_run_btn(text, on_release):
    btn = MDCard(
        elevation=0, radius=[dp(24)] * 4, md_bg_color=PRIMARY_COLOR,
        size_hint=(1, 1), padding=[dp(12), dp(8)],
        ripple_behavior=True, on_release=on_release,
    )
    content = MDBoxLayout(
        orientation="horizontal", spacing=dp(8), size_hint=(1, 1),
        pos_hint={"center_x": 0.5, "center_y": 0.5},
    )
    content.add_widget(MDIcon(
        icon="chart-line", theme_text_color="Custom", text_color=(1, 1, 1, 1),
        font_size="20sp", size_hint=(None, None), size=(dp(24), dp(24)),
        pos_hint={"center_y": 0.5},
    ))
    content.add_widget(MDLabel(
        text=text, theme_text_color="Custom", text_color=(1, 1, 1, 1),
        font_style="Button", halign="center", valign="center", size_hint_x=1,
    ))
    btn.add_widget(content)
    return btn



# ────────────────────────────────────────────────
#          SHARED GRAPH METHODS (Mixin)
# ────────────────────────────────────────────────

class GraphMixin:
    """Các method đồ thị dùng chung cho cả 2 màn hình."""

    def _build_graph(self, results, size_hint=(1, 1)):
        if not results:
            raise ValueError("Không có dữ liệu phổ để vẽ.")

        y_key = "absorbance" if "absorbance" in results[0] else "intensity"
        y_label = "Độ hấp thụ (a.u.)" if y_key == "absorbance" else "Cường độ (a.u.)"

        all_x = []
        all_y = []

        for r in results:
            xs = np.asarray(r.get("wavelengths", []), dtype=np.float64)
            ys = np.asarray(r.get(y_key, []), dtype=np.float64)

            finite_mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[finite_mask]
            ys = ys[finite_mask]
            wl_mask = (xs >= 460) & (xs <= 620)
            xs = xs[wl_mask]
            ys = ys[wl_mask]

            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())

        if not all_x or not all_y:
            raise ValueError(
                "Không có dữ liệu phổ hợp lệ để vẽ. "
                "Hãy kiểm tra lại ảnh đầu vào, calibration pixel→wavelength "
                "hoặc dải bước sóng wl_min/wl_max."
            )

        x_raw_min, x_raw_max = min(all_x), max(all_x)
        x_tick = 20
        x_min = float(math.floor(x_raw_min / x_tick) * x_tick)
        x_max = float(math.ceil(x_raw_max  / x_tick) * x_tick)

        y_min = min(0.0, min(all_y))
        y_max = max(all_y) * 1.15 if max(all_y) != 0 else 1.0

        if y_max <= y_min:
            y_max = y_min + 1.0

        y_range = y_max - y_min
        magnitude = math.floor(math.log10(y_range)) if y_range > 0 else 0
        y_tick = round(math.ceil((y_range / 5) / (10 ** magnitude)) * (10 ** magnitude), 10)
        if y_tick == 0:
            y_tick = y_range / 5 if y_range > 0 else 1.0

        graph = Graph(
            xlabel="Bước sóng (nm)", ylabel=y_label,
            x_ticks_minor=2, x_ticks_major=x_tick,
            y_ticks_minor=2, y_ticks_major=float(y_tick),
            y_grid_label=True, x_grid_label=True,
            padding=dp(8), x_grid=False, y_grid=False,
            xmin=x_min, xmax=x_max,
            ymin=float(y_min), ymax=float(y_max),
            label_options={"color": (0, 0, 0, 1), "bold": False},
            background_color=(1, 1, 1, 1),
            border_color=(0.8, 0.8, 0.8, 1),
            tick_color=(0.5, 0.5, 0.5, 1),
            size_hint=size_hint,
        )

        for i, r in enumerate(results):
            xs = np.asarray(r.get("wavelengths", []), dtype=np.float64)
            ys = np.asarray(r.get(y_key, []), dtype=np.float64)

            finite_mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[finite_mask]
            ys = ys[finite_mask]

            wl_mask = (xs >= 460) & (xs <= 620)
            xs = xs[wl_mask]
            ys = ys[wl_mask]

            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())
            if len(xs) == 0:
                continue

            plot = LinePlot(color=HEX_COLORS_RGBA[i % 6], line_width=2)
            plot.points = [(float(x), float(y)) for x, y in zip(xs, ys)]
            graph.add_plot(plot)

        return graph

    def _build_legend(self, results):
        legend_box = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(22),
            spacing=dp(8), padding=[dp(4), 0, dp(4), 0],
        )
        for i, r in enumerate(results):
            item = MDBoxLayout(orientation="horizontal", spacing=dp(4),
                               size_hint_x=None, width=dp(70))
            color_box = MDCard(
                size_hint=(None, None), size=(dp(10), dp(10)),
                radius=[2] * 4, md_bg_color=HEX_COLORS_RGBA[i % 6], elevation=0,
            )
            item.add_widget(color_box)
            item.add_widget(MDLabel(
                text=f"Kênh {r['channel']}", font_style="Caption",
                size_hint_x=None, width=dp(54),
            ))
            legend_box.add_widget(item)
        return legend_box

    def _make_chart_card_content(self, results, title_text):
        card_content = MDBoxLayout(orientation="vertical", spacing=dp(4))

        top_bar = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(32), spacing=dp(4),
        )
        top_bar.add_widget(MDLabel(
            text=title_text, font_style="Subtitle2", bold=True, size_hint_x=1,
        ))
        top_bar.add_widget(MDRectangleFlatIconButton(
            text="Phóng to", icon="fullscreen",
            theme_text_color="Custom", text_color=(0, 0, 0, 1),
            icon_color=(0, 0, 0, 1), line_color=(0, 0, 0, 0),
            on_release=lambda x: self._open_fullscreen(results),
        ))
        top_bar.add_widget(_make_reset_btn(self.reset_data))
        card_content.add_widget(top_bar)

        graph = self._build_graph(results, size_hint=(1, 1))
        card_content.add_widget(graph)
        card_content.add_widget(self._build_legend(results))
        return card_content

    def _open_fullscreen(self, results):
        modal = ModalView(size_hint=(1, 1), auto_dismiss=True, background_color=(1, 1, 1, 1))
        content = MDBoxLayout(
            orientation="vertical", padding=dp(12), spacing=dp(8),
            md_bg_color=(1, 1, 1, 1),
        )
        header = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(36))
        y_key = "absorbance" if "absorbance" in results[0] else "intensity"
        title_text = "Phổ Hấp Thụ" if y_key == "absorbance" else "Phổ Huỳnh Quang"
        header.add_widget(MDLabel(text=title_text, font_style="H6", bold=True, size_hint_x=1))
        header.add_widget(MDRectangleFlatIconButton(
            text="Đóng", icon="fullscreen-exit",
            theme_text_color="Custom", text_color=(0, 0, 0, 1),
            icon_color=(0, 0, 0, 1), line_color=(0, 0, 0, 0),
            on_release=lambda x: modal.dismiss(),
        ))
        content.add_widget(header)
        content.add_widget(self._build_graph(results, size_hint=(1, 1)))
        content.add_widget(self._build_legend(results))
        modal.add_widget(content)
        modal.open()

    def show_alert(self, title, text):
        self.dialog = MDDialog(
            title=title, text=text,
            buttons=[MDFlatButton(text="ĐÓNG", on_release=lambda x: self.dialog.dismiss())]
        )
        self.dialog.open()


# ────────────────────────────────────────────────
#          MÀN HÌNH HIỆU CHỈNH THIẾT BỊ
# ────────────────────────────────────────────────

class CalibrationScreen(GraphMixin, MDScreen):
    """
    Màn hình hiệu chỉnh thiết bị — chạy 1 lần khi lắp đặt
    hoặc khi người dùng muốn hiệu chỉnh lại.

    Flow:
        1. Chọn một hoặc nhiều ảnh Reference (đèn LED trắng, không mẫu) — trung bình theo pixel
        2. (Tùy chọn) Chọn một hoặc nhiều ảnh Dark — trung bình luminance rồi trừ khỏi reference
        3. Nhấn "HIỆU CHỈNH THIẾT BỊ" → gọi processing.run_calibration()
           trong thread riêng để không block UI
        4. Kết quả lưu vào calibration.json
           → các màn hình đo tự động dùng từ lần sau
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_paths: list = []
        self.dark_paths: list = []

        layout = MDBoxLayout(
            orientation="vertical", padding=dp(10), spacing=dp(8), size_hint=(1, 1),
        )

        # ── Card hướng dẫn ──
        self.status_card = MDCard(
            padding=dp(16), elevation=2, radius=[dp(14)] * 4,
            size_hint=(1, 0.55), md_bg_color=(1, 1, 1, 1),
        )
        self._refresh_status_card()
        layout.add_widget(self.status_card)

        # ── Hàng dưới: chọn ảnh + nút hiệu chỉnh ──
        bottom_row = MDBoxLayout(
            orientation="horizontal", size_hint=(1, 0.2), spacing=dp(8),
        )

        # Thẻ Dark
        self.lbl_dark_status = MDLabel(
            text="Chưa chọn (tùy chọn)", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_dark = MDLabel(
            text="Chọn ảnh Dark", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_dark = _make_image_card(
            "moon-waning-crescent", self.btn_dark,
            self.lbl_dark_status, self.choose_dark,
        )

        # Thẻ Reference
        self.lbl_ref_status = MDLabel(
            text="Chưa chọn", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_ref = MDLabel(
            text="Chọn ảnh Reference", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_ref = _make_image_card(
            "white-balance-sunny", self.btn_ref,
            self.lbl_ref_status, self.choose_ref,
        )

        # Nút HIỆU CHỈNH
        self.btn_run = _make_run_btn("HIỆU CHỈNH THIẾT BỊ", self.run_calibration)

        bottom_row.add_widget(self.card_dark)
        bottom_row.add_widget(self.card_ref)
        bottom_row.add_widget(self.btn_run)
        layout.add_widget(bottom_row)
        self.add_widget(layout)
        self._last_calib_saved_info = None

    def _refresh_status_card(self):
        """Cập nhật card trạng thái dựa vào có file calibration.json chưa."""
        self.status_card.clear_widgets()
        content = MDBoxLayout(orientation="vertical", spacing=dp(10))

        if processing.has_calibration():
            import json, os
            calib_path = os.path.join(os.path.dirname(processing.__file__), "calibration.json")
            try:
                with open(calib_path, encoding="utf-8") as f:
                    p = json.load(f)
                w = p.get("image_width", "?")
                h = p.get("image_height", "?")
                wl_min = p.get("wl_min", "?")
                wl_max = p.get("wl_max", "?")
                n_ch   = len(p.get("y_centers", []))
                status_text = (
                    f"[color=0F6E56][b]✓ Đã hiệu chỉnh[/b][/color]\n\n"
                    f"Kích thước ảnh: {w}×{h} px\n"
                    f"Số kênh: {n_ch}\n"
                    f"Dải bước sóng: {wl_min}–{wl_max} nm\n\n"
                    f"Để hiệu chỉnh lại, chọn ảnh Reference mới và nhấn nút bên dưới."
                )
            except Exception:
                status_text = "[color=0F6E56][b]✓ Đã hiệu chỉnh[/b][/color]\n\nNhấn để hiệu chỉnh lại."
        else:
            status_text = (
                "[color=E24B4A][b]⚠ Chưa hiệu chỉnh thiết bị[/b][/color]\n\n"
                "Để đo chính xác, bạn cần hiệu chỉnh thiết bị lần đầu:\n\n"
                "1. Chụp ảnh đèn LED trắng (không mẫu) → Chọn một hoặc nhiều ảnh Reference (trung bình)\n"
                "2. (Tùy chọn) Chụp ảnh trong tối → Chọn một hoặc nhiều ảnh Dark (trung bình)\n"
                "3. Nhấn [b]HIỆU CHỈNH THIẾT BỊ[/b]\n\n"
                "Chỉ cần làm lại khi thay đổi cách lắp camera hoặc cách tử."
            )

        content.add_widget(MDLabel(
            text=status_text,
            markup=True,
            font_style="Body1",
            halign="left",
            valign="top",
            size_hint=(1, 1),
        ))
        self.status_card.add_widget(content)

    # ── Chọn ảnh ──
    def choose_ref(self, instance):
        open_file_chooser(True, self._set_ref)

    def choose_dark(self, instance):
        open_file_chooser(True, self._set_dark)

    def _set_ref(self, selection):
        self.ref_paths = selection or []
        count = len(self.ref_paths)
        if count == 1:
            self.lbl_ref_status.text = os.path.basename(self.ref_paths[0])
        elif count > 1:
            self.lbl_ref_status.text = f"Đã chọn {count} ảnh (trung bình)"
        else:
            self.lbl_ref_status.text = "Chưa chọn"
        self.lbl_ref_status.theme_text_color = "Primary" if count else "Hint"

    def _set_dark(self, selection):
        self.dark_paths = selection or []
        count = len(self.dark_paths)
        if count == 1:
            self.lbl_dark_status.text = os.path.basename(self.dark_paths[0])
        elif count > 1:
            self.lbl_dark_status.text = f"Đã chọn {count} ảnh (trung bình)"
        else:
            self.lbl_dark_status.text = "Chưa chọn (tùy chọn)"
        self.lbl_dark_status.theme_text_color = "Primary" if count else "Hint"

    # ── Chạy hiệu chỉnh trong thread riêng ──
    def run_calibration(self, instance):
        if not self.ref_paths:
            self.show_alert("Thiếu ảnh", "Hãy chọn ít nhất một ảnh Reference (đèn LED trắng).")
            return

        # Hiện dialog loading
        self._loading_dialog = MDDialog(
            title="Đang hiệu chỉnh...",
            text="Hệ thống đang phân tích ảnh.\nVui lòng đợi trong giây lát.",
        )
        self._loading_dialog.open()

        def _worker():
            try:
                processing.run_calibration(
                    self.ref_paths,
                    dark_path=self.dark_paths if self.dark_paths else None,
                )
                calib_path = os.path.join(os.path.dirname(processing.__file__), "calibration.json")
                if os.path.exists(calib_path):
                    mtime = os.path.getmtime(calib_path)
                    ts = __import__("datetime").datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    self._last_calib_saved_info = (calib_path, ts)
                else:
                    self._last_calib_saved_info = None
                Clock.schedule_once(self._on_calib_success, 0)
            except Exception as e:
                err_msg = str(e)
                Clock.schedule_once(lambda dt: self._on_calib_error(err_msg), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_calib_success(self, dt):
        self._loading_dialog.dismiss()
        self._refresh_status_card()
        extra = ""
        if self._last_calib_saved_info:
            path, ts = self._last_calib_saved_info
            extra = f"\n\nĐã lưu tại:\n{path}\nCập nhật lúc: {ts}"
        self.show_alert(
            "Hiệu chỉnh thành công",
            "Thiết bị đã được hiệu chỉnh.\n"
            "Các thông số sẽ tự động áp dụng cho tất cả các lần đo."
            + extra,
        )

    def _on_calib_error(self, msg):
        self._loading_dialog.dismiss()
        self.show_alert("Lỗi hiệu chỉnh", msg)


# ────────────────────────────────────────────────
#          MÀN HÌNH HẤP THỤ
# ────────────────────────────────────────────────

class HapThuScreen(GraphMixin, MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_paths    = []
        self.sample_paths = []
        self.dark_paths   = []

        layout = MDBoxLayout(
            orientation="vertical", padding=dp(10), spacing=dp(8), size_hint=(1, 1),
        )

        # ── Card đồ thị ──
        self.plot_card = MDCard(
            padding=dp(10), elevation=2, radius=[dp(14)] * 4,
            size_hint=(1, 0.8), md_bg_color=(1, 1, 1, 1),
        )
        self.plot_card.add_widget(
            _make_empty_card_content(
                self.reset_data,
                "Chưa có dữ liệu phổ hấp thụ.\nHãy chọn ảnh để bắt đầu.",
            )
        )
        layout.add_widget(self.plot_card)

        # ── Hàng dưới: thẻ chọn ảnh + nút ──
        bottom_row = MDBoxLayout(
            orientation="horizontal", size_hint=(1, 0.2), spacing=dp(8),
        )

        # Thẻ Dark
        self.lbl_dark_status = MDLabel(
            text="Chưa chọn", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_dark = MDLabel(
            text="Chọn ảnh Dark", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_dark = _make_image_card(
            "moon-waning-crescent", self.btn_dark,
            self.lbl_dark_status, self.choose_dark_images,
        )

        # Thẻ I0
        self.lbl_ref_status = MDLabel(
            text="Chưa chọn", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_ref = MDLabel(
            text="Chọn ảnh I0", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_ref = _make_image_card(
            "camera-outline", self.btn_ref,
            self.lbl_ref_status, self.choose_ref_image,
        )

        # Thẻ Mẫu
        self.lbl_samples_status = MDLabel(
            text="Chưa chọn", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_samples = MDLabel(
            text="Chọn ảnh Mẫu", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_samples = _make_image_card(
            "image-multiple-outline", self.btn_samples,
            self.lbl_samples_status, self.choose_sample_images,
        )

        # Nút TÍNH & VẼ PHỔ
        self.btn_run = _make_run_btn("TÍNH & VẼ PHỔ", self.run_processing)

        bottom_row.add_widget(self.card_dark)
        bottom_row.add_widget(self.card_ref)
        bottom_row.add_widget(self.card_samples)
        bottom_row.add_widget(self.btn_run)
        layout.add_widget(bottom_row)
        self.add_widget(layout)

    # ── Chọn ảnh ──
    def choose_ref_image(self, instance):
        open_file_chooser(True, self.set_ref_paths)

    def choose_sample_images(self, instance):
        open_file_chooser(True, self.set_sample_paths)

    def choose_dark_images(self, instance):
        open_file_chooser(True, self.set_dark_paths)

    def set_ref_paths(self, selection):
        self.ref_paths = selection or []
        count = len(self.ref_paths)
        self.lbl_ref_status.text = f"Đã chọn {count} file" if count else "Chưa chọn"
        self.lbl_ref_status.theme_text_color = "Primary" if count else "Hint"

    def set_sample_paths(self, selection):
        self.sample_paths = selection or []
        count = len(self.sample_paths)
        self.lbl_samples_status.text = f"Đã chọn {count} file" if count else "Chưa chọn"
        self.lbl_samples_status.theme_text_color = "Primary" if count else "Hint"

    def set_dark_paths(self, selection):
        self.dark_paths = selection or []
        count = len(self.dark_paths)
        self.lbl_dark_status.text = f"Đã chọn {count} file" if count else "Chưa chọn"
        self.lbl_dark_status.theme_text_color = "Primary" if count else "Hint"

    # ── Reset ──
    def reset_data(self, instance=None):
        self.ref_paths = []
        self.sample_paths = []
        self.dark_paths = []
        self.lbl_ref_status.text = "Chưa chọn"
        self.lbl_ref_status.theme_text_color = "Hint"
        self.lbl_dark_status.text = "Chưa chọn"
        self.lbl_dark_status.theme_text_color = "Hint"
        self.lbl_samples_status.text = "Chưa chọn"
        self.lbl_samples_status.theme_text_color = "Hint"
        self.plot_card.clear_widgets()
        self.plot_card.add_widget(
            _make_empty_card_content(
                self.reset_data,
                "Chưa có dữ liệu phổ hấp thụ.\nHãy chọn ảnh để bắt đầu.",
            )
        )

    # ── Xử lý ──
    def run_processing(self, instance):
        if not self.ref_paths or not self.sample_paths:
            self.show_alert("Thiếu dữ liệu", "Hãy chọn ảnh I₀ và ít nhất một ảnh mẫu.")
            return
        if not processing.has_calibration():
            self.show_alert(
                "Chưa hiệu chỉnh thiết bị",
                "Vào menu → Hiệu chỉnh thiết bị để thực hiện hiệu chỉnh lần đầu.\n"
                "(Chỉ cần làm một lần.)",
            )
            return
        try:
            results = processing.compute_absorption_spectrum_6ch(
                self.ref_paths,
                self.sample_paths,
                dark_paths=self.dark_paths if self.dark_paths else None,
            )
            self.draw_chart(results)
        except Exception as e:
            self.show_alert("Lỗi xử lý", str(e))

    def draw_chart(self, results):
        self.plot_card.clear_widgets()
        self.plot_card.add_widget(self._make_chart_card_content(results, "Phổ Hấp Thụ"))


# ────────────────────────────────────────────────
#          MÀN HÌNH HUỲNH QUANG
# ────────────────────────────────────────────────

class HuynhQuangScreen(GraphMixin, MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_paths = []
        self.dark_paths  = []

        layout = MDBoxLayout(
            orientation="vertical", padding=dp(10), spacing=dp(8), size_hint=(1, 1),
        )

        # ── Card đồ thị ──
        self.plot_card = MDCard(
            padding=dp(10), elevation=2, radius=[dp(14)] * 4,
            size_hint=(1, 0.8), md_bg_color=(1, 1, 1, 1),
        )
        self.plot_card.add_widget(
            _make_empty_card_content(
                self.reset_data,
                "Chưa có dữ liệu phổ huỳnh quang.\nHãy chọn ảnh để bắt đầu.",
            )
        )
        layout.add_widget(self.plot_card)

        # ── Hàng dưới ──
        bottom_row = MDBoxLayout(
            orientation="horizontal", size_hint=(1, 0.2), spacing=dp(8),
        )

        # Thẻ Dark
        self.lbl_dark_status = MDLabel(
            text="Chưa chọn", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_dark = MDLabel(
            text="Chọn ảnh Dark", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_dark = _make_image_card(
            "moon-waning-crescent", self.btn_dark,
            self.lbl_dark_status, self.choose_dark_images,
        )

        # Thẻ Mẫu
        self.lbl_status = MDLabel(
            text="Chưa chọn", theme_text_color="Hint",
            font_style="Caption", size_hint_y=None, height=dp(16),
        )
        self.btn_samples = MDLabel(
            text="Chọn ảnh Mẫu", theme_text_color="Primary",
            font_style="Subtitle2", bold=True, valign="center",
        )
        self.card_samples = _make_image_card(
            "image-multiple-outline", self.btn_samples,
            self.lbl_status, self.choose_images,
        )

        # Nút VẼ PHỔ
        self.btn_run = _make_run_btn("VẼ PHỔ", self.run_processing)

        bottom_row.add_widget(self.card_dark)
        bottom_row.add_widget(self.card_samples)
        bottom_row.add_widget(self.btn_run)
        layout.add_widget(bottom_row)
        self.add_widget(layout)

    # ── Chọn ảnh ──
    def choose_images(self, instance):
        open_file_chooser(True, self.set_image_paths)

    def choose_dark_images(self, instance):
        open_file_chooser(True, self.set_dark_paths)

    def set_image_paths(self, selection):
        self.image_paths = selection or []
        count = len(self.image_paths)
        self.lbl_status.text = f"Đã chọn {count} file" if count else "Chưa chọn"
        self.lbl_status.theme_text_color = "Primary" if count else "Hint"

    def set_dark_paths(self, selection):
        self.dark_paths = selection or []
        count = len(self.dark_paths)
        self.lbl_dark_status.text = f"Đã chọn {count} file" if count else "Chưa chọn"
        self.lbl_dark_status.theme_text_color = "Primary" if count else "Hint"

    # ── Reset ──
    def reset_data(self, instance=None):
        self.image_paths = []
        self.dark_paths  = []
        self.lbl_status.text = "Chưa chọn"
        self.lbl_status.theme_text_color = "Hint"
        self.lbl_dark_status.text = "Chưa chọn"
        self.lbl_dark_status.theme_text_color = "Hint"
        self.plot_card.clear_widgets()
        self.plot_card.add_widget(
            _make_empty_card_content(
                self.reset_data,
                "Chưa có dữ liệu phổ huỳnh quang.\nHãy chọn ảnh để bắt đầu.",
            )
        )

    # ── Xử lý ──
    def run_processing(self, instance):
        if not self.image_paths:
            self.show_alert("Thiếu dữ liệu", "Hãy chọn ít nhất một ảnh mẫu.")
            return
        if not processing.has_calibration():
            self.show_alert(
                "Chưa hiệu chỉnh thiết bị",
                "Vào menu → Hiệu chỉnh thiết bị để thực hiện hiệu chỉnh lần đầu.\n"
                "(Chỉ cần làm một lần.)",
            )
            return
        try:
            results = processing.compute_fluorescence_spectrum_6ch(
                self.image_paths,
                dark_paths=self.dark_paths if self.dark_paths else None,
            )
            self.draw_chart(results)
        except Exception as e:
            self.show_alert("Lỗi xử lý", str(e))

    def draw_chart(self, results):
        self.plot_card.clear_widgets()
        self.plot_card.add_widget(self._make_chart_card_content(results, "Phổ Huỳnh Quang"))