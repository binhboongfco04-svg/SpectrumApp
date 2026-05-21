# main.py
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.utils import platform

# Import các màn hình từ ui_components.py
from ui_components import HapThuScreen, HuynhQuangScreen, CalibrationScreen

KV = '''
MDNavigationLayout:

    MDScreenManager:
        id: screen_manager

        MDScreen:
            name: "main"

            MDBoxLayout:
                orientation: "vertical"

                MDTopAppBar:
                    id: top_bar
                    title: "Đo Hấp Thụ"
                    md_bg_color: 0.098, 0.608, 0.596, 1
                    specific_text_color: 1, 1, 1, 1
                    left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]

                MDScreenManager:
                    id: content_manager

    MDNavigationDrawer:
        id: nav_drawer

        MDNavigationDrawerMenu:
            MDNavigationDrawerHeader:
                title: "Menu"
                title_color: 0.098, 0.608, 0.596, 1

            MDNavigationDrawerItem:
                text: "Đo Hấp Thụ"
                icon: "chart-line"
                on_release:
                    nav_drawer.set_state("close")
                    app.switch_to_hapthu()

            MDNavigationDrawerItem:
                text: "Đo Huỳnh Quang"
                icon: "chart-bell-curve"
                on_release:
                    nav_drawer.set_state("close")
                    app.switch_to_huynhquang()

            MDNavigationDrawerDivider:

            MDNavigationDrawerItem:
                text: "Hiệu chỉnh thiết bị"
                icon: "tune"
                on_release:
                    nav_drawer.set_state("close")
                    app.switch_to_calibration()
'''

class SpectrumApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.primary_hue = "600"
        self.theme_cls.theme_style = "Light"
        self.root = Builder.load_string(KV)
        self.content_manager = self.root.ids.content_manager
        self.screen_hapthu      = HapThuScreen(name="hap_thu")
        self.screen_huynhquang  = HuynhQuangScreen(name="huynh_quang")
        self.screen_calibration = CalibrationScreen(name="calibration")

        self.content_manager.add_widget(self.screen_hapthu)
        self.content_manager.add_widget(self.screen_huynhquang)
        self.content_manager.add_widget(self.screen_calibration)

        self.content_manager.current = "hap_thu"
        return self.root

    def on_start(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission  # type: ignore
            request_permissions([Permission.READ_MEDIA_IMAGES])

    def switch_to_hapthu(self):
        self.root.ids.content_manager.current = "hap_thu"
        self.root.ids.top_bar.title = "Đo Hấp Thụ"

    def switch_to_huynhquang(self):
        self.root.ids.content_manager.current = "huynh_quang"
        self.root.ids.top_bar.title = "Đo Huỳnh Quang"

    def switch_to_calibration(self):
        self.root.ids.content_manager.current = "calibration"
        self.root.ids.top_bar.title = "Hiệu Chỉnh Thiết Bị"

if __name__ == "__main__":
    SpectrumApp().run()