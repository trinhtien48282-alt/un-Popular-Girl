"""Single Qt stylesheet: restrained dark theme."""

BG = "#15151b"
PANEL = "#1d1d26"
BORDER = "#2e2e3a"
TEXT = "#e7e7ee"
MUTED = "#9b9bae"
ACCENT = "#d870a6"

STYLESHEET = f"""
* {{ font-family: "Segoe UI", sans-serif; font-size: 10.5pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QLabel#title {{ font-size: 21pt; font-weight: 600; }}
QLabel#version {{ color: {ACCENT}; font-weight: 600; padding-top: 10px; }}
QLabel#subtitle, QLabel#hint {{ color: {MUTED}; }}
QLabel#count {{ font-weight: 600; }}
QLabel#disclaimer {{ color: {MUTED}; font-size: 9pt; border-top: 1px solid {BORDER}; padding-top: 8px; }}
QPlainTextEdit, QTextBrowser, QListWidget {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px;
    selection-background-color: {ACCENT}; selection-color: #101014;
}}
QListWidget::item {{ padding: 7px 6px; border-radius: 5px; }}
QListWidget::item:selected {{ background: #2c2433; border-left: 3px solid {ACCENT}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 18px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #1a1018; font-weight: 600; }}
QPushButton#primary:hover {{ background: #e585b6; }}
QTabWidget::pane {{ border: none; top: 6px; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 18px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QSplitter::handle {{ background: transparent; }}
QStatusBar {{ color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
"""
