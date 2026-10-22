"""Single Qt stylesheet: restrained dark theme."""

BG = "#15151b"
PANEL = "#1d1d26"
BORDER = "#2e2e3a"
TEXT = "#e7e7ee"
MUTED = "#b0b0c2"
ACCENT = "#d870a6"

STYLESHEET = f"""
* {{ font-family: "Segoe UI", sans-serif; font-size: 10.5pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QLabel#title {{ font-size: 21pt; font-weight: 600; }}
QLabel#version {{ color: {ACCENT}; font-weight: 600; padding-top: 10px; }}
QLabel#subtitle, QLabel#hint {{ color: {MUTED}; }}
QLabel#count {{ font-weight: 600; }}
QLabel#disclaimer {{ color: {MUTED}; font-size: 10pt; border-top: 1px solid {BORDER}; padding-top: 8px; }}
QPlainTextEdit, QTextBrowser, QListWidget {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px;
}}
QPlainTextEdit, QTextBrowser {{ selection-background-color: {ACCENT}; selection-color: #101014; }}
QListWidget {{ outline: 0; }}
QListWidget::item {{ padding: 7px 6px; border-radius: 5px; }}
/* Light text on the selected row: the marker and label, not color, carry the category. */
QListWidget::item:selected {{ background: #3a2a45; color: {TEXT}; border-left: 3px solid {ACCENT}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 18px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #1a1018; font-weight: 600; }}
QPushButton#primary:hover {{ background: #e585b6; }}
QTabWidget::pane {{ border: none; top: 6px; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 18px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QSplitter::handle {{ background: transparent; }}
QStatusBar {{ color: {MUTED}; padding: 2px 10px; }}
QStatusBar::item {{ border: none; }}
"""
