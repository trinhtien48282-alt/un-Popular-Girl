"""Main window. All detection logic lives in scanner.py; this file is UI only."""
from html import escape

from PySide6.QtGui import QColor, QKeySequence, QTextCharFormat, QTextCursor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit,
    QPushButton, QSplitter, QTabWidget, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from . import __version__
from .models import Category, Entry, Match
from .scanner import Scanner, line_col

APP_NAME = "un-Popular Girl"
IDLE_STATUS = "Consulting the dictionary, not a therapist."
DISCLAIMER = (
    "Educational, rule-based tool. It is not a diagnostic instrument and cannot diagnose "
    "you or anyone else. It matches keywords and a few nearby clue words; it does not "
    "understand intent, sarcasm, or negation, so expect false positives and misses. "
    "Context always matters. Fully offline: no text ever leaves this computer."
)
EMPTY_SCAN = (
    "<h3>Nothing inspected yet</h3><p>Paste some text on the left and press "
    "<b>Inspect the vocabulary</b>. Each occurrence of a listed term will show up here "
    "with an explanation.</p><p>A match is a prompt to think about context, not an accusation.</p>"
)
NO_MATCHES = (
    "<h3>No listed terms found</h3><p>The vocabulary is, for now, unbothered. "
    "This only means none of the 15 bundled terms appeared in that text.</p>"
)

# Marker + color: the marker and the label carry the meaning, color is only a hint.
STYLE = {
    Category.POSSIBLE_MISUSE: ("\u25B2", "#ff8fa3"),
    Category.NEEDS_CONTEXT: ("\u25C6", "#f5c26b"),
    Category.PLAUSIBLE_USAGE: ("\u2713", "#7fd1a8"),
}


def _browser() -> QTextBrowser:
    b = QTextBrowser()
    b.setOpenLinks(False)  # source URLs are shown as text and never opened
    b.setOpenExternalLinks(False)
    return b


def _list_html(items: tuple[str, ...]) -> str:
    return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def _entry_body(e: Entry) -> str:
    source = (
        f"{escape(e.source_title)}<br>{escape(e.source_url)}"
        if e.source_url
        else "No single authoritative source is bundled for this entry."
    )
    return (
        f"<p><i>{escape(e.term_category)}</i></p>"
        f"<p>{escape(e.definition)}</p>"
        f"<h4>When it may be appropriate</h4><p>{escape(e.guidance)}</p>"
        f"<h4>Commonly oversimplified usage</h4>{_list_html(e.misuse_examples)}"
        f"<h4>Legitimate usage</h4>{_list_html(e.legitimate_examples)}"
        f"<h4>Further reference (not opened automatically)</h4><p>{source}</p>"
    )


def _match_html(m: Match, text_len: int) -> str:
    marker, color = STYLE[m.category]
    flat = lambda s: escape(s.replace("\n", " "))  # noqa: E731
    cut_left = "\u2026" if m.start > len(m.before) else ""
    cut_right = "\u2026" if m.end + len(m.after) < text_len else ""
    humor = (
        f"<p><i>{escape(m.entry.humor)}</i></p>"
        if m.entry.humor and m.category is Category.POSSIBLE_MISUSE
        else ""
    )
    return (
        f"<h3>{escape(m.entry.term)}</h3>"
        f'<p><b style="color:{color}">{marker} {m.category.value}</b><br>{escape(m.reason)}</p>'
        f"<p>{cut_left}{flat(m.before)}<b>{flat(m.text)}</b>{flat(m.after)}{cut_right}</p>"
        f"{humor}{_entry_body(m.entry)}"
    )


def _u16(text: str, index: int) -> int:
    """Python str index -> Qt (UTF-16) index; they differ after emoji etc."""
    return len(text[:index].encode("utf-16-le")) // 2


class MainWindow(QMainWindow):
    def __init__(self, entries: list[Entry]):
        super().__init__()
        self._entries = entries
        self._scanner = Scanner(entries)
        self._matches: list[Match] = []
        self._scanned_text = ""

        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setMinimumSize(860, 580)
        self.resize(1100, 720)

        tabs = QTabWidget()
        tabs.addTab(self._build_scan_tab(), "Scanner")
        tabs.addTab(self._build_dictionary_tab(), f"Dictionary ({len(entries)})")

        title_row = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        version = QLabel(__version__)
        version.setObjectName("version")
        title_row.addWidget(title)
        title_row.addWidget(version)
        title_row.addStretch(1)
        subtitle = QLabel("An offline vocabulary checker for therapy-speak. Educational, mildly sarcastic, never diagnostic.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setObjectName("disclaimer")
        disclaimer.setWordWrap(True)

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 14, 20, 12)
        layout.setSpacing(6)
        layout.addLayout(title_row)
        layout.addWidget(subtitle)
        layout.addWidget(tabs, 1)
        layout.addWidget(disclaimer)
        self.setCentralWidget(root)
        self.statusBar().showMessage(IDLE_STATUS)

    # ---- scanner tab -------------------------------------------------
    def _build_scan_tab(self) -> QWidget:
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Paste or type text to inspect...")
        self.editor.textChanged.connect(lambda: self.editor.setExtraSelections([]))
        scan_btn = QPushButton("Inspect the vocabulary")
        scan_btn.setObjectName("primary")
        scan_btn.setShortcut(QKeySequence("Ctrl+Return"))
        scan_btn.setToolTip("Ctrl+Enter")
        scan_btn.clicked.connect(self.scan)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        buttons = QHBoxLayout()
        buttons.addWidget(scan_btn)
        buttons.addWidget(clear_btn)
        buttons.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        hint = QLabel("Text to inspect")
        hint.setObjectName("hint")
        left_layout.addWidget(hint)
        left_layout.addWidget(self.editor, 1)
        left_layout.addLayout(buttons)

        self.count_label = QLabel("0 matched terms")
        self.count_label.setObjectName("count")
        self.results = QListWidget()
        self.results.currentRowChanged.connect(self._show_match)
        self.detail = _browser()
        self.detail.setHtml(EMPTY_SCAN)
        inner = QSplitter(Qt.Orientation.Vertical)
        inner.setChildrenCollapsible(False)
        inner.addWidget(self.results)
        inner.addWidget(self.detail)
        inner.setStretchFactor(0, 1)
        inner.setStretchFactor(1, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.addWidget(self.count_label)
        right_layout.addWidget(inner, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        return split

    def scan(self) -> None:
        text = self.editor.toPlainText()
        if not text.strip():
            self._set_results([], text)
            self.statusBar().showMessage("There's nothing to inspect. Give me some text first.")
            return
        matches = self._scanner.scan(text)
        self._set_results(matches, text)
        self.statusBar().showMessage("Scan complete. The vocabulary has been questioned.")

    def clear(self) -> None:
        self.editor.clear()
        self._set_results([], "")
        self.detail.setHtml(EMPTY_SCAN)
        self.statusBar().showMessage("Cleared. " + IDLE_STATUS)

    def _set_results(self, matches: list[Match], text: str) -> None:
        self._matches = matches
        self._scanned_text = text
        self.results.clear()
        for m in matches:
            marker, color = STYLE[m.category]
            line, col = line_col(text, m.start)
            item = QListWidgetItem(f"{marker} {m.category.value}  |  {m.entry.term}  (line {line}, col {col})")
            item.setForeground(QColor(color))
            self.results.addItem(item)
        counts = [
            f"{sum(m.category is c for m in matches)} {c.value.lower()}"
            for c in Category
            if any(m.category is c for m in matches)
        ]
        n = len(matches)
        self.count_label.setText(
            f"{n} matched term{'s' if n != 1 else ''}" + (f"  ({', '.join(counts)})" if counts else "")
        )
        self.editor.setExtraSelections([])
        if matches:
            self.results.setCurrentRow(0)
        else:
            self.detail.setHtml(NO_MATCHES if text.strip() else EMPTY_SCAN)

    def _show_match(self, row: int) -> None:
        if not 0 <= row < len(self._matches):
            return
        m = self._matches[row]
        self.detail.setHtml(_match_html(m, len(self._scanned_text)))
        # Highlight in the editor only if the text is unchanged since the scan.
        if self.editor.toPlainText() != self._scanned_text:
            return
        sel = QTextEdit.ExtraSelection()
        cursor = QTextCursor(self.editor.document())
        cursor.setPosition(_u16(self._scanned_text, m.start))
        cursor.setPosition(_u16(self._scanned_text, m.end), QTextCursor.MoveMode.KeepAnchor)
        sel.cursor = cursor
        fmt = QTextCharFormat()  # build then assign: PySide returns copies from sel.format
        fmt.setBackground(QColor("#6b2d55"))
        sel.format = fmt
        self.editor.setExtraSelections([sel])

    # ---- dictionary tab ----------------------------------------------
    def _build_dictionary_tab(self) -> QWidget:
        terms = QListWidget()
        for e in self._entries:
            terms.addItem(e.term)
        detail = _browser()

        def show(row: int) -> None:
            if 0 <= row < len(self._entries):
                e = self._entries[row]
                detail.setHtml(f"<h3>{escape(e.term)}</h3>{_entry_body(e)}")

        terms.currentRowChanged.connect(show)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(terms)
        split.addWidget(detail)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        terms.setCurrentRow(0)
        return split
