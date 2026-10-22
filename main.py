import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from app.dictionary import DictionaryError, load_entries
from app.styles import STYLESHEET
from app.window import APP_NAME, MainWindow


def main() -> int:
    qt = QApplication(sys.argv)
    try:
        entries = load_entries()
    except DictionaryError as exc:
        QMessageBox.critical(None, APP_NAME, f"Could not load the bundled terminology data:\n\n{exc}")
        return 1
    qt.setStyleSheet(STYLESHEET)
    window = MainWindow(entries)
    window.show()
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
