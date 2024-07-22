import sys
from typing import Callable
from functools import partial
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
)

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Wikipedia Search Engine"
SEARCH_BOX_HEIGHT = 64
SEARCH_BOX_LABEL = "Wikipedia Search"
SEARCH_BUTTON_TEXT = "&Search"

class SearchWindow(QMainWindow):
    """
    Main window for search engine.
    """
    def __init__(self):
        super().__init__()
        # Attributes:
        self.generalLayout = QVBoxLayout()
        self.searchBoxLabel = QLabel()
        self.searchBox = QLineEdit()
        self.searchButton = QPushButton(SEARCH_BUTTON_TEXT)

        # Formatting for widgets:
        self.searchBoxLabel.setText(SEARCH_BOX_LABEL)
        self.searchBox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.searchBox.setFixedHeight(SEARCH_BOX_HEIGHT)

        self.generalLayout.addWidget(self.searchBoxLabel)
        self.generalLayout.addWidget(self.searchBox)
        self.generalLayout.addWidget(self.searchButton)

        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)
        # Formatting for main window:
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle(WINDOW_TITLE)
        self.setCentralWidget(centralWidget)

class SearchController:
    """
    Controller class for the search engine app window.
    """
    def __init__(self, model: Callable, view: SearchWindow):
        # Attributes:
        self._evaluate: Callable = model
        self._view: SearchWindow = view

        # Widget initialisation:
        self._connectSignalsAndSlots()

    def _connectSignalsAndSlots(self) -> None:
        pass

def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow()

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

if __name__ == "__main__":
    main()
