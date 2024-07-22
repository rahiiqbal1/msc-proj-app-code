import sys
import math
from typing import Callable, Any
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

DISPLAY_WIDTH = 3840
DISPLAY_HEIGHT = 2160
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Wikipedia Search Engine"
SEARCH_BOX_HEIGHT = 64
SEARCH_BOX_LABEL = "<h1>Wikipedia Search</h1>"
SEARCH_BUTTON_TEXT = "&Search"

class SearchWindow(QMainWindow):
    """
    Main window for search engine.
    """
    def __init__(self) -> None:
        super().__init__()

        # Initialising window to display search page and retrieving search
        # button widget for use in controller:
        self.searchButton: QPushButton = self.drawSearchPage()["searchButton"]

        # Basic window formatting:
        self.setGeometry(
            math.floor(0.4 * DISPLAY_WIDTH),
            math.floor(0.4 * DISPLAY_HEIGHT),
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
        self.setWindowTitle(WINDOW_TITLE)

    def drawSearchPage(self) -> dict[str, Any]:
        """
        Draws the search page for the window.

        Returns a dictionary with each widget created for the search page.
        """
        # Widgets:
        generalLayout = QVBoxLayout()
        searchBoxLabel = QLabel()
        searchBox = QLineEdit()
        searchButton = QPushButton(SEARCH_BUTTON_TEXT)

        # Formatting for widgets:
        searchBoxLabel.setText(SEARCH_BOX_LABEL)
        searchBox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        searchBox.setFixedHeight(SEARCH_BOX_HEIGHT)

        # Adding to layout:
        generalLayout.addWidget(searchBoxLabel)
        generalLayout.addWidget(searchBox)
        generalLayout.addWidget(searchButton)

        # Creating and setting central widget:
        centralWidget = QWidget(self)
        centralWidget.setLayout(generalLayout)

        self.setCentralWidget(centralWidget)

        return {
            "generalLayout": generalLayout,
            "searchBoxLabel": searchBoxLabel,
            "searchBox": searchBox,
            "searchButton": searchButton
        }

class SearchController:
    """
    Controller class for the search engine app window.
    """
    def __init__(self, model: Callable, view: SearchWindow):
        # Attributes:
        self._searchFunction: Callable = model
        self._view: SearchWindow = view

        # Connect widgets:
        self._connectSignalsAndSlots()

    def _connectSignalsAndSlots(self) -> None:
        """
        Connects search button to given search function.
        """
        self._view.searchButton.clicked.connect(self._searchFunction)

def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow()

    testFunction = lambda: print("test")
    # Creating controller. Does not need to be stored as a variable as it holds
    # references to the model and view:
    SearchController(testFunction, searchWindow)

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

if __name__ == "__main__":
    main()
