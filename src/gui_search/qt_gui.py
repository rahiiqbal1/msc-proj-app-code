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
    QLineEdit,
    QPushButton,
    QStackedWidget,
)

DISPLAY_WIDTH = 3840
DISPLAY_HEIGHT = 2160
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Wikipedia Search Engine"
SEARCH_BOX_HEIGHT = 64
SEARCH_BOX_LABEL = "<h1>Wikipedia Search</h1>"
SEARCH_BUTTON_TEXT = "&Search"
NUM_RESULTS_TO_SHOW = 10

class SearchWindow(QMainWindow):
    """
    Main window for search engine.
    """
    def __init__(self) -> None:
        super().__init__()

        # Widgets:
        self.generalWidget = QStackedWidget()

        # Getting layout for search page and adding to the generalWidget, 
        # retrieving search button widget for use in controller:
        self.searchButton: QPushButton = self.showSearchPage()["searchButton"]

        # Basic window formatting:
        self.setGeometry(
            math.floor(0.4 * DISPLAY_WIDTH),
            math.floor(0.4 * DISPLAY_HEIGHT),
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
        self.setWindowTitle(WINDOW_TITLE)

        # Set layout:
        self.setCentralWidget(self.generalWidget)

    def showSearchPage(self) -> dict[str, QWidget]:
        """
        Generates the layout for the search page, adds it to a widget instance,
        and adds that widget to the general (stacked) widget. 

        Returns a dictionary of each widget with it's variable name as key.
        """
        # Widgets:
        overallLayout = QVBoxLayout()
        searchBoxLabel = QLabel()
        searchBox = QLineEdit()
        searchButton = QPushButton(SEARCH_BUTTON_TEXT)

        # Formatting for widgets:
        searchBoxLabel.setText(SEARCH_BOX_LABEL)
        searchBox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        searchBox.setFixedHeight(SEARCH_BOX_HEIGHT)
        searchButton.setCheckable(True)

        # Adding to layout:
        overallLayout.addWidget(searchBoxLabel)
        overallLayout.addWidget(searchBox)
        overallLayout.addWidget(searchButton)

        centralWidget = QWidget()
        centralWidget.setLayout(overallLayout)

        self.generalWidget.addWidget(centralWidget)

        return {
            "overallLayout": overallLayout,
            "searchBoxLabel": searchBoxLabel,
            "searchBox": searchBox,
            "searchButton": searchButton,
            "centralWidget": centralWidget
        }

    def showResultsPage(self, num_results_to_show: int = 10) -> dict[str, Any]:
        """
        Shows results page for given results.
        """
        # Widgets:
        generalLayout = QVBoxLayout()

        # Adding results to widget:
        for i in range(num_results_to_show):
            result = QLabel(f"test{i}")
            generalLayout.addWidget(result)
            print(i)

        # Creating central widget:
        centralWidget = QWidget()
        centralWidget.setLayout(generalLayout)

        # Adding to general stacked widget of class instance:
        self.generalWidget.addWidget(centralWidget)

        # Switching general stacked widget's index to display results:
        self.generalWidget.setCurrentIndex(
            self.generalWidget.currentIndex() + 1
        )

        return {
            "generalLayout": generalLayout
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
        self._view.searchButton.clicked.connect(self._view.showResultsPage)

def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow()

    # Creating controller. Does not need to be stored as a variable as it holds
    # references to the model and view:
    SearchController(searchWindow.showResultsPage, searchWindow)

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

if __name__ == "__main__":
    main()
