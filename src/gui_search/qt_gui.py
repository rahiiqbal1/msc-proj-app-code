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
# Display:
DISPLAY_WIDTH = 3840    
DISPLAY_HEIGHT = 2160
# Window:
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Wikipedia Search Engine"
# Search box:
SEARCH_BOX_HEIGHT = 64
SEARCH_BOX_WIDTH = 1024
SEARCH_BOX_LABEL = "<h1>Wikipedia Search</h1>"
# Search button:
SEARCH_BUTTON_TEXT = "&Search"
# Data and results:
NUM_RESULTS_TO_SHOW = 10

def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow()

    # Creating controller. Does not need to be stored as a variable as it holds
    # references to the model and view:
    def testSearchFunction(_: str) -> list[dict[str, str]]:
        return [
            {"name": "blah",      "url": "here.com"},
            {"name": "more blah", "url": "there.org"}
        ] * 5
    SearchController(testSearchFunction, searchWindow)

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

class SearchWindow(QMainWindow):
    """
    Main window for search engine.
    """
    def __init__(self) -> None:
        super().__init__()

        # Widgets:
        self.generalWidget = QStackedWidget()

        # Getting layout for search page and adding to the generalWidget, 
        # retrieving search button widget for use in controller. The search 
        # page should always be at index 0 within the stacked widget:
        self.searchWidgets: dict[str, Any] = self.showSearchPage() 

        # # Basic window formatting:
        # self.setGeometry(
        #     math.floor(0.4 * DISPLAY_WIDTH),
        #     math.floor(0.4 * DISPLAY_HEIGHT),
        #     WINDOW_WIDTH,
        #     WINDOW_HEIGHT
        # )
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
        searchBoxLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        searchBox.setFixedWidth(SEARCH_BOX_WIDTH)
        searchBox.setFixedHeight(SEARCH_BOX_HEIGHT)

        searchButton.setFixedWidth(SEARCH_BOX_WIDTH)
        searchButton.setFixedHeight(SEARCH_BOX_HEIGHT)

        overallLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Adding to layout:
        overallLayout.addStretch()
        overallLayout.addWidget(searchBoxLabel)
        overallLayout.addWidget(searchBox)
        overallLayout.addWidget(searchButton)
        overallLayout.addStretch()

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

    def showResultsPage(
        self,
        searchFunction: Callable,
        ) -> dict[str, Any]:
        """
        Shows results page using given search function and query.
        """
        # List of keys within the results which we want to see:
        FieldsToShow: tuple[str, ...] = ("name", "url")

        # Evaluating results:
        results: list[dict[str, str]] = searchFunction()

        # Widgets:
        overallLayout = QVBoxLayout()

        # Adding results to widget, only showing NUM_RESULTS_TO_SHOW results:
        for resultIdx in range(NUM_RESULTS_TO_SHOW):
            # Vertical box layout to contain all information for the current
            # result. This will be added to the overallLayout when ready:
            thisResultLayout = QVBoxLayout()

            # Only showing desired fields. Here we construct the string which
            # will be the label, and then add it to the layout when it is fully
            # constructed:
            currentResultStringToShow: str = ""
            field: str
            for field in FieldsToShow:
                if field == "name":
                    currentResultStringToShow += (
                        f"<b>{results[resultIdx][field]}</b><br>"
                    )
                # Inserting url as hyperlink:
                elif field == "url":
                    currentResultStringToShow += (
                        f"<a href='{results[resultIdx][field]}'>" + 
                        f"{results[resultIdx][field]}</a>"
                    )

            # Creating QLabel with string to show and adding to layout:
            resultLabelWidget = QLabel(currentResultStringToShow)
            resultLabelWidget.setTextFormat(Qt.RichText)
            resultLabelWidget.setTextInteractionFlags(
                Qt.LinksAccessibleByMouse
            )
            resultLabelWidget.setOpenExternalLinks(True)

            # Adding the desired result outputs to this result's layout, then
            # adding that layout to the overall layout. Uses a sub-widget to
            # make this possible:
            thisResultLayout.addWidget(resultLabelWidget)
            subWidget = QWidget()
            subWidget.setLayout(thisResultLayout)
            overallLayout.addWidget(subWidget)

        # Creating central widget:
        centralWidget = QWidget()
        centralWidget.setLayout(overallLayout)

        # Adding to general stacked widget of class instance. Placing at index
        # 1:
        self.generalWidget.insertWidget(1, centralWidget)

        # Switching general stacked widget's index to display results:
        self.generalWidget.setCurrentIndex(1)

        return {
            "overallLayout": overallLayout
        }

class SearchController:
    """
    Controller class for the search engine app window.
    """
    def __init__(self, model: Callable, view: SearchWindow, *model_args):
        # Attributes. model should be a function returning results represented
        # as list[dict[str, str]], a list of deserialised JSON objects. The 
        # final argument of model must be the search query:
        self._model: Callable = model
        self._view: SearchWindow = view
        # Arguments to be passed to model, to be unpacked when used:
        self._modelArgs: tuple[Any, ...] = model_args

        # Connect widgets:
        self._connectSignalsAndSlots()

    def _searchFunction(self) -> list[dict[str, str]]:
        """
        Deals with searching through data with query present in search box.
        """
        searchQuery: str = self._view.searchWidgets["searchBox"].text()
        # Passing given modelArgs to model along with search query present in
        # text box:
        print(searchQuery)
        return self._model(*(self._modelArgs + (searchQuery, )))

    def _connectSignalsAndSlots(self) -> None:
        """
        Connects search button to given search function.
        """
        # Connecting search button to search function. showResultsPage takes
        # the results of the search function as it's argument:
        self._view.searchWidgets["searchButton"].clicked.connect(
            partial(
                self._view.showResultsPage, 
                self._searchFunction
            )
        )

if __name__ == "__main__":
    main()
