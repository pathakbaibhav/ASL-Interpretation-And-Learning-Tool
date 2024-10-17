import json

class State:
    """
    A class to manage and persist the state of the application, 
    including predicted letter, current word, confidence score, and count.
    """

    def __init__(self, predictedLetter="", currentWord="", confidenceScore=0.0, count=0):
        self._predictedLetter = predictedLetter
        self._currentWord = currentWord
        self._confidenceScore = confidenceScore
        self._count = count
        self._filename = 'state.json'  # File to save the state.

    # Getters
    def get_predicted_letter(self):
        """Returns the predicted letter."""
        return self._predictedLetter

    def get_current_word(self):
        """Returns the current word being formed."""
        return self._currentWord

    def get_confidence_score(self):
        """Returns the current confidence score."""
        return self._confidenceScore

    def get_count(self):
        """Returns the current count value."""
        return self._count

    # Setters
    def set_predicted_letter(self, predictedLetter):
        """Sets the predicted letter and saves the state."""
        self._predictedLetter = predictedLetter
        self.save_state()

    def set_current_word(self, currentWord):
        """Sets the current word and saves the state."""
        self._currentWord = currentWord
        self.save_state()

    def set_confidence_score(self, confidenceScore):
        """
        Sets the confidence score if it is between 0.0 and 1.0.
        Raises a ValueError if the score is outside this range.
        """
        if 0.0 <= confidenceScore <= 1.0:
            self._confidenceScore = confidenceScore
            self.save_state()
        else:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

    def set_count(self, count):
        """
        Sets the count value if it is non-negative.
        Raises a ValueError if the count is negative.
        """
        if count < 0:
            raise ValueError("Count cannot be negative")
        self._count = count
        self.save_state()

    def append_to_current_word(self, letter):
        """Appends a letter to the current word and saves the state."""
        self._currentWord += letter
        self.save_state()

    def save_state(self):
        """Saves the current state to a JSON file."""
        with open(self._filename, 'w') as file:
            json.dump(self.__dict__, file)

    def load_state(self):
        """
        Loads the state from a JSON file. If the file is not found,
        initializes with default values.
        """
        try:
            with open(self._filename, 'r') as file:
                state_data = json.load(file)
                self._predictedLetter = state_data['_predictedLetter']
                self._currentWord = state_data['_currentWord']
                self._confidenceScore = state_data['_confidenceScore']
                self._count = state_data['_count']
        except FileNotFoundError:
            self.__init__()  # Initialize with default values if the file is missing.

    def clear_state(self):
        """Resets all state attributes to default values and saves the cleared state."""
        self._predictedLetter = ""
        self._currentWord = ""
        self._confidenceScore = 0.0
        self._count = 0
        self.save_state()

    def __str__(self):
        """Returns a string representation of the current state."""
        return (
            f"Predicted Letter: {self._predictedLetter}\n"
            f"Current Word: {self._currentWord}\n"
            f"Confidence Score: {self._confidenceScore:.2f}\n"
            f"Count: {self._count}"
        )
