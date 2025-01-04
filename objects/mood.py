import json

class Mood:
    #TODO: define an evaluation
    
    def __init__(self, text: str):
        self.__text = text
        self.__weight = None
    
    def get_weight(self) -> float:
        return self.__weight
    
    def get_text(self) -> str:
        return self.__text
    
    # definition of equivalences.
    
    def __eq__(self, other) -> bool: # class equivalence is defined by text
        return self.__text == other.get_text()
    
    def __weight_is_defined(self, other) -> bool: # class inequivalence is defined by weights
        if self.__weight == None or other.get_weight() == None:
            return False # returns false if evaluation is undefined for any object
        else:
            return True
    
    def __le__(self, other) -> bool:
        return self.__weight_is_defined(other) and self.__weight < other.get_weight()
    
    def __gt__(self, other) -> bool:
        return self.__weight_is_defined(other) and self.__weight > other.get_weight()
    
    def __le__(self, other) -> bool:
        return self.__weight_is_defined(other) and self.__weight <= other.get_weight()
    
    def __ge__(self, other) -> bool:
        return self.__weight_is_defined(other) and self.__weight >= other.get_weight()
    
    # definition of operations.
    
    def __add__(self, other):
        self.__weight + other.weight()
        del other
    
    # representation.
    
    def get_json(self):
        return json.dumps({"text": self.__text, "weight": self.__weight})
    
    def __str__(self):
        return self.__text + " | " + str(self.__weight)