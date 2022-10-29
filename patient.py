class Patient:
    name: str
    presentations: list

    def __init__(self, name: str) -> None:
        self.name = name
        self.presentations = presentations

    def name(self) -> str:
        return self.name

    def presentations(self) -> list:
        return self.presentations
