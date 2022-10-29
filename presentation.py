class Presentation:
    data: dict
    id: int

    def __init__(self, idx: int, data: dict) -> None:
        self.id = idx
        self.data = data

    def data(self) -> dict:
        return self.data
