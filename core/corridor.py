class Corridor:
    """
    Represents circulation space (walkable passage network) in the building.

    Attributes:
        name                 – unique identifier (e.g. "Circulation_1")
        polygon              – walkable region polygon [(x,y), ...]
        width                – corridor clear width in metres
        length               – approximated centerline length in metres
        connects             – room names adjacent to this circulation space
        connectivity_to_exit – whether this circulation reaches the exit/entrance path
        spine_points         – source spine polyline points used to generate the walkable region
    """

    def __init__(
        self,
        name,
        polygon,
        width,
        length,
        connects=None,
        connectivity_to_exit=True,
        spine_points=None,
    ):
        self.name = name
        self.polygon = polygon
        self.width = width
        self.length = length
        self.connects = connects or []
        self.connectivity_to_exit = connectivity_to_exit
        self.spine_points = spine_points or []

    @property
    def area(self):
        return round(self.width * self.length, 3)

    @property
    def walkable_area(self):
        return self.area

    def __repr__(self):
        return (
            f"CirculationSpace({self.name!r}, w={self.width}, l={self.length}, "
            f"walkable_area={self.walkable_area}, connects={self.connects}, "
            f"connectivity_to_exit={self.connectivity_to_exit})"
        )
