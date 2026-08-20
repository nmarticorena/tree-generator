from random import Random
from forsym.tree.config import LSystemConfig
import math

from anytree import NodeMixin

from forsym.fractal import rule_parser as parser
from forsym.fractal import turtle_steer as steering


class Point:
    """An x-y-z coordinate"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        if isinstance(other, Point):
            return (
                round(self.x, 5) == round(other.x, 5)
                and round(self.y, 5) == round(other.y, 5)
                and round(self.z, 5) == round(other.z, 5)
            )
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __str__(self):
        return f"Point({round(self.x, 5)}, {round(self.y, 5)}, {round(self.z, 5)})"

    def distance_to(self, other):
        """Calculate the Euclidean distance to another point.

        Parameters
        ----------
        other : Point
            Point at the other end of the distance measurement.

        Returns
        -------
        float
            Euclidean distance between the points.

        Raises
        ------
        TypeError
            If ``other`` is not a :class:`Point`.
        """
        if isinstance(other, Point):
            dx = self.x - other.x
            dy = self.y - other.y
            dz = self.z - other.z
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            return distance
        raise TypeError("Distance can only be calculated between two Point instances.")


class TurtleLine:
    """A three-dimensional turtle segment and its local coordinate frame."""

    def __init__(self, start: Point, hlu, width):
        self.start = start
        self.end = None
        self.hlu = hlu
        self.width = width

    def is_nan(self):
        """Check whether either endpoint is missing or contains NaN.

        Returns
        -------
        bool
            ``True`` when the line is incomplete or has a NaN coordinate.
        """
        if self.end is None or self.start is None:
            return True

        if math.isnan(self.start.x) or math.isnan(self.start.y) or math.isnan(self.start.z):
            return True

        return math.isnan(self.end.x) or math.isnan(self.end.y) or math.isnan(self.end.z)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        assert self.start is not None and self.end is not None, "Cannot hash a TurtleLine with None endpoints."
        return hash((self.start.x, self.start.y, self.start.z, self.end.x, self.end.y, self.end.z))


class TurtleBranch(NodeMixin):
    """A turtle segment with its parent-child branch relationships."""

    def __init__(self, turtle_line: TurtleLine, parent=None):
        super().__init__()

        self.parent = parent
        self.turtle_line = turtle_line

    def __eq__(self, other):
        return self.turtle_line == other.turtle_line

    def __hash__(self):
        return hash(self.turtle_line)


def translate(point:Point, length:float, heading:list[float]) -> Point:
    """Translate a point along a heading vector.

    Parameters
    ----------
    point : Point
        Coordinates before translation.
    length : float
        Translation distance.
    heading : sequence of float
        Three-dimensional heading vector.

    Returns
    -------
    Point
        Translated coordinates.
    """
    x = point.x + length * heading[0]
    y = point.y + length * heading[1]
    z = point.z + length * heading[2]
    return Point(x, y, z)


def l_string_to_turtle_lines(l_string:str, l_config:LSystemConfig, rng:Random) -> list[TurtleLine]:
    """Interpret an expanded L-system as turtle line segments.

    Parameters
    ----------
    l_string : str
        Expanded L-system commands.
    l_config : forsym.tree.config.LSystemConfig
        Turn angles, tropism, and stochastic angle spread.
    rng : random.Random
        Local random-number generator used for stochastic turn angles.

    Returns
    -------
    list of TurtleLine
        Ordered line segments used to construct the tree hierarchy.
    """

    def _to_radians(angle):
        return angle * math.pi / 180

    def _parse_angle(_command):
        turn_angle = parser.extract_values(token=_command)
        return _to_radians(turn_angle[0])

    stack = []

    theta_r = _to_radians(l_config.initial_angle)
    sigma_r = _to_radians(l_config.angle_std)

    init_hlu = ([0, 0, 1], [-math.sin(theta_r), -math.cos(theta_r), 0], [-math.cos(theta_r), math.sin(theta_r), 0])

    t_line = TurtleLine(start=Point(0, 0, 0), hlu=init_hlu, width=1.0)
    l_tokens = parser.tokenize(l_string)

    t_lines = []
    tropism = [component * l_config.bending for component in l_config.tropism]

    for command in l_tokens:
        start, hlu, line_width = t_line.start, t_line.hlu, t_line.width

        if command.startswith("F"):
            seg_lens = parser.extract_values(token=command)
            t_line.end = translate(point=start, length=seg_lens[0], heading=hlu[0])
            t_line.hlu = steering.apply_tropism(hlu, tropism=tropism)

        # Page 46, section 1.10.3 Turtle interpretation of parametric words
        elif "(" in command and parser.extract_operand(succ=command) == "+({})":
            alpha = _parse_angle(_command=command)
            hlu = steering.rotate_around_u(rng.gauss(alpha, sigma_r), hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif "(" in command and parser.extract_operand(succ=command) == "&({})":
            gamma = _parse_angle(_command=command)
            hlu = steering.rotate_around_l(rng.gauss(gamma, sigma_r), hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif "(" in command and parser.extract_operand(succ=command) == "/({})":
            phi = _parse_angle(_command=command)
            hlu = steering.rotate_around_h(rng.gauss(phi, sigma_r), hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif command == "[":
            point = t_line.end if t_line.end is not None else t_line.start
            split_t_line = TurtleLine(start=point, hlu=hlu, width=line_width)
            stack.append(split_t_line)
            t_lines.append(t_line)
            t_line = split_t_line

        elif command == "]":
            t_line = stack.pop()

        elif "(" in command and parser.extract_operand(succ=command) == "N({})":
            line_width = parser.extract_values(token=command)[0]
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif command == "$":
            hlu = steering.keep_l_horizontal(hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

    return t_lines


def turtle_lines_to_branches(turtle_lines: list[TurtleLine])-> TurtleBranch:
    """Build a branch hierarchy from connected turtle lines.

    Parameters
    ----------
    turtle_lines : list of TurtleLine
        Ordered line segments whose endpoints define parent relationships.

    Returns
    -------
    TurtleBranch
        Root of the resulting branch hierarchy.

    Raises
    ------
    IndexError
        If ``turtle_lines`` is empty.
    KeyError
        If a line does not begin at an existing branch endpoint.
    """

    t_root = TurtleBranch(turtle_line=turtle_lines[0])
    branch_store_by_line = {turtle_lines[0]: t_root}
    branch_store_by_end = {turtle_lines[0].end: t_root}

    for i in range(1, len(turtle_lines)):
        if turtle_lines[i].is_nan():
            continue
        if turtle_lines[i] not in branch_store_by_line:
            parent = branch_store_by_end[turtle_lines[i].start]
            branch = TurtleBranch(turtle_line=turtle_lines[i], parent=parent)
            branch_store_by_line[turtle_lines[i]] = branch
            branch_store_by_end[turtle_lines[i].end] = branch

    return t_root
