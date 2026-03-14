def shared_edge(room1, room2):
    if room1.polygon is None or room2.polygon is None:
        return None
    
    p1 = room1.polygon
    p2 = room2.polygon

    edges1 = list(zip(p1, p1[1:] + [p1[0]]))
    edges2 = list(zip(p2, p2[1:] + [p2[0]]))

    for e1 in edges1:
        for e2 in edges2:
            if is_collinear_overlap(e1, e2):
                return e1, e2
    return None


def is_collinear_overlap(e1, e2):
    (x1, y1), (x2, y2) = e1
    (x3, y3), (x4, y4) = e2

    # Vertical edges
    if x1 == x2 == x3 == x4:
        y_range1 = sorted([y1, y2])
        y_range2 = sorted([y3, y4])
        return max(y_range1[0], y_range2[0]) < min(y_range1[1], y_range2[1])

    # Horizontal edges
    if y1 == y2 == y3 == y4:
        x_range1 = sorted([x1, x2])
        x_range2 = sorted([x3, x4])
        return max(x_range1[0], x_range2[0]) < min(x_range1[1], x_range2[1])

    return False