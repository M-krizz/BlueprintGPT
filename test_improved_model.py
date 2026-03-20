from learned.model.sample import load_model, constrained_sample_layout

# Load improved model
print('Loading improved model...')
model, tok = load_model('learned/model/checkpoints/improved_v1.pt', device='cpu')

# Generate 3BHK layout
spec = {
    'rooms': [
        {'type': 'Bedroom', 'count': 3},
        {'type': 'Bathroom', 'count': 2},
        {'type': 'Kitchen', 'count': 1},
        {'type': 'LivingRoom', 'count': 1},
    ]
}

print('Generating layout...')
rooms = constrained_sample_layout(
    model, tok,
    spec=spec,
    building_type='Residential',
    temperature=0.8,
    device='cpu'
)

print(f'Generated {len(rooms)} rooms:')
for r in rooms:
    w = r.x_max - r.x_min
    h = r.y_max - r.y_min
    cx = (r.x_min + r.x_max) / 2
    cy = (r.y_min + r.y_max) / 2
    print(f'  {r.room_type:15s} size=({w:.3f}, {h:.3f}) center=({cx:.3f}, {cy:.3f})')

# Check coverage
x_min = min(r.x_min for r in rooms)
x_max = max(r.x_max for r in rooms)
y_min = min(r.y_min for r in rooms)
y_max = max(r.y_max for r in rooms)
coverage = (x_max - x_min) * (y_max - y_min)

print(f'\nLayout coverage: {coverage:.2%}')
print(f'Bounding box: ({x_min:.3f}, {y_min:.3f}) to ({x_max:.3f}, {y_max:.3f})')

# Check centroid spread
import math
centroids = [((r.x_min + r.x_max)/2, (r.y_min + r.y_max)/2) for r in rooms]
dists = []
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        dx = centroids[i][0] - centroids[j][0]
        dy = centroids[i][1] - centroids[j][1]
        dists.append(math.sqrt(dx*dx + dy*dy))
avg_dist = sum(dists) / len(dists) if dists else 0
print(f'Average centroid distance: {avg_dist:.3f}')
