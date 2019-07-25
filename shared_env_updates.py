import numpy as np

def init_plants(n, bounds=None):
    if not bounds:
        bounds = [range(0,7), range(0,7)]
    plant_dict = {}
    placed = 0
    while placed < n:
        x = np.random.randint(low=list(bounds[0])[0],
                              high=list(bounds[0])[-1])
        y = np.random.randint(low=list(bounds[1])[0],
                              high=list(bounds[1])[-1])
        if (x, y) not in plant_dict.keys():
            plant_dict[(x, y)] = 5
            placed += 1

    return plant_dict


def step_vegetation(plants, bounds=None):
    if not bounds:
        bounds = [range(0,7), range(0,7)]
    plants_updated = {}
    for plant, val in plants.items():
        plants[plant] = min(5, val + 0.5)
        if np.random.random() < 0.1:
            x = np.random.randint(low=list(bounds[0])[0],
                                  high=list(bounds[0])[-1])
            y = np.random.randint(low=list(bounds[1])[0],
                                  high=list(bounds[1])[-1])
            while (x, y) in plants.keys():
                x = np.random.randint(low=list(bounds[0])[0],
                                      high=list(bounds[0])[-1])
                y = np.random.randint(low=list(bounds[1])[0],
                                      high=list(bounds[1])[-1])
            plants_updated[(x, y)] = 5
        else:
            plants_updated[plant] = plants[plant]

    return plants_updated
