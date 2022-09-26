import pickle
file_name = "/Users/ulloacastillo/Desktop/rust/motivus-random-forest/random_forest.pickle"
objects = []
with (open(file_name, "rb")) as f:
    while True:
        try:
            objects.append(pickle.load(f))
        except EOFError:
            break


def fn(a=None, *params):
    print(params)


fn(1, 2)
