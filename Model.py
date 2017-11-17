import numpy as np
#import opencv-python
from challenge2017.lib.dataset import record_reader, get_unique_labels
#!/usr/bin/env python3


# browse filetree and find all data records
records = record_reader("C:/Users/Valio/Desktop/Hackatum_data/train")
all_labels = get_unique_labels(records)

print("Available records {}".format(len(records)))

# check that all labels naming one cathegory are of the same size
print("Labels:")
for key, value in sorted(all_labels.items()):
    print(key)
    print("\tsize: {}x{}".format(value.size[0], value.size[1]))
    print("\tpositions:")
    for p in value.positions:
        print("\t\t{}".format(p))


if __name__ == "__main__":
    pass