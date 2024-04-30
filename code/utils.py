import random

def shuffle_and_split(input_list, split_percent=0.7):
    # Shuffle the input list
    random.shuffle(input_list)
    
    # Calculate the split index based on the split percent
    split_index = int(len(input_list) * split_percent)
    
    # Split the list into two parts
    train = input_list[:split_index]
    val = input_list[split_index:]
    
    return train, val