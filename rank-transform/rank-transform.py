def rank_transform(values: list) -> list:
    """
    Returns the one-based average rank of every value.
    """
    sorted_list = sorted(values)
    result = []
    # Get the index of the sorted consecutive value 
    for x in values :
        # test the successive values as one 
        # for the consecutive values take the mean for each value 
        index_liste = [i+1 for i, val in enumerate(sorted_list) if val == x]
        moyenne = sum(index_liste)/len(index_liste)
        result.append(moyenne)
    return result