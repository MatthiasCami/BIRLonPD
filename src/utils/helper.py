def binary_to_choice(binary):
    if (binary == 0):
        return "DEFECT"
    elif (binary == 1):
        return "COOPERATE"
    else:
        raise Exception("THIS IS NOT A VALID CHOICE")
