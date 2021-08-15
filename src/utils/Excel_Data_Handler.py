import pandas as pd

def get_experiment(groupnumber, filenumber, treatment):
    Dataframe = None
    if (treatment == 'Control'):
        Dataframe = pd.read_csv('./Control{}.csv'.format(filenumber))
        print("Data file currently being used is " + './Control{}.csv'.format(filenumber))
    elif(treatment == 'Changed'):
        Dataframe = pd.read_csv('./Changed{}.csv'.format(filenumber))
        print("Data file currently being used is " + './Changed{}.csv'.format(filenumber))

    print("The group number is " + str(groupnumber))

    Dataframe = Dataframe.loc[Dataframe['Group'] == groupnumber]
    Participants = Dataframe['Subject'].unique()

    Player_moves = {}
    Player_moves_real = {}

    for player_number in Participants:
        key = 'Player_{}'.format(player_number)
        array = Dataframe.loc[Dataframe['Subject'] == player_number]['Action_real'].copy()
        array_unchanged = None

        if (treatment == 'Changed'):
            array_unchanged = Dataframe.loc[Dataframe['Subject'] == player_number]['Action_real'].copy()
            #Next arrays use ncoop and nreal to change the array with actions based on what they actually get to see (agent should have the same info)
            ncoop_array = Dataframe.loc[Dataframe['Subject'] == player_number]['N_coop'].copy()
            nreal_array = Dataframe.loc[Dataframe['Subject'] == player_number]['N_coop_real'].copy()

            for i in range(len(ncoop_array)):
                if (ncoop_array.iloc[i] == 0):
                    if (nreal_array.iloc[i] == 1):
                        array.iloc[i] = 0
                    elif (nreal_array.iloc[i] == 2):
                        array.iloc[i] = 0

                elif (ncoop_array.iloc[i] == 2):
                    if (nreal_array.iloc[i] == 0):
                        array.iloc[i] = 1
                    elif(nreal_array.iloc[i] == 1):
                        array.iloc[i] = 0
        elif (treatment == "Control"):
            array_unchanged = array
        else:
            print("THIS IS AN UNVALID TYPE OF TREATMENT")

        Player_moves[key] = array
        Player_moves_real[key] = array_unchanged

    return Player_moves, Player_moves_real

if __name__ == '__main__':
    print("THIS IS A TEST")